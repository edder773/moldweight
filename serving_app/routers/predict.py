"""
예측 엔드포인트 (RULES.md 4장, 5장).

POST /predict            - 곡선 1개(128×2)로 무게 예측
POST /predict/batch-test - 실측 무게가 붙은 샷 여러 개를 예측하고 드리프트 판정
                           (scripts/simulate_drift.py, 대시보드 드리프트 버튼)

곡선 길이·채널 수·값 범위 검증은 serving_app/schemas.py에서 하고, 어긋나면 422입니다.
"""
import os
import threading

from fastapi import APIRouter

from serving_app import model_loader
from serving_app.config import RETRAIN_SHOTS, WINDOW_SIZE
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import (
    BatchTestRequest,
    BatchTestResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()

# RULES 5장 C ↔ D: 매 batch 끝에 두 리스트를 check_and_trigger에 넘긴다.
recent_predictions: list[dict] = []  # [{"predicted", "actual"}], 최근 WINDOW_SIZE개
recent_shots: list[dict] = []        # [{"curve", "weight", "cycle_counter"}], 최근 RETRAIN_SHOTS개
_batch_lock = threading.Lock()       # 두 리스트를 건드리는 batch-test를 한 번에 하나씩


class _MockModel:
    """B의 모델이 오기 전까지 쓰는 가짜 모델 (RULES 9장 "가짜로 먼저")."""

    version = "mock"

    def predict_one(self, curve: list[list[float]]) -> float:
        return 115.16


# 연결 1에서 B의 model_loader로 바꾼다: USE_MOCK_MODEL=0
USE_MOCK_MODEL = os.getenv("USE_MOCK_MODEL", "1") == "1"


def _get_model():
    if USE_MOCK_MODEL:
        return _MockModel()
    return model_loader.get_model()


_CURVE_422 = {
    422: {"description": "곡선 길이가 CURVE_LEN이 아님, 한 점의 값이 N_CHANNELS개가 아님, 값이 VALUE_MIN~VALUE_MAX 밖"}
}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="곡선 1개로 무게 예측",
    responses=_CURVE_422,
)
def predict(req: PredictRequest):
    model = _get_model()
    predicted_weight = model.predict_one(req.curve)
    return PredictResponse(predicted_weight=round(predicted_weight, 2), model_version=model.version)


@router.post(
    "/predict/batch-test",
    response_model=BatchTestResponse,
    summary="실측 무게가 붙은 샷들을 예측하고 드리프트 판정",
    description=(
        "샷마다 예측한 뒤 (예측, 실측)과 곡선·실측 무게를 쌓고, 배치 끝에 최근 WINDOW_SIZE개로 "
        "드리프트를 판정합니다. 드리프트면 최근 RETRAIN_SHOTS개로 재학습합니다."
    ),
    responses={422: {"description": "/predict와 같은 곡선 규칙 위반, 또는 shots가 비어 있음"}},
)
def batch_test(req: BatchTestRequest):
    # 배치 요청이 동시에 와도(대시보드 버튼 + simulate_drift.py) 기록이 섞이거나
    # 재학습이 겹쳐 돌지 않도록 한 번에 하나씩 처리한다
    with _batch_lock:
        return _run_batch(req)


def _run_batch(req: BatchTestRequest) -> BatchTestResponse:
    model = _get_model()
    predictions: list[float] = []

    for shot in req.shots:
        predicted = model.predict_one(shot.curve)
        predictions.append(round(predicted, 2))
        recent_predictions.append({"predicted": predicted, "actual": shot.actual_weight})
        # 재학습용으로 곡선과 실측 무게도 같이 쌓는다
        recent_shots.append(
            {"curve": shot.curve, "weight": shot.actual_weight, "cycle_counter": shot.cycle_counter}
        )

    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]
    recent_shots[:] = recent_shots[-RETRAIN_SHOTS:]

    drift_check = check_and_trigger(recent_predictions, recent_shots)

    # 재학습이 끝나면(통과든 실패든) 새 모델 기준으로 다시 WINDOW_SIZE개를 모은다
    if drift_check["status"] == "retrain_triggered":
        recent_predictions.clear()

    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
