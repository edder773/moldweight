"""
FastAPI 요청/응답 Pydantic 스키마 (RULES.md 4장 API 형식).

한 샷은 길이 CURVE_LEN(128)인 곡선이고, 한 점은 [사출압력, 금형 내 압력] 두 값입니다.
서빙 시점 입력이 학습 시점 입력(data/features.py의 128×2)과 어긋나지 않도록
길이·채널 수·값 범위를 스키마 단에서 강제합니다. 어긋나면 FastAPI가 422를 돌려줍니다.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from serving_app.config import CURVE_LEN, N_CHANNELS, VALUE_MAX, VALUE_MIN

# 압력값 한 개: VALUE_MIN~VALUE_MAX(bar), NaN/inf와 문자열·불리언은 받지 않음
PressureValue = Annotated[
    float, Field(ge=VALUE_MIN, le=VALUE_MAX, strict=True, allow_inf_nan=False)
]
# 곡선의 한 점: [사출압력, 금형 내 압력]
CurvePoint = Annotated[
    list[PressureValue], Field(min_length=N_CHANNELS, max_length=N_CHANNELS)
]
# 한 샷의 곡선: CURVE_LEN개 점, 시간 순서
Curve = Annotated[
    list[CurvePoint],
    Field(
        min_length=CURVE_LEN,
        max_length=CURVE_LEN,
        description=f"{CURVE_LEN}×{N_CHANNELS} 곡선. 한 점은 [사출압력, 금형 내 압력] (bar)",
    ),
]

# Swagger "Try it out"에서 그대로 보내도 200이 나오는 예시 곡선 (CURVE_LEN개 점)
_EXAMPLE_CURVE = [[0.0, 0.0]] * CURVE_LEN


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"curve": _EXAMPLE_CURVE}]})

    curve: Curve


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"predicted_weight": 115.21, "model_version": "2"}]}
    )

    predicted_weight: float = Field(..., description="예측 무게 (g)")
    model_version: str = Field(
        ..., description='MLflow Registry 버전 번호 문자열 ("1", "2" …). 로컬 모델이면 "local"'
    )


class BatchShot(BaseModel):
    cycle_counter: int = Field(..., description="샷 번호 (생산 순서)")
    curve: Curve
    actual_weight: float = Field(..., allow_inf_nan=False, description="실측 무게 (g)")


class BatchTestRequest(BaseModel):
    # scripts/simulate_drift.py, 대시보드 드리프트 버튼이 샷 여러 개를 한 번에 보냄
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"shots": [{"cycle_counter": 20933, "curve": _EXAMPLE_CURVE, "actual_weight": 114.12}]}
            ]
        }
    )

    shots: list[BatchShot] = Field(..., min_length=1, description="실측 무게가 붙은 샷 1개 이상")


class DriftCheck(BaseModel):
    status: Literal["collecting", "ok", "retrain_triggered"] = Field(
        ..., description="collecting: 판정 윈도우를 모으는 중 · ok: 드리프트 없음 · retrain_triggered: 재학습 실행"
    )
    rmse: float | None = Field(None, description="최근 윈도우 RMSE (g). 윈도우가 다 차기 전이면 null")
    window: int = Field(..., description="지금까지 쌓인 예측 개수 (최대 WINDOW_SIZE)")
    promoted: bool | None = Field(None, description="retrain_triggered일 때만: 새 모델이 게이트를 통과했는지")
    model_version: str | None = Field(None, description="retrain_triggered일 때만: 판정 후 Production 버전")

    @model_serializer(mode="wrap")
    def _drop_retrain_fields(self, handler):
        # rmse는 null이어도 내보내고, promoted/model_version은 재학습 때만 내보낸다
        data = handler(self)
        if self.status != "retrain_triggered":
            data.pop("promoted", None)
            data.pop("model_version", None)
        return data


class BatchTestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predictions": [114.55, 114.61],
                    "drift_check": {
                        "status": "retrain_triggered",
                        "rmse": 0.452,
                        "window": 21,
                        "promoted": True,
                        "model_version": "2",
                    },
                }
            ]
        }
    )

    predictions: list[float] = Field(..., description="샷 순서대로 예측 무게 (g)")
    drift_check: DriftCheck
