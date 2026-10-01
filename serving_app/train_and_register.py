"""
MLflow로 무게 예측 LSTM을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
드리프트 감지 후에는 Production 가중치에서 이어서 학습하는 fine-tuning 재학습을 담당합니다.

시나리오:
    1) sample_moldweight.csv로 base 모델 학습 -> 검증 RMSE 확인
    2) 게이트 통과 시 Production으로 승격
    3) 드리프트 감지 시 Production 가중치에서 warm-start -> 최근 RETRAIN_SHOTS(41)샷으로
       FINE_TUNE_EPOCHS(10)만 fine-tuning (처음부터 다시 학습하지 않음 - 41샷으로
       스크래치 학습은 불안정)

실행:
    python scripts/train_baseline_v1.py        # 최초 1회 (scaler.pkl 생성)
    python serving_app/train_and_register.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import load_rows, build_sequences, train_test_split, CurveScaler
from serving_app.config import (
    BASE_EPOCHS,
    BASELINE_IMPROVE,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    MODEL_NAME,
    REGRESSION_MARGIN,
    RMSE_GATE,
    SCALER_PATH,
    SEED,
)
from serving_app.lstm_model import build_model

# 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 RMSE가 흔들려 게이트
# 통과 여부가 운에 좌우됩니다. numpy/tensorflow/python random을 한 번에 고정합니다.
keras.utils.set_random_seed(SEED)

DEFAULT_CSV = "data/sample_moldweight.csv"

# fine_tune 평가 분할: 41샷 중 뒤 FINE_TUNE_TEST_SHOTS개로 평가하고 나머지로 학습합니다.
# 드리프트는 최근 샷에서 진행 중이므로, 가장 최근 구간으로 평가해야 "지금 쓸 수 있는
# 모델인가"를 재게 됩니다. 앞쪽으로 평가하면 이미 적응한 구간을 재어 과대평가됩니다.
FINE_TUNE_TEST_SHOTS = 9

# 기존 학습 분포(실험 15·20)에서의 성능을 재는 고정 검증셋. base 학습의 test 몫입니다.
#
# 게이트에는 쓰지 않고 forgetting 지표로만 기록합니다. 드리프트는 공정 조건이 바뀐 것이고
# 앞으로 들어올 데이터는 새 조건이므로, "옛 분포에서 나빠졌다"는 사실만으로 배포를 막으면
# 정당한 적응까지 차단됩니다. 실측으로 fine-tune 후 이 값이 0.148 -> 0.405로 악화되는 것을
# 확인했습니다(warm-start fine-tuning의 전형적인 트레이드오프). 대신 재학습이 기존 지식을
# 얼마나 잃었는지를 MLflow에 남겨 사후 분석과 기획서 근거로 씁니다.
HOLDOUT_CSV = DEFAULT_CSV
_holdout_cache: tuple | None = None


def _holdout(scaler: CurveScaler):
    """고정 검증셋(base 학습의 test 몫)을 돌려줍니다. 파일을 매번 읽지 않도록 캐시합니다."""
    global _holdout_cache
    if _holdout_cache is None:
        rows = load_rows(HOLDOUT_CSV)
        X, y = build_sequences(rows, scaler)
        _, _, X_test, y_test = train_test_split(X, y, rows)
        _holdout_cache = (np.array(X_test, dtype="float32"), y_test)
    return _holdout_cache


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _prepare(rows: list[dict], scaler: CurveScaler, test_ratio: float | None = None):
    """rows를 학습용 배열로 변환합니다. test_ratio가 None이면 실험별 층화 분할을 씁니다."""
    X, y = build_sequences(rows, scaler)
    if test_ratio is None:
        X_train, y_train, X_test, y_test = train_test_split(X, y, rows)
    else:
        split_at = int(len(X) * (1 - test_ratio))
        X_train, y_train = X[:split_at], y[:split_at]
        X_test, y_test = X[split_at:], y[split_at:]

    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_weight(v) for v in y_train], dtype="float32")
    return X_train, y_train_scaled, X_test, y_test


def _current_production() -> tuple[str | None, float | None]:
    """
    현재 Production 버전과 그 학습 당시 기록된 RMSE를 돌려줍니다. 없으면 (None, None).

    여기서 돌려주는 RMSE는 그 모델이 학습될 때의 검증 데이터에서 잰 값이라, 지금 들어오는
    데이터와는 분포가 다를 수 있습니다. fine_tune은 이 값을 쓰지 않고 기존 모델을 현재
    평가셋으로 직접 다시 재서 비교합니다(_score_existing 참고).
    """
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except Exception:
        return None, None
    if not versions:
        return None, None

    mv = versions[0]
    prev_rmse = None
    if mv.run_id:
        try:
            prev_rmse = client.get_run(mv.run_id).data.metrics.get("rmse")
        except Exception:
            prev_rmse = None
    return str(mv.version), prev_rmse


def _evaluate_gates(
    score: float,
    holdout_score: float,
    y_eval,
    prev_version: str | None,
    prev_rmse: float | None,
) -> tuple[bool, str]:
    """
    배포 게이트를 판정합니다. 자동 재학습의 진짜 위험은 "자동으로 더 나쁜 모델을
    배포하는 것"이라, 통과 조건을 모두 만족해야만 승격합니다.

    - 절대 성능: score <= RMSE_GATE                     (지금 현장에서 쓸 수 있는가)
    - 첫 배포 한정: 평균 예측 대비 BASELINE_IMPROVE 이상 개선
    - 회귀 방지: score <= prev_rmse + REGRESSION_MARGIN  (2번째 배포 이후)

    회귀 방지의 prev_rmse는 기존 Production 모델을 **지금과 같은 평가셋**으로 다시 잰
    값입니다. 두 모델을 같은 데이터에서 비교해야 "앞으로 서빙할 데이터에서 새 모델이 더
    나은가"라는 질문에 답할 수 있습니다. 서로 다른 데이터에서 잰 값을 비교하면 판정이
    무의미해집니다.

    holdout_score는 판정에 쓰지 않고 forgetting 기록용으로만 받습니다(HOLDOUT_CSV 주석 참고).

    반환: (통과 여부, 사유 문자열)
    """
    # NaN/inf를 가장 먼저 거릅니다. NaN은 모든 비교가 False라 아래의 `score > RMSE_GATE`를
    # 그대로 통과해 버리고, 그러면 학습이 발산한 모델이 Production을 교체합니다.
    # (가중치가 NaN이 되면 예측도 NaN이 되고 RMSE도 NaN입니다.)
    for name, value in (("rmse", score), ("holdout_rmse", holdout_score)):
        if not math.isfinite(value):
            return False, f"{name}가 유한한 값이 아님 ({value}) - 학습 발산 가능"

    if score > RMSE_GATE:
        return False, f"절대 성능 미달 rmse={score:.3f} > {RMSE_GATE}"

    if prev_version is None:
        # 첫 배포: "평균만 답해도 이 정도"라는 하한을 넘는지 확인합니다.
        mean_pred = float(np.mean(y_eval))
        baseline = rmse(y_eval, [mean_pred] * len(y_eval))
        required = baseline * (1 - BASELINE_IMPROVE)
        if score > required:
            return False, (
                f"첫 배포 개선폭 미달 rmse={score:.3f} > {required:.3f} "
                f"(평균예측 {baseline:.3f}의 {1 - BASELINE_IMPROVE:.0%})"
            )
        return True, f"첫 배포 rmse={score:.3f} (평균예측 {baseline:.3f} 대비 개선)"

    # prev_rmse가 NaN이면 비교 자체가 무의미하므로(어떤 비교도 False) 회귀 방지를 건너뛰지 않고
    # 막습니다. 기존 Production이 발산한 모델이라는 뜻이라 사람이 봐야 합니다.
    if prev_rmse is not None and not math.isfinite(prev_rmse):
        return False, f"기존 v{prev_version}의 rmse가 유한한 값이 아님 ({prev_rmse}) - 수동 확인 필요"

    if prev_rmse is not None and score > prev_rmse + REGRESSION_MARGIN:
        return False, (
            f"회귀 방지 위반 rmse={score:.3f} > 같은 평가셋에서 기존 v{prev_version} "
            f"{prev_rmse:.3f} + {REGRESSION_MARGIN}"
        )
    prev_text = "n/a" if prev_rmse is None else f"{prev_rmse:.3f}"
    return True, (
        f"rmse={score:.3f} (같은 평가셋 기존 v{prev_version} {prev_text}) "
        f"holdout={holdout_score:.3f}"
    )


def _register_if_gate_passed(
    run_id: str,
    score: float,
    holdout_score: float,
    y_eval,
    prev_rmse_on_eval: float | None = None,
) -> dict:
    """
    게이트를 판정하고, 통과하면 Registry에 등록 후 Production으로 승격합니다.

    prev_rmse_on_eval: 기존 Production 모델을 지금과 같은 평가셋으로 잰 RMSE.
        fine_tune이 넘깁니다. None이면 Registry에 기록된 값을 대신 씁니다
        (base 학습처럼 비교 가능한 재측정이 없는 경우).
    """
    prev_version, prev_logged = _current_production()
    prev_rmse = prev_rmse_on_eval if prev_rmse_on_eval is not None else prev_logged
    passed, reason = _evaluate_gates(score, holdout_score, y_eval, prev_version, prev_rmse)

    result = {
        "run_id": run_id,
        "rmse": score,
        "holdout_rmse": holdout_score,
        "promoted": False,
        "version": None,
        "prev_version": prev_version,
        "prev_rmse": prev_rmse,
    }

    if not passed:
        print(f"[GATE FAILED] {reason} -> 배포 차단, 기존 Production 유지")
        return result

    v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    # archive_existing_versions=True: 이전 Production을 Archived로 내립니다. 기본값(False)이면
    # 승격만 되고 예전 버전이 Production에 남아 Production 버전이 여러 개가 됩니다.
    MlflowClient().transition_model_version_stage(
        name=MODEL_NAME,
        version=v.version,
        stage="Production",
        archive_existing_versions=True,
    )
    result["promoted"] = True
    result["version"] = str(v.version)
    print(f"[GATE PASSED] {reason} -> {MODEL_NAME} v{v.version} promoted to Production")
    return result


def train_and_register(csv_path: str | None = None, rows: list[dict] | None = None) -> dict:
    """처음부터(scratch) 학습합니다. 데이터가 충분한 base 학습에서만 사용합니다."""
    if rows is None:
        rows = load_rows(csv_path or DEFAULT_CSV)
    scaler = CurveScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    with mlflow.start_run(run_name="base-train") as run:
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

        preds = [scaler.inverse_weight(p) for p in model.predict(X_test, verbose=0).flatten()]
        score = rmse(y_test, preds)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_param("n_shots", len(rows))
        mlflow.log_metric("rmse", score)
        # base 학습의 검증셋이 곧 고정 검증셋이므로 같은 값입니다. 뒤따르는 fine_tune이
        # 회귀 비교 대상으로 읽을 수 있도록 같은 이름으로도 남깁니다.
        mlflow.log_metric("holdout_rmse", score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(run.info.run_id, score, score, y_test)


def fine_tune(rows: list[dict]) -> dict:
    """
    현재 Production 모델 가중치에서 이어서(warm start), 넘겨받은 최근 샷으로 짧게
    fine-tuning합니다. 41샷처럼 적을 때도 스크래치 학습보다 훨씬 안정적입니다.

    rows: RULES 3-3 형식의 최근 샷 (D가 recent_shots를 변환해 넘깁니다)
    반환: {"promoted", "rmse", "holdout_rmse", "version", "prev_version", "prev_rmse"}
          rmse       - 새 모델을 최근 평가셋에서 잰 값 (게이트 판정 기준)
          prev_rmse  - 기존 Production 모델을 같은 평가셋에서 잰 값 (회귀 비교 대상)
          holdout_rmse - 기존 학습 분포에서 잰 값 (forgetting 기록용, 판정에는 쓰지 않음)

    스케일러는 다시 fit하지 않습니다. 드리프트 데이터는 무게가 학습 범위 아래로
    내려가 scale_weight가 음수를 반환하는데, 그 값을 그대로 학습해야 모델이 새 구간으로
    이동할 수 있습니다.
    """
    scaler = CurveScaler.load(SCALER_PATH)
    test_ratio = min(0.5, FINE_TUNE_TEST_SHOTS / max(len(rows), 1))
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler, test_ratio=test_ratio)

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")

    # 회귀 방지 게이트의 비교 기준: 지금 Production 모델이 바로 이 평가셋에서 내는 성적.
    # fine-tuning 전에 재야 하므로 여기서 먼저 측정합니다.
    prev_preds = [scaler.inverse_weight(p) for p in model.predict(X_test, verbose=0).flatten()]
    prev_rmse_on_eval = rmse(y_test, prev_preds)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune") as run:
        model.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, verbose=0)

        preds = [scaler.inverse_weight(p) for p in model.predict(X_test, verbose=0).flatten()]
        score = rmse(y_test, preds)

        # 회귀 판정용: 기존 Production과 같은 데이터에서 다시 잰 값
        X_hold, y_hold = _holdout(scaler)
        hold_preds = [scaler.inverse_weight(p) for p in model.predict(X_hold, verbose=0).flatten()]
        holdout_score = rmse(y_hold, hold_preds)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("lr", FINE_TUNE_LR)
        mlflow.log_param("n_shots", len(rows))
        mlflow.log_param("n_eval_shots", len(y_test))
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("holdout_rmse", holdout_score)
        # 같은 평가셋에서 기존 모델이 낸 성적. 재학습이 실제로 개선이었는지 사후에 확인할 수
        # 있도록 남깁니다.
        mlflow.log_metric("prev_rmse_on_eval", prev_rmse_on_eval)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(
            run.info.run_id, score, holdout_score, y_test, prev_rmse_on_eval=prev_rmse_on_eval
        )


if __name__ == "__main__":
    train_and_register(sys.argv[1] if len(sys.argv) > 1 else None)
