"""드리프트 감지 결과에 따라 MoldWeight 모델 재학습을 실행한다."""

import logging

from serving_app.config import DRIFT_THRESHOLD, MODEL_NAME, RETRAIN_SHOTS, WINDOW_SIZE
from serving_app.monitoring.drift_detector import compute_rmse

logger = logging.getLogger("aiops")


def check_and_trigger(recent_predictions: list[dict], recent_shots: list[dict]) -> dict:
    """최근 예측을 검사하고 필요하면 최근 샷으로 fine-tuning을 실행한다."""
    if len(recent_predictions) < WINDOW_SIZE:
        return {
            "status": "collecting",
            "rmse": None,
            "window": len(recent_predictions),
        }

    rolling_rmse = compute_rmse(recent_predictions)
    if rolling_rmse <= DRIFT_THRESHOLD:
        return {
            "status": "ok",
            "rmse": rolling_rmse,
            "window": WINDOW_SIZE,
        }

    logger.warning(
        "[WARN] drift detected rmse=%.3f window=%s threshold=%s",
        rolling_rmse,
        WINDOW_SIZE,
        DRIFT_THRESHOLD,
    )

    rows = recent_shots[-RETRAIN_SHOTS:]
    logger.info("[INFO] retrain triggered shots=%s", len(rows))

    # B PR #6이 병합되기 전에도 collecting/ok 경로는 정상 동작하도록 지연 import한다.
    from serving_app import model_loader
    from serving_app.train_and_register import fine_tune

    result = fine_tune(rows)
    if result["promoted"]:
        model_loader.reload()
        model_version = result["version"]
        logger.info(
            "[OK] gate passed rmse=%.3f -> %s v%s promoted",
            result["rmse"],
            MODEL_NAME,
            model_version,
        )
    else:
        model_version = result["prev_version"]
        logger.warning(
            "[FAIL] gate failed rmse=%.3f -> keep %s v%s",
            result["rmse"],
            MODEL_NAME,
            model_version,
        )

    return {
        "status": "retrain_triggered",
        "rmse": rolling_rmse,
        "window": WINDOW_SIZE,
        "promoted": result["promoted"],
        "model_version": model_version,
    }
