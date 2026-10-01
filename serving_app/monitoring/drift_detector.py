"""최근 MoldWeight 예측 오차를 이용한 드리프트 판정."""

import math

from serving_app.config import DRIFT_THRESHOLD, WINDOW_SIZE


def compute_rmse(recent_predictions: list[dict]) -> float:
    """최근 ``WINDOW_SIZE``개의 predicted/actual 쌍으로 RMSE를 계산한다."""
    window = recent_predictions[-WINDOW_SIZE:]
    if not window:
        return 0.0

    squared_errors = [
        (prediction["actual"] - prediction["predicted"]) ** 2
        for prediction in window
    ]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def is_drift(recent_predictions: list[dict]) -> bool:
    """윈도우가 찼고 RMSE가 임계값을 초과한 경우에만 드리프트로 판정한다."""
    if len(recent_predictions) < WINDOW_SIZE:
        return False
    return compute_rmse(recent_predictions) > DRIFT_THRESHOLD
