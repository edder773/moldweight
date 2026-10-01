CURVE_LEN = 128            # 한 샷 곡선 포인트 수 (2,048 → 16개씩 평균)
N_CHANNELS = 2             # [사출압력, 금형 내 압력]
VALUE_MIN, VALUE_MAX = -100.0, 1500.0   # 압력값 허용 범위 (bar). 밖이면 422

RMSE_GATE = 0.3            # 배포 게이트: 검증 RMSE ≤ 0.3 g
DRIFT_THRESHOLD = 0.3      # 드리프트: 최근 21샷 RMSE > 0.3 g
WINDOW_SIZE = 21           # 드리프트 판정 윈도우 (실측 있는 샷 수)
RETRAIN_SHOTS = 41         # 재학습에 쓰는 최근 샷 수
REGRESSION_MARGIN = 0.05   # 새 모델 RMSE ≤ 기존 Production RMSE + 0.05 g
BASELINE_IMPROVE = 0.5     # 첫 배포만: 평균 예측보다 50% 이상 개선

BASE_EPOCHS = 50
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-3        # 예비 실험에서 3/3 통과한 값 (스켈레톤 기본 1e-4 아님)
SEED = 42

MODEL_NAME = "MoldWeight_Predictor"
SCALER_PATH = "serving_app/models/scaler.pkl"
