"""
실습을 시작하기 전에 1회만 실행하는 부트스트랩 스크립트입니다.

아직 MLflow가 등장하지 않는 단계이므로, FastAPI 서버가 곧바로 로드할 수 있는
"사전 학습된" 로컬 LSTM 모델 파일을 만들어 둡니다. 이 모델은 뒤에서
train_and_register.py(MLflow 버전)로 대체됩니다.

여기서 fit한 스케일러(scaler.pkl)는 이후 모든 단계에서 계속 재사용됩니다 - 서빙 시점
정규화 기준이 실습 내내 바뀌지 않아야 하기 때문입니다. 이 파일은 커밋합니다.

실행:
    python scripts/train_baseline_v1.py                      # data/sample_moldweight.csv 사용
    python scripts/train_baseline_v1.py data/uploads/xxx.csv  # 업로드 파일로 학습

서버는 데이터 없이도 뜨므로(lazy 모드) 먼저 띄워도 됩니다:
    uvicorn serving_app.main:app --reload
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import load_rows, build_sequences, train_test_split, CurveScaler
from serving_app.config import BASE_EPOCHS, RMSE_GATE, SEED, SCALER_PATH
from serving_app.lstm_model import build_model

DEFAULT_CSV = "data/sample_moldweight.csv"
MODEL_PATH = "serving_app/models/moldweight_v1.keras"

# 검증 RMSE가 이 값을 밑돌면 학습 길이가 충분하다고 본 지점 (lstm_model.py 실험 표 참고).
RECOMMENDED_EPOCHS = 200


def rmse(y_true, y_pred) -> float:
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def main():
    import numpy as np
    from tensorflow import keras

    # 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 검증 RMSE가 흔들려
    # 게이트 통과 여부가 운에 좌우됩니다.
    keras.utils.set_random_seed(SEED)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rows = load_rows(csv_path)
    print(f"{csv_path}: {len(rows)}샷 로드")

    scaler = CurveScaler().fit(rows)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    scaler.save(SCALER_PATH)
    print(f"scaler fit -> {SCALER_PATH}")
    print(f"  채널 min/max: inj [{scaler.channel_min[0]:.2f}, {scaler.channel_max[0]:.2f}]  "
          f"cav [{scaler.channel_min[1]:.2f}, {scaler.channel_max[1]:.2f}]")
    print(f"  무게 min/max: [{scaler.weight_min:.3f}, {scaler.weight_max:.3f}] g")

    X, y = build_sequences(rows, scaler)
    X_train, y_train, X_test, y_test = train_test_split(X, y, rows)
    print(f"실험별 층화 시간순 분할: train {len(X_train)}샷 / test {len(X_test)}샷")

    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    # 입력 곡선과 같은 스케일로 학습해야 loss가 과도하게 커지지 않고 안정적으로 수렴합니다.
    y_train_scaled = np.array([scaler.scale_weight(v) for v in y_train], dtype="float32")

    model = build_model()
    print(f"학습 시작: {BASE_EPOCHS} epoch, 파라미터 {model.count_params():,}개")
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = [scaler.inverse_weight(p) for p in preds_scaled]  # 실제 g 단위로 복원
    score = rmse(y_test, preds)
    print(f"baseline v1 검증 RMSE = {score:.3f} g  (배포 게이트: {RMSE_GATE} g)")

    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")

    if score > RMSE_GATE:
        print(f"※ 게이트({RMSE_GATE} g) 미달입니다. 이 로컬 모델은 서빙 확인용이며, "
              "배포 판정은 train_and_register.py에서 MLflow로 다시 정식 검증합니다.")
    if BASE_EPOCHS < RECOMMENDED_EPOCHS:
        print(f"※ config.BASE_EPOCHS={BASE_EPOCHS}입니다. 측정상 {RECOMMENDED_EPOCHS} epoch에서 "
              "검증 RMSE가 최소였고 드리프트 감지 윈도우도 2배였습니다 (lstm_model.py 참고).")


if __name__ == "__main__":
    main()
