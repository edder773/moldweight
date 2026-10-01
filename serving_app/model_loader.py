"""
모델 로딩 전략과 모델 소스를 한 곳에서 감싸는 모듈.

로딩 모드: Lazy Loading vs Eager Loading을 비교합니다. LSTM은 로딩 자체가 무거워
서버 시작 시간 / 첫 요청 응답 시간 차이가 뚜렷하게 드러납니다.

모델 소스: MODEL_SOURCE=mlflow 로 전환하면 로컬 .keras 파일 대신 MLflow Model
Registry의 Production 버전을 로드합니다. main.py / train_and_register.py는 그대로 두고
이 파일만 손대면 되도록 설계되어 있습니다 - 이것이 "조립 블록" 구조입니다.

스케일러(scaler.pkl)는 MODEL_SOURCE와 무관하게 항상 로컬 파일에서 로드합니다.
정규화 기준이 바뀌면 이미 그 기준으로 학습된 가중치와 어긋나기 때문입니다.

환경변수
    LOADING_MODE = lazy(기본값) | eager
    MODEL_SOURCE = local(기본값) | mlflow
    MLFLOW_TRACKING_URI = MODEL_SOURCE=mlflow 일 때 필요
"""
import os
import time

from data.features import CurveScaler
from serving_app.config import MODEL_NAME, SCALER_PATH

LOCAL_MODEL_PATH = "serving_app/models/moldweight_v1.keras"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"

_model_cache = None  # Lazy Loading 캐시


class LoadedModel:
    """local .keras와 mlflow 두 소스를 동일한 인터페이스로 감싸는 래퍼."""

    def __init__(self, keras_model, scaler: CurveScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, curve: list[list[float]]) -> float:
        """
        curve: 샷 하나의 압력 곡선 128×2. 채널 순서는 [사출압력, 금형 내 압력].
        반환: 예측 무게 (g).
        """
        import numpy as np

        scaled = self.scaler.transform_curve(curve)
        x = np.array([scaled], dtype="float32")  # (1, 128, 2)
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])
        return self.scaler.inverse_weight(pred_scaled)

    def predict_many(self, curves: list[list[list[float]]]) -> list[float]:
        """곡선 여러 개를 한 번에 예측합니다. 배치 테스트에서 샷마다 predict를 부르는
        것보다 훨씬 빠릅니다 (keras predict 호출 오버헤드를 한 번만 냅니다)."""
        import numpy as np

        if not curves:
            return []
        x = np.array([self.scaler.transform_curve(c) for c in curves], dtype="float32")
        preds = self._keras_model.predict(x, verbose=0).flatten()
        return [self.scaler.inverse_weight(float(p)) for p in preds]


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = CurveScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    """Registry의 Production 버전을 로드합니다. 스케일러는 항상 로컬 파일에서 읽습니다."""
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)
    scaler = CurveScaler.load(SCALER_PATH)

    # 몇 번째 버전이 서빙 중인지 /predict 응답(model_version)에 실어 보내기 위해 조회합니다.
    versions = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
    version = versions[0].version if versions else "production"
    return LoadedModel(keras_model=keras_model, scaler=scaler, version=str(version))


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager() -> LoadedModel:
    """Eager Loading: 서버 시작 시점에 즉시 모델을 로드한다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 로드하고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache


def reload() -> None:
    """
    재학습으로 Production 버전이 바뀐 뒤, 캐시에 남은 옛 모델이 계속 서빙되지 않도록
    교체합니다. 재학습이 끝난 뒤 C가 호출합니다 (RULES 5장: recent_predictions를 비울 때 함께).

    교체 방식은 LOADING_MODE를 따릅니다. 로딩 비용을 "언제, 누가" 치를지가 이 환경변수로
    선언된 운영 방침이고, 재학습은 그 방침에서 예외가 아니기 때문입니다.

    - eager: 여기서 즉시 새 모델을 올립니다. eager를 켠 이유가 "사용자는 로딩 비용을 내지
      않는다"인데, 캐시만 비우면 그 보장이 재학습마다 깨집니다. 게다가 재학습은 드리프트가
      감지된 시점 - 예측이 가장 중요한 때 - 에 일어납니다.
      또 /health의 model_loaded가 false로 떨어지는 구간이 사라집니다. 그 구간이 있으면
      scripts/smoke_test.sh(model_loaded=true를 요구)와 컨테이너 헬스체크가 재학습 직후
      실패합니다. "서버가 떠 있다"와 "예측할 준비가 됐다"를 eager 모드에서는 계속
      일치시켜 두는 편이 맞습니다.
    - lazy: 캐시만 비우고 다음 요청이 올리게 합니다. 재학습 직후 응답 경로를 막지 않습니다.
    """
    global _model_cache
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        load_eager()
        return
    _model_cache = None
