# runtime: 모델 준비 없이 서버 기동 확인. trained: 실제 학습·게이트·Registry 검증.
# 멀티 아키텍처 digest를 고정해 태그 갱신으로 의존성 캐시가 무효화되지 않게 합니다.
FROM python:3.11-slim@sha256:9f6ef439f51f4b36dc5c6bb265c2d3dd810bde94c5bbec77ae86b6650c488cf9 AS dependencies

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    MODEL_SOURCE=local \
    LOADING_MODE=lazy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
# 서버 기동 확인. 실제 모델 준비·예측 검증은 smoke_test.sh에서 수행합니다.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); assert json.load(r).get('status') == 'ok'"
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 학습에 필요한 파일만 복사합니다. 화면·라우터·smoke 수정으로 재학습하지 않습니다.
FROM dependencies AS model-build
ENV MODEL_SOURCE=mlflow \
    LOADING_MODE=eager
COPY serving_app/__init__.py serving_app/config.py serving_app/lstm_model.py serving_app/model_loader.py serving_app/train_and_register.py serving_app/
COPY data/__init__.py data/features.py data/sample_moldweight.csv data/
COPY scripts/train_baseline_v1.py scripts/
# 학습 구간 스케일러만 준비한 뒤 200 epoch 학습은 MLflow에서 한 번 수행합니다.
# 게이트·Production 로딩 실패 시 빌드 실패. local 소스용 .keras도 같은 모델로 보존합니다.
RUN mkdir -p data/uploads logs mlruns \
 && cp data/sample_moldweight.csv data/uploads/moldweight_build_seed.csv \
 && python scripts/train_baseline_v1.py --scaler-only \
 && python serving_app/train_and_register.py \
 && python -c "import mlflow.tensorflow; from serving_app.model_loader import MLFLOW_MODEL_URI, LOCAL_MODEL_PATH, load_eager; mlflow.tensorflow.load_model(MLFLOW_MODEL_URI).save(LOCAL_MODEL_PATH); load_eager()"

FROM dependencies AS runtime
COPY serving_app/*.py serving_app/
COPY serving_app/routers/ serving_app/routers/
COPY serving_app/monitoring/ serving_app/monitoring/
COPY serving_app/models/.gitkeep serving_app/models/*.pkl serving_app/models/
# 원본 HDF5와 개발 중 업로드 파일은 제외합니다.
COPY data/__init__.py data/features.py data/storage.py data/sample_moldweight.csv data/drift_shots_23.csv data/
COPY scripts/train_baseline_v1.py scripts/simulate_drift.py scripts/smoke_test.sh scripts/
RUN mkdir -p data/uploads logs mlruns \
 && chmod +x scripts/smoke_test.sh
COPY serving_app/static/ serving_app/static/

FROM runtime AS trained
ENV MODEL_SOURCE=mlflow \
    LOADING_MODE=eager
COPY --from=model-build /app/mlflow.db /app/mlflow.db
COPY --from=model-build /app/mlruns/ /app/mlruns/
COPY --from=model-build /app/serving_app/models/ /app/serving_app/models/
COPY --from=model-build /app/data/uploads/moldweight_build_seed.csv /app/data/uploads/
