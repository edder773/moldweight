# runtime: B/C 구현 전 서버 기동 확인용. 모델 로딩은 첫 예측까지 미룹니다.
# trained: B/C 구현 후 시드 CSV → baseline → Registry → eager 로딩을 검증합니다.
#   docker build --target runtime -f serving_app/Dockerfile -t moldweight-serving:runtime .
#   docker build --target trained -f serving_app/Dockerfile -t moldweight-serving:trained .
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    MODEL_SOURCE=local \
    LOADING_MODE=lazy

# 소스 수정 시 TensorFlow/MLflow 의존성 설치 레이어를 재사용합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving_app/*.py serving_app/
COPY serving_app/routers/ serving_app/routers/
COPY serving_app/monitoring/ serving_app/monitoring/
COPY serving_app/static/ serving_app/static/
# 고정 스케일러만 복사합니다. .gitkeep은 스케일러가 아직 없어도 빌드되게 합니다.
COPY serving_app/models/.gitkeep serving_app/models/*.pkl serving_app/models/
# 원본 HDF5와 개발 중 업로드 파일을 이미지에 넣지 않습니다.
COPY data/__init__.py data/features.py data/storage.py data/sample_moldweight.csv data/drift_shots_23.csv data/
COPY scripts/train_baseline_v1.py scripts/simulate_drift.py scripts/smoke_test.sh scripts/
RUN mkdir -p data/uploads logs mlruns \
 && chmod +x scripts/smoke_test.sh

EXPOSE 8000
# 이 healthcheck는 서버 기동 확인입니다. 모델 준비·예측 검증은 smoke_test.sh가 담당합니다.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); assert json.load(r).get('status') == 'ok'"
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS trained
ENV MODEL_SOURCE=mlflow \
    LOADING_MODE=eager
# B의 데이터·스케일러·학습·로더 구현이 합쳐진 뒤 사용합니다.
# 게이트 미통과/Production 부재/로더 미구현이면 이미지 빌드를 실패시킵니다.
RUN cp data/sample_moldweight.csv data/uploads/moldweight_build_seed.csv \
 && python scripts/train_baseline_v1.py \
 && python serving_app/train_and_register.py \
 && python -c "from serving_app import model_loader; model_loader.load_eager()"
