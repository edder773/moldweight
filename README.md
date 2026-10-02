# MoldWeight

사출압력과 금형 내 압력 곡선으로 성형품 무게를 예측하고, 드리프트 감지 → 재학습 → 성능 게이트 → Production 전환을 연결하는 New 4조 팀 프로젝트입니다. 팀 규칙의 이름·숫자·함수 계약은 [RULES.md](RULES.md), 실행 설정은 [serving_app/config.py](serving_app/config.py)를 기준으로 합니다.

## 현재 통합 상태

2026-10-02 기준, PR #1~#14가 main에 반영됐습니다. 통합 기준은 [`52bcf24`](https://github.com/edder773/moldweight/commit/52bcf24936be4312b1d0c7721fc2c570d8cdad42)입니다. 실제 모델·통합 검증에 이어 CI 경량화와 변경 범위별 검증을 적용합니다.

| 영역 | 반영된 내용 |
|---|---|
| 데이터 | HDF5 전처리 스크립트, 실험 15·20의 526샷 CSV, 실험 23의 303샷 CSV |
| 모델 | 128×2 입력 LSTM, 고정 CurveScaler, MLflow 학습·Registry·Production 로딩 |
| API | 실제 모델 예측, 곡선 검증, CSV 업로드, 샷 배치 테스트 |
| 모니터링 | 최근 21샷 RMSE, 41샷 수집 후 fine-tune, 게이트 실패 시 기존 Production 유지 |
| 인프라 | runtime/trained Docker 타깃, 학습 캐시 분리, Compose, smoke·p95·상태 전환 검증, 변경 범위별 CI |
| 대시보드·시연 | CSV 자동 로드, 샷 단위 라인 가동·중지·계속, 2샷마다 실측·드리프트 후 전수검사, 게이트·서빙 버전·알람 표시 |

실제 모델 연결과 모니터링 수정은 반영됐습니다. 예측 API에는 가짜 모델이나 `USE_MOCK_MODEL` 분기가 없습니다. 드리프트 판단용 예측 21개와 재학습용 샷 41개가 모두 모이기 전에는 재학습을 시작하지 않습니다. 배포 게이트는 NaN·무한대 RMSE를 거부하며, 스케일러는 검증 데이터를 제외한 학습 샷으로만 fit합니다.

## 역할과 남은 작업

| 역할 | 담당 | GitHub | 브랜치 | 담당 범위와 후속 작업 |
|---|---|---|---|---|
| A 데이터·기획·제출 | 장인우 | @inwoo-jang | `feat/data` | 데이터·설정·RULES 관리, B의 실측을 받아 RULES 6장과 제출 자료 반영 |
| B 모델·학습 | 이도권 | @dokwon33 | `feat/model` | 모델·스케일러·게이트·Registry, 분할 방식과 검증 수치를 A에게 전달 |
| C 서빙 API | 김선주 | @ssunju-02 | `feat/api` | 스키마·예측·업로드·배치 API, Swagger 명세 전달 |
| D 모니터링·재학습 | 김주은 | @jekim20 | `feat/monitor` | 드리프트·fine-tune 연결·게이트 결과 로그 |
| E 인프라·통합 | 김보석 | @edder773 | `feat/infra` | Docker·PR 통합·smoke·p95·CI, 실제 모델·화면 통합 검증 |
| F 대시보드·시연 | 조영우 | @evermate | `feat/dashboard` | 샷 기반 대시보드·시뮬레이터, 시연 화면 캡처 |

파일별 소유자는 RULES 1장을 따릅니다. 소유자가 아닌 파일의 수정은 해당 담당자에게 요청합니다.

E 통합 검증을 완료했습니다. PR #11의 C+D 호출 경로는 승격 성공 시 reload 1회·실패 시 0회이며, lazy/eager 모드에서 확인했습니다. 이 호출 횟수 검증은 학습과 모델 로딩을 대체하여 수행했고, 아래 드리프트·게이트·버전 전환 검증은 실제 모델로 별도 수행했습니다. 대시보드의 정상 샷 전송, 게이트 실패 시 v1 유지, 게이트 통과 후 v2 전환과 알람을 확인했습니다. PR #14의 중지·초기화·조건 변경 시 이전 실행 취소도 점검했습니다.

현재 후속 작업은 다음과 같습니다.

- **A/B:** RULES 6장의 빈 표에 실제 분할 방식과 검증 RMSE를 반영합니다. 현재 코드의 base 분할은 420/106샷, fine-tune 분할은 최근 41샷 중 32/9샷입니다.
- **C:** 제출 자료용 Swagger 화면·API 명세를 A에게 전달합니다.
- **F:** 현재 샷 단위 화면의 정상 판정·게이트 실패·게이트 통과·버전 전환을 캡처합니다. 중지·초기화 직전의 버전 조회 응답이 늦게 도착하면 이전 버전 표시로 덮어쓸 수 있는 후속 문제도 보완합니다. 실제 Production 전환과 중복 샷 루프에 대한 검증은 통과했습니다.

## 실행

Python 3.11 또는 Docker Compose를 사용하며, 모든 명령은 저장소 루트에서 실행합니다.

### Docker: 실제 모델로 실행

`trained` 타깃은 학습 구간으로 스케일러만 준비하고 MLflow에서 200 epoch 학습·성능 게이트를 한 번 수행합니다. 승인된 Production 모델을 로컬 `.keras`로도 저장합니다. 학습 또는 Production 로딩이 실패하면 빌드도 실패합니다.

```bash
MOLDWEIGHT_BUILD_TARGET=trained MODEL_SOURCE=mlflow LOADING_MODE=eager \
  docker compose -f serving_app/docker-compose.yml up --build -d --wait

docker compose -f serving_app/docker-compose.yml exec -T \
  serving-app bash scripts/smoke_test.sh

docker compose -f serving_app/docker-compose.yml exec -T \
  -e P95_SAMPLES=100 serving-app bash scripts/smoke_test.sh
```

Compose의 기본값은 `runtime/local/lazy`이므로 실제 학습 실행에는 위 세 환경변수를 함께 지정합니다. 서버는 기본적으로 호스트의 `127.0.0.1:8000`에 노출됩니다. 다른 포트를 쓰려면 기동 명령에 `MOLDWEIGHT_PORT=18082` 등을 추가합니다.

호스트 Python으로도 확인할 수 있습니다.

```bash
BASE_URL=http://127.0.0.1:8000 P95_SAMPLES=100 bash scripts/smoke_test.sh
```

대시보드: [http://localhost:8000/](http://localhost:8000/) · Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

업로드 CSV·로그·MLflow DB·아티팩트는 컨테이너 안에 저장합니다. `restart`는 상태를 유지하지만, 컨테이너를 재생성하면 이미지에 포함된 초기 상태로 돌아갑니다.

### 대시보드와 드리프트 시연

1. 새 trained 서버에서 대시보드를 엽니다. `static/demo/`의 정상·공정 변경 CSV가 자동으로 로드되므로 파일을 직접 선택할 필요가 없습니다.
2. **정상 생산 → 라인 가동**으로 2샷마다 한 번 실측합니다. 42샷 진행 후 실측 21건으로 정상 여부를 판정합니다.
3. 라인을 중지하고 **공정 조건이 바뀜 → 라인 가동**을 선택합니다. 기본 시작 샷은 `21101`이며 `21135`부터 금형온도 70°C 구간입니다.
4. 기본 분기점 정지 설정에서 드리프트·게이트 실패를 확인하고 **계속**을 누릅니다. 드리프트 후에는 전수검사로 전환해 다음 실측 21건을 모읍니다. 게이트 통과·Production 전환 후 다시 **계속**을 누르면 2샷마다 검사하는 모드로 돌아갑니다.

시드 v1로 점검한 샷 단위 시연에서는 `21142`에서 게이트 실패 후 v1을 유지하고, `21163`에서 통과 후 v2로 전환했습니다. 학습 결과와 누적 샷에 따라 판정 위치·RMSE·승격 결과는 달라질 수 있습니다. 41샷이 모이지 않은 상태에서는 재학습을 보류합니다. 중지·초기화·조건 변경 후에는 이전 샷 전송 루프가 이어지지 않습니다.

**21샷 묶음 전송 · 다른 파일 사용 · 응답 JSON**을 펼치면 배치 전송과 파일 선택을 사용할 수 있습니다. CLI 배치 시연은 샷 단위 화면과 시작 구간이 다릅니다.

```bash
python scripts/simulate_drift.py --url http://127.0.0.1:8000 \
  --drift-start 21122 --max-drift-batches 2
```

자동 로드·파일 선택은 브라우저에서 CSV를 읽어 `{shots}`를 만드는 동작입니다. `/data/upload`는 API로 사용할 수 있지만 현재 대시보드 업로드 패널은 주석 처리되어 표시되지 않습니다. 업로드는 파일 저장·요약만 수행하며 모델을 학습하거나 전환하지 않습니다. base 학습은 `data/sample_moldweight.csv`, fine-tune은 배치 API로 누적한 최근 41샷을 사용합니다. 업로드 파일로 base 학습하려면 `train_and_register(csv_path=...)`에 저장 경로를 명시합니다.

### E 통합 검증 재현

새 trained 테스트 서버에서만 아래 명령을 실행합니다. CSV 파일을 저장하고 샷 버퍼와 Production 모델 버전을 변경하므로 매번 새 컨테이너로 시작합니다.

```bash
docker compose -f serving_app/docker-compose.yml exec -T \
  -e INTEGRATION_TEST=1 serving-app bash scripts/smoke_test.sh
```

기본 smoke 3개 항목에 더해 40행 업로드 거부·정상 CSV 저장 및 요약, 20샷 판정 대기·21샷 정상·총 40샷 재학습 보류, 70°C 구간의 재학습·게이트 결과와 실제 `/predict.model_version` 일치, 운영 알람을 확인합니다. 게이트 실패가 발생하면 기존 버전 유지도 검사하며, 두 드리프트 배치에서 최소 한 번의 승격을 요구합니다. 기본 smoke에는 이 상태 변경 검증을 포함하지 않으며 재학습 요청의 제한 시간은 `INTEGRATION_REQUEST_TIMEOUT`(기본 600초)으로 조절합니다.

### Docker: 서버 기동만 확인

```bash
docker compose -f serving_app/docker-compose.yml up --build -d --wait
```

기본 `runtime` 이미지는 모델 학습을 수행하지 않습니다. 모델을 준비하기 전에는 `/health`의 `model_loaded=false`이며 정상 예측과 모델 smoke를 수행할 수 없습니다. Docker의 `healthy` 표시는 서버 기동 상태만 확인합니다.

### 로컬: 최초 모델 준비와 Production 실행

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 최초 MLflow 준비: 학습 구간의 scaler.pkl만 생성 (중복 모델 학습 생략)
python scripts/train_baseline_v1.py --scaler-only

export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python serving_app/train_and_register.py

MODEL_SOURCE=mlflow LOADING_MODE=eager \
  uvicorn serving_app.main:app --host 127.0.0.1 --port 8000
```

부트스트랩은 스케일러를 다시 저장하므로 기존 모델을 운영하는 중에 임의로 재실행하지 않습니다. 서빙·fine-tune은 저장된 스케일러를 재사용합니다. 로컬 파일 예측은 `python scripts/train_baseline_v1.py` 기본 명령으로 로컬 모델을 생성한 뒤 `MODEL_SOURCE=local`로 실행합니다. `--scaler-only`는 로컬 모델을 생성하지 않습니다. Production 기반 재학습 시연에는 MLflow 모델이 필요합니다.

## 데이터·모델·게이트

| 항목 | 현재 구현 |
|---|---|
| 입력 | 한 샷의 128×2 압력 곡선, 채널 순서 `[inj, cav]`, 허용 범위 −100~1500 bar |
| 타깃 | 같은 샷의 실측 무게(g), `experiment`는 기록용이며 모델 입력에서 제외 |
| base 데이터 | `data/sample_moldweight.csv`: 실험 15·20, 526샷 |
| drift 데이터 | `data/drift_shots_23.csv`: 실험 23, 303샷 |
| 전처리 | `scripts/preprocess_hdf5.py`: 2,048포인트를 16개씩 평균하여 128포인트로 변환 |
| 스케일러 | 학습 420샷으로만 fit한 `serving_app/models/scaler.pkl`, 검증·서빙·재학습에서 재사용 |
| 모델 | LSTM 32유닛 + Dense 1, 파라미터 4,513개 |
| base 분할·학습 | 실험별 생산 순서 앞 80% / 뒤 20% → train 420 / 검증 106, 200 epoch |
| fine-tune | Production 가중치에서 시작, 최근 41샷 중 앞 32샷 학습 / 뒤 9샷 평가, 10 epoch |
| 드리프트 | 최근 실측 있는 21샷 RMSE > 0.3g, 재학습은 41샷 확보 후 실행 |
| 배포 게이트 | RMSE ≤ 0.3g, 첫 배포는 평균 예측 대비 50% 이상 개선, 이후 기존 모델의 같은 평가셋 RMSE + 0.05g 이내 |

RMSE·holdout RMSE·비교 대상 RMSE의 NaN/무한대는 배포를 차단합니다. 기존 학습 분포의 holdout RMSE는 지식 손실 분석용으로 기록하며, 별도의 수치 임계값을 두지 않습니다. 게이트 실패 시 기존 Production을 유지하고, 통과 시 기존 버전을 Archived로 전환한 뒤 새 버전을 Production으로 올립니다.

CSV 필수 컬럼은 `cycle_counter, weight, inj_000~inj_127, cav_000~cav_127`이며 `experiment`는 선택 컬럼입니다. 업로드는 UTF-8(BOM 포함), 최소 41행, 유한한 무게와 허용 범위의 압력값을 요구합니다. 학습된 `.keras` 파일은 Git에 포함하지 않고 부트스트랩 또는 trained 빌드로 생성합니다.

## API 계약

| 메서드 | 경로 | 현재 동작 |
|---|---|---|
| POST | `/predict` | `curve` 128×2 → `predicted_weight`, `model_version` |
| POST | `/predict/batch-test` | `shots`(cycle_counter·curve·actual_weight) → `predictions`, `drift_check` |
| GET | `/health` | `status`, `model_loaded`, `loading_mode` |
| POST | `/data/upload` | multipart CSV 검증·저장 → `filename`, `rows` |
| GET | `/data/status` | 가장 최근 업로드 파일의 행 수·cycle 범위·weight 범위 |
| GET | `/logs` | 로그 파일 목록 |
| GET | `/logs/{filename}` | 로그 내용 |

`model_version`은 Registry 버전 번호 문자열(`"1"`, `"2"` 등)이며 로컬 모델이면 `"local"`입니다. 잘못된 곡선 길이·채널 수·압력 범위·비유한 값 또는 빈 `shots`는 422, 잘못된 업로드 CSV는 400을 반환합니다. 단일 `/predict`에는 실측 무게가 없으므로 드리프트 누적은 `/predict/batch-test`에서 수행합니다.

## 최근 통합 검증

2026-10-02 CI 경량화 소스 `25aa595`를 새 Linux ARM64 trained 이미지로 빌드하고, 소스 볼륨 연결 없이 검증했습니다.

| 검증 | 결과 |
|---|---|
| Production 학습·게이트 | v1 등록, 검증 RMSE **0.117024g**, 200 epoch 유지 |
| RULES smoke·의존성 | smoke **3/3**, `pip check` 통과 |
| warm p95 | **48.980ms**, 컨테이너 내부 loopback, 100회 순차 요청·nearest-rank |
| 업로드·재학습 가드 | 40행 거부·526행 저장, 20/21/40샷 경계 통과 |
| 게이트·서빙 전환 | 실패 시 v1 유지, 통과 후 v2, 운영 알람 확인 |
| local 모델 호환 | 동일 Production 모델의 `.keras` 로드·유한한 예측 확인 |
| 학습 캐시 분리 | 화면·health 라우터·smoke 수정 시 학습 레이어 캐시 재사용, 로컬 빌드 **3.57초** |

Python 3.11.17, TensorFlow 2.21.0, MLflow 3.16.0, NumPy 2.4.4이며 base MLflow run ID는 `d6c5324770884365b663ae49b6100e71`입니다. 같은 소스의 [Linux AMD64 CI](https://github.com/edder773/moldweight/actions/runs/36944917631)에서도 smoke·통합 검증을 통과했고 컨테이너 내부 p95는 **51.530ms**였습니다. 두 측정 모두 첫 로딩·학습 시간은 제외합니다.

아래는 이전 버전의 호스트 → Docker 측정으로, 위 컨테이너 내부 측정과 경로가 다릅니다.

### 이전 호스트 경유 p95 실측

2026-10-01, `dc4670709504a0450f2ee159baadde2b44c9642f` 기반 소스에 E 통합 검증 스크립트를 추가하여 별도 trained 이미지를 새로 빌드하고 아래 결과를 확인했습니다.

| 검증 | 결과 |
|---|---|
| Production 학습·게이트 | v1 등록, 검증 RMSE **0.117024g** |
| Registry와 API 버전 일치 | Registry `"1"` = `/predict.model_version` |
| RULES 10장 smoke | `/health model_loaded=true`, 정상 예측 200·유한한 숫자, 127포인트 곡선 422 — **3/3 통과** |
| 단일 예측 응답 시간 p95 | **44.584ms**, 100회 순차 요청, nearest-rank |
| 업로드 | 화면에서 41행 저장·요약, 자동 검증에서 40행 400·526행 저장·요약 |
| 드리프트·재학습 가드 | 20샷 판정 대기, 21샷 정상, 총 40샷 드리프트 감지 후 재학습 보류 |
| 실제 게이트·서빙 전환 | 실패 시 v1 유지, 통과 후 v2 서빙, WARN·INFO·FAIL·OK 로그 확인 |

측정 환경은 Apple M5 / macOS 26.6.2, Docker 29.6.1(Linux ARM64, CPU 10개·메모리 약 7.75GiB), Python 3.11.16, TensorFlow 2.21.0, MLflow 3.16.0, NumPy 2.4.4입니다. 모델은 `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager`, Production v1이며 MLflow run ID는 `7033f0cd6a3e40b7a31a7255a4a96a04`입니다.

p95는 호스트에서 `http://127.0.0.1:18085/predict`로 실험 23의 첫 곡선을 반복 전송해 측정했습니다. 정상 예측 smoke 후 수행한 warm 요청으로, 요청 직전부터 응답 JSON 수신까지의 시간을 정렬하여 95번째 값을 기록했습니다. 시작·첫 모델 로딩·재학습 시간은 포함하지 않습니다. 이 수치는 해당 환경의 실측이며 다른 하드웨어나 동시 부하의 성능을 보장하지 않습니다.

재현 명령은 다음과 같습니다. 서버 기동 명령에도 같은 포트를 지정합니다.

```bash
BASE_URL=http://127.0.0.1:18085 P95_SAMPLES=100 bash scripts/smoke_test.sh
```

## CI

GitHub Actions는 `main` push, `main` 대상 PR, 수동 실행에서 정적 검사를 항상 수행하고 변경 범위에 맞춰 Docker 검증을 선택합니다.

| 변경 범위 | 실행 |
|---|---|
| README·RULES·AGENTS·LICENSE 또는 `docs/*.md`만 수정 | 정적 검사 |
| `serving_app/static/`만 수정하거나 문서와 함께 수정 | 정적 검사 + runtime |
| 모델·데이터·백엔드·의존성·Docker·CI·그 외 파일 | 정적 검사 + runtime/trained 병렬 |
| 수동 실행·비교 기준 없음·빈 diff | 전체 검사 |

PR은 base 커밋부터 전체 변경을 비교하고 push는 이벤트의 이전 커밋부터 비교합니다. 파일 삭제·이름 변경도 검사 대상으로 포함하며, 비교 커밋을 확인할 수 없으면 전체 검증합니다. 워크플로 자체는 항상 실행하므로 문서 변경의 상태 검사도 완료됩니다.

1. Python 3.11·Bash·대시보드 JavaScript 문법, CI 선택 경계 회귀 검사, 커밋 공백 오류, runtime/trained Compose 설정.
2. Linux AMD64 runtime 빌드·healthcheck, `/health`·`/`·`/docs`·`/logs` 응답, `pip check`.
3. Linux AMD64 trained의 실제 학습·게이트·Production 로딩, Registry/API 버전 일치, RULES smoke 3개, 100회 warm p95.
4. 새 trained 서버에서 업로드·드리프트·재학습·게이트·실제 서빙 버전 전환.

Docker 베이스는 멀티 아키텍처 digest로 고정하고 Linux AMD64에서는 TensorFlow 2.21.0 CPU wheel을 사용합니다. trained는 스케일러만 준비한 뒤 모델을 한 번 학습합니다. 모델 학습 레이어를 화면·API 라우터·smoke와 분리해 해당 파일만 바뀌면 캐시를 재사용하며 데이터·모델 코드·설정 변경은 재학습합니다. 기본 로컬 baseline의 학습 동작은 유지합니다.

CI는 containerd 이미지 저장소와 Docker 드라이버를 사용해 빌드 결과를 러너에 직접 로드합니다. runtime/trained 캐시를 나누고 캐시 전송 대기 한도는 2분으로 설정합니다. 일반 수정의 2분 내 완료를 목표로 하되 최초 빌드·의존성 변경·200 epoch 재학습은 2분을 넘을 수 있습니다. 검증을 강제로 중단하는 2분 제한은 두지 않습니다. 1차 개선의 [실측](https://github.com/edder773/moldweight/actions/runs/36944917631)은 **5분 41초**로, [개선 전 9분 35초](https://github.com/edder773/moldweight/actions/runs/36942484118)에서 감소했습니다. 이후 변경 범위 선택과 직접 로드를 추가했습니다.

실행 요약에는 선택한 검사·이미지 크기·커밋·Production 버전·smoke·p95·통합 결과를 남깁니다. CI p95는 컨테이너 내부 loopback 측정이며 고정 합격 임계값은 없습니다. 이미지를 배포하거나 Registry에 push하지 않으며 결과는 [Actions](https://github.com/edder773/moldweight/actions)에서 확인합니다.

## 협업

본인 역할의 브랜치에서 담당 파일을 수정하고 main 대상 PR을 올립니다. main에는 PR로만 반영합니다. 원본 HDF5, 실행 로그, MLflow 기록, 가상환경, 학습된 `.keras` 모델은 커밋하지 않습니다. 기획서·PPT는 저장소 밖에서 작업합니다.

## 데이터 출처와 이용 조건

Bogedale, L.; Doerfel, S.; Schrodt, A.; Heim, H.-P. *Online Prediction of Molded Part Quality in the Injection Molding Process Using High-Resolution Time Series.* Polymers 2023, 15, 978. [논문](https://doi.org/10.3390/polym15040978) (CC BY 4.0).

CSV는 원본 압력 곡선을 평균 다운샘플링한 파생 데이터입니다. 원본 HDF5는 Git에 포함하지 않으며 전처리 스크립트와 파생 CSV 두 개를 추적합니다.
