# MoldWeight

사출압력과 금형 내 압력 곡선으로 성형품 무게를 예측하고, 드리프트 감지 → 재학습 → 성능 게이트 → Production 전환을 연결하는 New 4조 팀 프로젝트입니다. 팀 규칙의 이름·숫자·함수 계약은 [RULES.md](RULES.md), 실행 설정은 [serving_app/config.py](serving_app/config.py)를 기준으로 합니다.

## 현재 통합 상태

2026-10-01 기준, PR #1~#9가 main에 반영됐습니다. 아래 실측은 기준 커밋 [`8bafcc7`](https://github.com/edder773/moldweight/commit/8bafcc7eb9a8878f84c04f0f2733e93b5aaa5408)의 코드로 수행했습니다.

| 영역 | 반영된 내용 |
|---|---|
| 데이터 | HDF5 전처리 스크립트, 실험 15·20의 526샷 CSV, 실험 23의 303샷 CSV |
| 모델 | 128×2 입력 LSTM, 고정 CurveScaler, MLflow 학습·Registry·Production 로딩 |
| API | 실제 모델 예측, 곡선 검증, CSV 업로드, 샷 배치 테스트 |
| 모니터링 | 최근 21샷 RMSE, 41샷 수집 후 fine-tune, 게이트 실패 시 기존 Production 유지 |
| 인프라 | runtime/trained Docker 타깃, Compose, smoke, 실제 모델 p95 측정, CI |
| 대시보드·시연 | 화면은 제공되지만 압력 곡선 API로의 변환이 남아 있음 |

실제 모델 연결과 모니터링 수정은 반영됐습니다. 예측 API에는 가짜 모델이나 `USE_MOCK_MODEL` 분기가 없습니다. 드리프트 판단용 예측 21개와 재학습용 샷 41개가 모두 모이기 전에는 재학습을 시작하지 않습니다. 배포 게이트는 NaN·무한대 RMSE를 거부하며, 스케일러는 검증 데이터를 제외한 학습 샷으로만 fit합니다.

## 역할과 남은 작업

| 역할 | 담당 | GitHub | 브랜치 | 담당 범위와 후속 작업 |
|---|---|---|---|---|
| A 데이터·기획·제출 | 장인우 | @inwoo-jang | `feat/data` | 데이터·설정·RULES 관리, B의 실측을 받아 RULES 6장과 제출 자료 반영 |
| B 모델·학습 | 이도권 | @dokwon33 | `feat/model` | 모델·스케일러·게이트·Registry, 분할 방식과 검증 수치를 A에게 전달 |
| C 서빙 API | 김선주 | @ssunju-02 | `feat/api` | 스키마·예측·업로드·배치 API, 중복 reload 정리, Swagger 명세 전달 |
| D 모니터링·재학습 | 김주은 | @jekim20 | `feat/monitor` | 드리프트·fine-tune 연결·게이트 결과 로그 |
| E 인프라·통합 | 김보석 | @edder773 | `feat/infra` | Docker·PR 통합·smoke·p95·CI, F 수정 후 화면 통합 재검증 |
| F 대시보드·시연 | 조영우 | @evermate | `feat/dashboard` | 대시보드와 시뮬레이터를 샷 계약으로 변환, 시연 화면 캡처 |

파일별 소유자는 RULES 1장을 따릅니다. 소유자가 아닌 파일의 수정은 해당 담당자에게 요청합니다.

현재 후속 작업은 다음과 같습니다.

- **C:** `serving_app/routers/predict.py:125`의 추가 reload 호출을 제거하고 재학습 후 예측 기록 초기화는 유지합니다. 현재 C+D 호출 경로는 승격 성공 시 reload 2회, 실패 시 1회입니다. D 단독 경로는 성공 1회·실패 0회로 정상이며, eager 모드의 불필요한 모델 로딩을 줄이려면 호출을 D에 일원화해야 합니다. 이 결과는 기준 main에서 학습·로딩을 대체한 호출 횟수 검증으로 재현했습니다.
- **F:** `serving_app/static/index.html`의 주가 CSV 안내와 `{prices}` 요청을 MoldWeight CSV·`{shots}` 계약으로 변경합니다. 현재 배치 버튼의 요청은 API에서 422가 됩니다.
- **F:** `scripts/simulate_drift.py`의 `Close` 기반 로직과 미구현 `send_batch`를 실험 23의 압력 곡선·실측 무게 전송으로 변경합니다.
- **E:** C 수정 후 승격 성공 시 reload 1회·실패 시 0회를 확인하고, F의 수정 반영 후 화면에서 업로드·예측 → 드리프트 → 재학습 → 게이트 → 모델 버전 전환을 확인합니다. 해당 화면 시연은 아직 완료 상태로 표시하지 않습니다.
- **A/B:** RULES 6장의 빈 표에 실제 분할 방식과 검증 RMSE를 반영합니다. 현재 코드의 base 분할은 420/106샷, fine-tune 분할은 최근 41샷 중 32/9샷입니다.

대시보드 수정 전에는 [Swagger](http://localhost:8000/docs)와 smoke 스크립트로 실제 API를 확인합니다.

## 실행

Python 3.11 또는 Docker Compose를 사용하며, 모든 명령은 저장소 루트에서 실행합니다.

### Docker: 실제 모델로 실행

`trained` 타깃은 시드 CSV로 스케일러·로컬 모델을 준비하고, MLflow 학습과 성능 게이트를 거쳐 Production 모델을 생성합니다. 학습 또는 Production 로딩이 실패하면 빌드도 실패합니다.

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

# 최초 준비 시 실행: 학습 구간만으로 scaler.pkl과 로컬 모델 생성
python scripts/train_baseline_v1.py

export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python serving_app/train_and_register.py

MODEL_SOURCE=mlflow LOADING_MODE=eager \
  uvicorn serving_app.main:app --host 127.0.0.1 --port 8000
```

부트스트랩은 스케일러를 다시 저장하므로 기존 모델을 운영하는 중에 임의로 재실행하지 않습니다. 서빙·fine-tune은 저장된 스케일러를 재사용합니다. 로컬 파일 예측만 확인하려면 모델 준비 후 `MODEL_SOURCE=local`을 사용할 수 있지만, Production 기반 재학습 시연에는 MLflow 모델이 필요합니다.

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

## 검증과 p95 실측

2026-10-01, 기준 커밋 `8bafcc7eb9a8878f84c04f0f2733e93b5aaa5408`으로 별도 trained 이미지를 빌드하고 아래 결과를 확인했습니다.

| 검증 | 결과 |
|---|---|
| Production 학습·게이트 | v1 등록, 검증 RMSE **0.117024g** |
| Registry와 API 버전 일치 | Registry `"1"` = `/predict.model_version` |
| RULES 10장 smoke | `/health model_loaded=true`, 정상 예측 200·유한한 숫자, 127포인트 곡선 422 — **3/3 통과** |
| 단일 예측 응답 시간 p95 | **41.748ms**, 100회 순차 요청, nearest-rank |

측정 환경은 Apple M5 / macOS 26.6.2, Docker 29.6.1(Linux ARM64, CPU 10개·메모리 약 7.75GiB), Python 3.11.16, TensorFlow 2.21.0, MLflow 3.16.0, NumPy 2.4.4입니다. 모델은 `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager`, Production v1이며 MLflow run ID는 `1d75efa777234a30b4616a7bda3ccc5c`입니다.

p95는 호스트에서 `http://127.0.0.1:18082/predict`로 실험 23의 첫 곡선을 반복 전송해 측정했습니다. 정상 예측 smoke 후 수행한 warm 요청으로, 요청 직전부터 응답 JSON 수신까지의 시간을 정렬하여 95번째 값을 기록했습니다. 시작·첫 모델 로딩·재학습 시간은 포함하지 않습니다. 이 수치는 해당 환경의 실측이며 다른 하드웨어나 동시 부하의 성능을 보장하지 않습니다.

재현 명령은 다음과 같습니다. 서버 기동 명령에도 같은 포트를 지정합니다.

```bash
BASE_URL=http://127.0.0.1:18082 P95_SAMPLES=100 bash scripts/smoke_test.sh
```

## CI와 main 보호

GitHub Actions는 `main` push, `main` 대상 PR, 수동 실행에서 순서대로 확인합니다.

1. Python 3.11·Bash 문법, 커밋 공백 오류, runtime/trained Compose 설정.
2. Linux AMD64 runtime 이미지 빌드, 컨테이너 healthcheck, `/health`·`/`·`/docs`·`/logs` 응답, `pip check`.
3. Linux AMD64 trained 이미지의 실제 학습·게이트·Production 로딩, Registry와 API 버전 일치, RULES smoke 3개 항목, 100회 warm 예측 p95.

trained CI의 커밋·Production 버전·smoke·p95는 실행 요약에 남습니다. CI의 p95는 컨테이너 내부 loopback에서 측정하므로 위 호스트 → Docker 실측과 측정 경로가 다릅니다. 고정 p95 합격 기준은 설정하지 않았습니다. 이미지를 배포하거나 Registry에 push하지 않으며, 결과는 [Actions](https://github.com/edder773/moldweight/actions)에서 확인합니다.

2026-10-01 main 보호를 복구했습니다. 관리자에게도 PR 경로를 적용하며 직접 push·강제 push·main 삭제를 차단합니다. 별도의 필수 승인 수나 필수 CI 상태 조건은 설정하지 않았습니다. E는 통합 검증과 smoke 결과를 확인한 후 PR을 병합합니다.

## 협업

본인 역할의 브랜치에서 담당 파일을 수정하고 main 대상 PR을 올립니다. main에는 PR로만 반영합니다. 원본 HDF5, 실행 로그, MLflow 기록, 가상환경, 학습된 `.keras` 모델은 커밋하지 않습니다. 기획서·PPT는 저장소 밖에서 작업합니다.

## 데이터 출처와 이용 조건

Bogedale, L.; Doerfel, S.; Schrodt, A.; Heim, H.-P. *Online Prediction of Molded Part Quality in the Injection Molding Process Using High-Resolution Time Series.* Polymers 2023, 15, 978. [논문](https://doi.org/10.3390/polym15040978) (CC BY 4.0).

CSV는 원본 압력 곡선을 평균 다운샘플링한 파생 데이터입니다. 원본 HDF5는 Git에 포함하지 않으며 전처리 스크립트와 파생 CSV 두 개를 추적합니다.
