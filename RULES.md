# 몰드웨이트 팀 규칙 (RULES.md)

> 저장소 루트에 두고, 모두가 이 문서 기준으로 코드를 짭니다.
> **여기 적힌 이름·형식·숫자는 바꾸지 않습니다.** 바꿔야 하면 단톡에 먼저 올리고, A가 이 문서를 고친 뒤 다 같이 반영합니다.
> 구조는 개인 실습 스켈레톤(`project/`)을 그대로 씁니다. 새 폴더를 만들지 않습니다.

---

## 1. 역할과 파일 주인

파일마다 주인은 한 명입니다. 주인이 아닌 사람은 그 파일을 직접 고치지 않고, 주인에게 요청합니다.

| 역할 | 담당 | 맡는 일 | 주인인 파일 |
|---|---|---|---|
| **A** 데이터 · 기획 · 제출 | 장인우 | HDF5 → CSV 전처리, 이 규칙, 기획서·PPT, 최종 제출 | `scripts/preprocess_hdf5.py`, `data/sample_moldweight.csv`, `data/drift_shots_23.csv`, `serving_app/config.py`, `RULES.md` |
| **B** 모델 · 학습 | 이도권 | 입력 (128, 2) LSTM, 스케일러, 처음 학습, fine-tune, 배포 게이트, Registry | `data/features.py`, `serving_app/lstm_model.py`, `serving_app/train_and_register.py`, `serving_app/model_loader.py`, `scripts/train_baseline_v1.py` |
| **C** 서빙 API | 김선주 | 요청·응답 스키마, `/predict`, `/predict/batch-test`, 업로드 검증, 422/400 | `serving_app/schemas.py`, `serving_app/routers/predict.py`, `serving_app/routers/data.py`, `data/storage.py` |
| **D** 모니터링 · 재학습 | 김주은 | 21샷 RMSE 판정, 재학습 트리거, 게이트 실패 시 기존 유지, aiops 로그 | `serving_app/monitoring/drift_detector.py`, `serving_app/monitoring/retrain_trigger.py` |
| **E** 인프라 · 통합 | 김보석 | Docker, main 머지와 통합 확인, 확인 스크립트, 응답 시간 p95 측정 | `serving_app/Dockerfile`, `serving_app/docker-compose.yml`, `requirements.txt`, `scripts/smoke_test.sh` |
| **F** 대시보드 · 시연 | 조영우 | 대시보드 수정, 드리프트 시뮬레이션, ⑥ 동작 화면 캡처 | `serving_app/static/index.html`, `scripts/simulate_drift.py` |

그대로 두는 파일: `serving_app/main.py`, `routers/health.py`, `routers/logs.py`
(`main.py`의 앱 제목만 E가 `"MoldWeight Serving & AIOps"`로 바꿉니다.)

---

## 2. 설정값 — `serving_app/config.py` (A가 만들고 잠금)

숫자는 모두 이 파일에서 가져옵니다. **다른 파일에 숫자를 직접 적지 않습니다.** 대시보드(`index.html`)는 파이썬을 못 읽으니, F가 같은 값을 맨 위 상수로 옮겨 적고 이 파일과 맞는지 확인합니다.

```python
# serving_app/config.py
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
```

---

## 3. 데이터 형식 (A → 전원)

### 3-1. CSV 파일 — 한 행 = 샷 하나

```
cycle_counter, experiment, weight, inj_000, …, inj_127, cav_000, …, cav_127
```

| 컬럼 | 뜻 | 형식 |
|---|---|---|
| `cycle_counter` | 샷 번호 (생산 순서) | int, 오름차순 정렬 |
| `experiment` | 실험 번호 15 / 20 / 23 | int. **모델 입력으로 쓰지 않음** (기록용) |
| `weight` | 실측 무게 | float, g, 소수 3자리 |
| `inj_000`~`inj_127` | 사출압력 곡선 (`Einspritzdruck`) | float, bar, 소수 2자리 |
| `cav_000`~`cav_127` | 금형 내 압력 곡선 (`Werkzeuginnendruck`) | float, bar, 소수 2자리 |

- 2,048포인트를 앞에서부터 16개씩 묶어 평균 낸 값입니다. 시간 순서를 유지합니다.
- **채널 순서는 항상 [사출압력, 금형 내 압력]**입니다. 모든 코드에서 `inj`가 0번, `cav`가 1번입니다.

### 3-2. 파일 두 개

| 파일 | 내용 | 쓰는 곳 |
|---|---|---|
| `data/sample_moldweight.csv` | 실험 15·20, **526샷** | Docker 빌드 시드, 대시보드 업로드, 처음 학습 |
| `data/drift_shots_23.csv` | 실험 23, **303샷** | 드리프트 주입 (`simulate_drift.py`, 대시보드 버튼) |

원본 HDF5(91 MB)는 Git에 올리지 않습니다. 전처리 스크립트와 CSV 두 개만 올립니다.

### 3-3. 파이썬 안에서 샷 하나 (B가 `data/features.py`에 구현)

```python
row = {
    "cycle_counter": 20191,
    "experiment": 15,
    "weight": 115.021,
    "curve": [[inj_0, cav_0], [inj_1, cav_1], ..., [inj_127, cav_127]],   # 128 × 2
}
load_rows(csv_path) -> list[row]      # cycle_counter 오름차순
```

### 3-4. 스케일러 (B)

- 이름: `CurveScaler` (`data/features.py`). 스켈레톤 `HAICScaler`를 대체합니다.
- 채널별 min-max, 무게도 min-max. **`sample_moldweight.csv`로 한 번만 fit**해서 `serving_app/models/scaler.pkl`에 저장하고 커밋합니다.
- 재학습 때 다시 fit하지 않습니다.
- 메서드: `fit(rows)`, `transform_curve(curve) -> 128×2`, `scale_weight(w)`, `inverse_weight(s)`, `save(path)`, `load(path)`

---

## 4. API 형식 (C → D, F, A)

실제 필드 이름은 이 표와 같아야 합니다. ⑤ API 명세(기획서)도 이 표를 그대로 씁니다.

### `POST /predict`

```json
// Request
{ "curve": [[812.4, 0.0], [905.1, 2.3], "… 128개 …"] }

// Response 200
{ "predicted_weight": 115.21, "model_version": "2" }
```
- `model_version`: MLflow Registry의 **버전 번호 문자열** (`"1"`, `"2"` …). 로컬 모델이면 `"local"`.
- 422: 곡선 길이 ≠ 128, 한 점의 값이 2개가 아님, 값이 `VALUE_MIN`~`VALUE_MAX` 밖.

### `POST /predict/batch-test`

```json
// Request
{ "shots": [ { "cycle_counter": 20933, "curve": [[…], …], "actual_weight": 114.12 }, … ] }

// Response 200
{
  "predictions": [114.55, 114.61, …],
  "drift_check": {
    "status": "collecting | ok | retrain_triggered",
    "rmse": 0.452,          // 윈도우가 21개 미만이면 null
    "window": 21,           // 지금까지 쌓인 개수 (최대 21)
    "promoted": true,       // retrain_triggered일 때만
    "model_version": "2"    // retrain_triggered일 때만, 판정 후 Production 버전
  }
}
```
- `shots`는 1개 이상입니다. 샷마다 예측 → `(predicted, actual)`을 쌓고, **곡선 + 실측 무게도 같이 쌓습니다** (재학습용).
- 422: `/predict`와 같은 규칙, `shots`가 비어 있음.

### `GET /health` — 그대로

```json
{ "status": "ok", "model_loaded": true, "loading_mode": "eager" }
```

### `POST /data/upload`

- multipart `file`. 필수 컬럼: `cycle_counter`, `weight`, `inj_000`~`inj_127`, `cav_000`~`cav_127` (`experiment`는 있어도 되고 없어도 됨)
- 최소 행 수: `RETRAIN_SHOTS`(41)
- 400: UTF-8 아님, 필수 컬럼 누락, 행 수 부족
- Response: `{"filename": "moldweight_1727771234.csv", "rows": 526}`
- 저장 파일 이름 접두어: `haic_` → **`moldweight_`**

### `GET /data/status`

```json
{ "exists": true, "filename": "…", "rows": 526,
  "first_cycle": 20191, "last_cycle": 20897, "weight_min": 114.48, "weight_max": 116.69 }
```

### `GET /logs`, `GET /logs/{filename}` — 그대로

---

## 5. 모듈 사이 함수 약속

### C ↔ D: 예측 기록 넘기기

`routers/predict.py`(C)가 두 리스트를 관리하고, 매 batch 끝에 D의 함수를 부릅니다.

```python
recent_predictions: list[dict]   # [{"predicted": float, "actual": float}], 최근 WINDOW_SIZE개만 유지
recent_shots: list[dict]         # [{"curve": 128×2, "weight": float, "cycle_counter": int}], 최근 RETRAIN_SHOTS개만 유지

drift_check = check_and_trigger(recent_predictions, recent_shots)   # D, monitoring/retrain_trigger.py
```
- 재학습이 끝나면(통과든 실패든) **C가 `recent_predictions`를 비웁니다.** 그래야 새 모델로 다시 21개를 모아 판정합니다.
- 재학습 후 새 모델을 쓰도록 `model_loader`의 캐시를 비우는 함수는 B가 제공합니다: `model_loader.reload()`.

### D ↔ B: 재학습 부르기

```python
result = fine_tune(rows)   # B, serving_app/train_and_register.py
# rows: recent_shots를 3-3 형식으로 바꾼 것 (최근 41샷)
# result: {"promoted": bool, "rmse": float, "version": str | None, "prev_version": str, "prev_rmse": float}
```
- 게이트 판단(② 절대 성능, ④ 회귀 방지)은 **B의 `fine_tune` 안에서** 합니다. D는 결과만 보고 로그를 씁니다.
- fine_tune의 평가 방식(41샷 중 어디로 평가할지)은 **B가 1일차에 실험해서 이 문서 6장에 적습니다.**

---

## 6. B가 1일차에 정해서 채울 칸

| 항목 | 값 | 정한 근거 |
|---|---|---|
| 처음 학습 train/검증 분할 | (예: 시간순 앞 80% / 뒤 20%) | |
| fine_tune 평가 데이터 | (예: 41샷 중 뒤 9샷) | |
| 처음 학습 검증 RMSE | | |

---

## 7. 로그 문구 (D가 쓰고, F가 캡처하고, A가 기획서에 씀)

`logs/aiops.log`에 아래 문구를 **그대로** 씁니다. 형식은 `main.py`의 기존 포맷(`시각 [레벨] 메시지`)을 따릅니다.

| 시점 | 레벨 | 메시지 |
|---|---|---|
| 드리프트 감지 | `warning` | `[WARN] drift detected rmse=0.452 window=21 threshold=0.3` |
| 재학습 시작 | `info` | `[INFO] retrain triggered shots=41` |
| 게이트 통과 | `info` | `[OK] gate passed rmse=0.244 -> MoldWeight_Predictor v2 promoted` |
| 게이트 실패 | `warning` | `[FAIL] gate failed rmse=0.331 -> keep MoldWeight_Predictor v1` |

숫자는 소수 3자리입니다.

---

## 8. Git 규칙

- 브랜치: `feat/model`(B), `feat/api`(C), `feat/monitor`(D), `feat/infra`(E), `feat/dashboard`(F). A는 `feat/data`, 기획서는 저장소 밖에서 작업합니다.
- **정해진 머지 시간**에 각자 PR을 올립니다. 덜 끝났어도 서버가 뜨는 상태면 올립니다.
- **저장소 주인과 머지는 E가 맡습니다.** E가 저장소를 만들고 전원을 초대하며, main에는 PR로만 들어가게 보호합니다(시작 전 A의 첫 커밋만 예외).
- main에서 `docker compose up` 후 `scripts/smoke_test.sh`가 통과해야 머지합니다.
- 커밋하지 않는 것: 원본 HDF5, `mlruns/`, `mlflow.db`, `logs/`, `data/uploads/`, `.venv/`, `__pycache__/`
- 커밋하는 것: CSV 두 개, `serving_app/models/scaler.pkl`

---

## 9. 일정 (1일차 오후 시작)

| 시각 | 할 일 | 누가 | 순서 |
|---|---|---|---|
| **1일차 시작 전 ①** | 저장소 생성 (스켈레톤 `project/` 그대로 첫 커밋, `.gitignore`), main 보호 설정, 5명 초대 | E | 가장 먼저 |
| **1일차 시작 전 ②** | 전처리 CSV 2개, `config.py`, 이 문서를 main에 직접 커밋 | A | ① 다음 |
| **1일차 시작** | 이 문서 같이 읽고 확정, 브랜치 생성 | 전원 | — |
| **1일차 개발** | 각자 개발 (아래 "가짜로 먼저") | B C D E F 동시 | 서로 안 기다림 |
| **1일차 머지 1** | **머지 1** | 전원 PR → E | — |
| **1일차 연결 1** | **연결 1**: B 모델·스케일러 → C가 가짜 모델 교체 → E 컨테이너 확인 (업로드 → 학습 → `/predict`) | B → C → E | 순서 있음 |
| **2일차 09:00–10:30** | D가 `fine_tune` 연결, F가 대시보드를 실제 API에 연결, B가 6장 채움 | B D F 동시 | — |
| **10:30** | **머지 2** | 전원 PR → E | — |
| **10:30–12:00** | **연결 2**: 3일차 주입 → `[WARN]` → 재학습 → 게이트 → 버전 전환 | D → F → E | 순서 있음 |
| **13:00–14:00** | 버그 수정 | 전원 | — |
| **14:00** | **코드 동결** | — | 이후 버그 수정만 |
| **14:00–15:30** | ⑥ 캡처(F), p95 측정(E), Swagger 화면을 A에게 전달(C), 6장 수치 전달(B) | E F C B | — |
| **15:30–** | 기획서에 실제 값 반영, PDF, 리허설, 제출 | A (+ 전원 리허설) | — |

### 가짜로 먼저 만들기 (1일차 개발 중)

| 역할 | 진짜가 오기 전까지 쓰는 것 |
|---|---|
| B | `sample_moldweight.csv`로 바로 학습 (데이터는 이미 있음) |
| C | 항상 `115.16`을 돌려주는 가짜 모델, `model_version: "mock"` |
| D | `fine_tune`을 흉내 내는 가짜 함수 (`promoted=True, rmse=0.244`) |
| E | 스켈레톤 그대로 Docker 빌드 → 시드 CSV 이름만 교체 |
| F | 4장 형식대로 응답하는 가짜 JSON으로 화면 먼저 수정 |

---

## 10. 확인 스크립트 — `scripts/smoke_test.sh` (E)

머지할 때마다 main에서 실행합니다. 셋 다 통과해야 머지합니다.

1. `GET /health` → `model_loaded: true`
2. `POST /predict` (드리프트 파일 첫 샷) → 200, `predicted_weight`가 숫자
3. `POST /predict` (127포인트 곡선) → 422
