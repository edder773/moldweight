# MoldWeight

사출압력과 금형 내 압력 곡선으로 성형품 무게를 예측하고, 드리프트 감지 → 재학습 → 성능 게이트 → Production 전환을 연결하는 New 4조 팀 프로젝트입니다.

이 저장소는 개인 실습 `project/`를 가져온 **개발 시작용 베이스**입니다. 아래 역할별 변환 작업이 남아 있습니다. 팀 규칙의 이름·숫자·함수 계약은 `RULES.md`를 기준으로 합니다.

## 역할과 브랜치

| 역할 | 담당 | GitHub | 브랜치 | 시작할 일 |
|---|---|---|---|---|
| A 데이터·기획·제출 | 장인우 | @inwoo-jang | `feat/data` | HDF5 전처리, CSV 2개 추가, 공통 설정·규칙 관리 |
| B 모델·학습 | 이도권 | @dokwon33 | `feat/model` | CurveScaler, (128, 2) LSTM, 학습·Registry·게이트, reload() |
| C 서빙 API | 김선주 | @ssunju-02 | `feat/api` | curve 스키마, 업로드 검증, batch-test, 최근 샷 기록 |
| D 모니터링·재학습 | 김주은 | @jekim20 | `feat/monitor` | 21샷 RMSE, 41샷 fine_tune 연결, 규정 로그 |
| E 인프라·통합 | 김보석 | @edder773 | `feat/infra` | Docker 통합, PR 머지, smoke test, p95 측정 |
| F 대시보드·시연 | 조영우 | @evermate | `feat/dashboard` | 대시보드·드리프트 주입 스크립트, 화면 캡처 |

파일별 소유자는 `RULES.md` 1장을 따릅니다. 다른 사람의 파일 변경은 소유자에게 요청합니다. `main.py`는 앱 제목만 변경했고, `health.py`, `logs.py` 및 기존 패키지 구조는 보존했습니다.

## 현재 베이스 상태

- `RULES.md`: 제공받은 원문 그대로.
- `serving_app/config.py`: RULES 2장 설정 그대로.
- `.gitignore`: 저장소 구조 문서의 제외 규칙 그대로. `scaler.pkl`은 추적 대상입니다.
- `scripts/smoke_test.sh`: `/health`, 올바른 곡선 예측, 길이가 부족한 곡선의 422를 확인합니다.
- 주가 예제 `data/sample_haic_prices.csv`는 삭제했습니다. 나머지 담당 파일의 기존 주가 로직·TODO는 담당자가 MoldWeight 계약으로 변환해야 합니다.
- 원본 ZIP에 `.dockerignore`가 없어 기존 개인 실습 작업본의 `.dockerignore`를 가져왔습니다.

다음은 아직 제공되지 않았으며 가짜 파일로 대체하지 않았습니다.

| 산출물 | 담당 | 완료 조건 |
|---|---|---|
| `scripts/preprocess_hdf5.py` | A | 원본 HDF5의 2,048 포인트를 16개씩 평균, 128×2 곡선 CSV 출력 |
| `data/sample_moldweight.csv` | A | 실험 15·20, 526샷 |
| `data/drift_shots_23.csv` | A | 실험 23, 303샷 |
| `serving_app/models/scaler.pkl` | B | sample CSV로 한 번 fit한 CurveScaler 저장 후 커밋 |

B는 train/검증 분할과 fine_tune 평가 방식을 실험한 뒤 RULES 6장 반영을 A에게 요청합니다. 성능 수치는 측정한 값만 기록합니다.

## 실행

Python 3.11과 Docker Compose를 사용합니다. 저장소 루트에서 실행합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 스켈레톤 서버 기동 확인: 모델을 준비하기 전에는 health.model_loaded=false
LOADING_MODE=lazy uvicorn serving_app.main:app --host 0.0.0.0 --port 8000
```

대시보드: http://localhost:8000/ · Swagger: http://localhost:8000/docs

A의 CSV, B의 모델·스케일러, C의 API 변환을 통합한 뒤 Docker로 재현합니다. 현재 베이스에서는 이 통합 실행 및 smoke test가 아직 완료되지 않았습니다.

```bash
docker compose -f serving_app/docker-compose.yml up --build
scripts/smoke_test.sh
```

확인 서버가 다른 주소라면 `BASE_URL=http://localhost:8000 scripts/smoke_test.sh`를 사용합니다.

## API 계약

| 메서드 | 경로 | 계약 |
|---|---|---|
| POST | `/predict` | `curve` 128×2 → `predicted_weight`, Registry 버전 문자열 `model_version` |
| POST | `/predict/batch-test` | `shots` → `predictions`, `drift_check` |
| GET | `/health` | `status`, `model_loaded`, `loading_mode` |
| POST | `/data/upload` | multipart CSV → `filename`, `rows` |
| GET | `/data/status` | 업로드 현황, cycle 범위, weight 범위 |
| GET | `/logs` | 로그 파일 목록 |
| GET | `/logs/{filename}` | 로그 내용 |

위 API는 팀 구현 목표입니다. 현재 스켈레톤의 주가용 요청·응답을 C가 RULES 4장 계약에 맞춥니다. 채널 순서는 항상 `[사출압력, 금형 내 압력]`이고 `experiment`는 모델 입력으로 쓰지 않습니다.

## 협업 순서

```bash
git clone https://github.com/edder773/moldweight.git
cd moldweight
git fetch origin
# 본인 역할의 브랜치 선택 (예: C)
git switch --track origin/feat/api
```

각자 담당 파일만 개발해 정해진 머지 시간에 `main` 대상 PR을 올립니다. E가 Docker 통합 실행과 `scripts/smoke_test.sh` 통과를 확인하고 머지합니다. 초기 베이스 커밋 이후 main에는 PR로만 반영합니다. 원본 HDF5, 실행 로그, MLflow 기록, 가상환경, 학습된 `.keras` 모델은 커밋하지 않습니다. 기획서·PPT는 저장소 밖에서 작업합니다.

## 데이터 출처와 이용 조건

Bogedale, L.; Doerfel, S.; Schrodt, A.; Heim, H.-P. *Online Prediction of Molded Part Quality in the Injection Molding Process Using High-Resolution Time Series.* Polymers 2023, 15, 978. [https://doi.org/10.3390/polym15040978](https://doi.org/10.3390/polym15040978) (CC BY 4.0).

CSV는 원본 압력 곡선을 평균 다운샘플링한 파생 데이터입니다. 실제 CSV는 A가 전처리 후 추가합니다.

기초 코드는 제공된 개인 실습 스켈레톤을 사용합니다. 원본 안내: “다음 실습 코드는 학습 목적으로만 사용 바랍니다.” 문의: architect@sk.com, audit@korea.ac.kr 임성열 Ph.D.
