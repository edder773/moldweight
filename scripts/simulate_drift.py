"""
드리프트 감지 시뮬레이션 (RULES.md 3-2, 4장 /predict/batch-test).

핵심 프로세스 (교안의 5단계를 MoldWeight 데이터로 치환):
    1) 기준 통계 산출   - 정상 샷(실험 15·20)과 드리프트 샷(실험 23)의 실측 무게 평균·표준편차
    2) 정상 입력 테스트 - 학습 분포와 같은 샷을 보내 status=ok 확인 (베이스라인)
    3) 드리프트 데이터  - 생성하지 않습니다. 실험 23(금형온도·재료 lot 변경)의 실제 샷을 그대로 씁니다
    4) 드리프트 주입    - 실험 23 샷을 배치 크기(기본 WINDOW_SIZE=21)씩 순서대로 전송
    5) 결과 관찰       - RMSE 상승 -> [WARN] -> 재학습 -> 게이트 -> /predict의 model_version 전환 확인

정상 배치는 data/sample_moldweight.csv의 마지막 배치 크기만큼(실험 20의 끝)을 씁니다.
B의 분할(실험별 뒤 20%가 검증)에서 학습에 쓰이지 않은 같은 분포의 샷이라 ok가 나와야 합니다.

드리프트 판정은 배치 끝에 한 번만 돌아가므로, 배치 크기를 WINDOW_SIZE(21)로 맞추면
배치 하나 = 판정 한 번입니다. 재학습은 recent_shots가 RETRAIN_SHOTS(41) 이상일 때만 실행되므로
정상 배치 없이 드리프트부터 보내면 첫 배치는 status=collecting(rmse 포함)으로 보류되고
두 번째 배치에서 재학습이 돕니다.

사전 준비: 서버가 떠 있어야 합니다 (uvicorn serving_app.main:app 또는 docker compose up).
실행:
    python scripts/simulate_drift.py                       # 정상 1배치 -> 드리프트 배치 순서로 전송
    python scripts/simulate_drift.py --batch-size 41       # 배치 크기 변경
    python scripts/simulate_drift.py --max-drift-batches 3 # 드리프트 배치 수 제한
    python scripts/simulate_drift.py --url http://localhost:8077
"""
import argparse
import csv
import math
import os
import statistics
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving_app.config import CURVE_LEN, DRIFT_THRESHOLD, RETRAIN_SHOTS, WINDOW_SIZE

DEFAULT_URL = "http://localhost:8000"
NORMAL_CSV = "data/sample_moldweight.csv"
DRIFT_CSV = "data/drift_shots_23.csv"

INJ_COLUMNS = [f"inj_{i:03d}" for i in range(CURVE_LEN)]
CAV_COLUMNS = [f"cav_{i:03d}" for i in range(CURVE_LEN)]


def load_shots(csv_path: str) -> list[dict]:
    """CSV(한 행 = 샷 하나)를 /predict/batch-test의 shots 항목 형식으로 읽는다. cycle_counter 오름차순.

    data/features.py의 load_rows와 같은 컬럼을 읽지만, 이 스크립트는 서버 밖에서 돌기 때문에
    학습 모듈을 import하지 않고 CSV만 직접 읽는다.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        shots = [
            {
                "cycle_counter": int(r["cycle_counter"]),
                "curve": [[float(r[inj]), float(r[cav])] for inj, cav in zip(INJ_COLUMNS, CAV_COLUMNS)],
                "actual_weight": float(r["weight"]),
            }
            for r in csv.DictReader(f)
        ]
    shots.sort(key=lambda s: s["cycle_counter"])
    return shots


def weight_stats(shots: list[dict]) -> tuple[float, float]:
    weights = [s["actual_weight"] for s in shots]
    return statistics.mean(weights), statistics.pstdev(weights)


def rmse_of(actual: list[float], predicted: list[float]) -> float | None:
    pairs = list(zip(actual, predicted))
    if not pairs:
        return None
    return math.sqrt(sum((a - p) ** 2 for a, p in pairs) / len(pairs))


def serving_version(base_url: str, shot: dict) -> str:
    """/predict를 한 번 호출해 지금 실제로 서빙 중인 모델 버전을 돌려준다.

    drift_check.model_version은 Registry의 Production 버전이라, 서빙 캐시가 갱신됐는지는
    /predict 응답으로 따로 확인해야 한다 (통합가이드 2-1).
    """
    resp = requests.post(f"{base_url}/predict", json={"curve": shot["curve"]}, timeout=30)
    resp.raise_for_status()
    return resp.json()["model_version"]


def describe(check: dict, sent_shots: int) -> str:
    """drift_check를 한 줄 설명으로 바꾼다. collecting에 rmse가 있으면 '감지됐지만 보류'다."""
    status = check.get("status")
    rmse = check.get("rmse")
    if status == "ok":
        return f"정상 (rmse={rmse:.3f} <= {DRIFT_THRESHOLD})"
    if status == "collecting":
        if rmse is None:
            return f"판정 윈도우 수집 중 ({check.get('window')}/{WINDOW_SIZE})"
        return (
            f"드리프트 감지 (rmse={rmse:.3f} > {DRIFT_THRESHOLD}) - 재학습 샷 부족, "
            f"보류 (보낸 샷 {min(sent_shots, RETRAIN_SHOTS)}/{RETRAIN_SHOTS})"
        )
    if status == "retrain_triggered":
        outcome = "게이트 통과 -> 승격" if check.get("promoted") else "게이트 실패 -> 기존 버전 유지"
        return f"드리프트 감지 (rmse={rmse:.3f}) -> 재학습 -> {outcome} (Production v{check.get('model_version')})"
    return f"알 수 없는 status: {status}"


def send_batch(base_url: str, shots: list[dict], label: str, sent_before: int) -> dict:
    """배치 하나를 /predict/batch-test로 보내고 drift_check를 출력·반환한다."""
    first, last = shots[0]["cycle_counter"], shots[-1]["cycle_counter"]
    resp = requests.post(f"{base_url}/predict/batch-test", json={"shots": shots}, timeout=600)
    if resp.status_code != 200:
        print(f"[{label}] HTTP {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
    result = resp.json()
    check = result["drift_check"]
    client_rmse = rmse_of([s["actual_weight"] for s in shots], result["predictions"])
    print(f"[{label}] 샷 {first}~{last} ({len(shots)}개) · 이 배치 RMSE={client_rmse:.3f}g")
    print(f"[{label}] drift_check = {check}")
    print(f"[{label}] -> {describe(check, sent_before + len(shots))}")
    return result


def main():
    parser = argparse.ArgumentParser(description="MoldWeight 드리프트 감지 시뮬레이션")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"서빙 서버 주소 (기본 {DEFAULT_URL})")
    parser.add_argument("--batch-size", type=int, default=WINDOW_SIZE, help=f"배치당 샷 수 (기본 WINDOW_SIZE={WINDOW_SIZE})")
    parser.add_argument("--normal-csv", default=NORMAL_CSV)
    parser.add_argument("--drift-csv", default=DRIFT_CSV)
    parser.add_argument("--skip-normal", action="store_true", help="정상 배치를 보내지 않고 드리프트부터 주입")
    parser.add_argument(
        "--drift-start", type=int, default=None,
        help="드리프트 파일에서 이 샷 번호(cycle_counter)부터 주입 (기본: 파일 처음부터). "
             "v1 모델 기준 실험 23 앞부분(금형온도 80·90°C)은 RMSE 0.3 이내라, 시연을 빨리 보려면 "
             "금형온도 70°C 구간이 시작되는 21122 근처를 지정",
    )
    parser.add_argument("--max-drift-batches", type=int, default=None, help="보낼 드리프트 배치 수 (기본: 파일 끝까지)")
    parser.add_argument("--stop-on-retrain", action="store_true", help="첫 재학습이 끝나면 중단")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=30).json()
    print(f"[0] /health = {health}")

    normal = load_shots(args.normal_csv)
    drift = load_shots(args.drift_csv)
    n_mean, n_std = weight_stats(normal)
    d_mean, d_std = weight_stats(drift)
    print(f"[1] 기준 통계: 정상 {len(normal)}샷 무게 mean={n_mean:.3f} std={n_std:.3f} / "
          f"드리프트 {len(drift)}샷 mean={d_mean:.3f} std={d_std:.3f} (단위 g)")

    sent = 0
    version_before = serving_version(base_url, drift[0])
    print(f"[1] 시작 시 서빙 중인 모델 버전: {version_before}")

    if not args.skip_normal:
        print(f"[2] 정상 입력 테스트 전송 (sample CSV 마지막 {args.batch_size}샷)...")
        send_batch(base_url, normal[-args.batch_size:], "normal", sent)
        sent += args.batch_size

    if args.drift_start is not None:
        drift = [s for s in drift if s["cycle_counter"] >= args.drift_start]
        if not drift:
            sys.exit(f"--drift-start {args.drift_start} 이후 샷이 {args.drift_csv}에 없습니다.")
    batches = [drift[i : i + args.batch_size] for i in range(0, len(drift), args.batch_size)]
    if args.max_drift_batches is not None:
        batches = batches[: args.max_drift_batches]
    print(f"[3-4] 드리프트 주입: 실험 23 샷 {len(drift)}개(샷 {drift[0]['cycle_counter']}부터)를 "
          f"{args.batch_size}개씩 {len(batches)}배치로 전송...")
    for i, batch in enumerate(batches, start=1):
        result = send_batch(base_url, batch, f"drift_{i}", sent)
        sent += len(batch)
        if result["drift_check"]["status"] == "retrain_triggered":
            version_after = serving_version(base_url, batch[-1])
            print(f"[5] 서빙 모델 버전: {version_before} -> {version_after}"
                  + ("" if version_after != version_before else " (변경 없음)"))
            version_before = version_after
            if args.stop_on_retrain:
                break

    print("[5] 결과 확인: 대시보드 재학습 로그 패널 또는 logs/aiops.log에서 "
          "[WARN] drift detected -> [INFO] retrain triggered -> [OK]/[FAIL] 순서를 확인하세요.")


if __name__ == "__main__":
    main()
