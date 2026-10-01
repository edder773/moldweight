#!/usr/bin/env bash
# 기본: RULES 10장의 세 검증 모두 통과해야 성공합니다.
# BASE_URL, WAIT_SECONDS, REQUEST_TIMEOUT으로 주소·대기 시간을 조절할 수 있습니다.
# P95_SAMPLES를 지정하면 검증 후 단일 /predict 요청의 응답 시간 p95를 측정합니다.
set -euo pipefail
cd "$(dirname "$0")/.."
export BASE_URL="${BASE_URL:-http://localhost:8000}"
export WAIT_SECONDS="${WAIT_SECONDS:-60}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10}"
export P95_SAMPLES="${P95_SAMPLES:-0}"
"${PYTHON:-python3}" - <<'SMOKE_PY'
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from serving_app.config import CURVE_LEN, N_CHANNELS, VALUE_MIN, VALUE_MAX

base_url = os.environ["BASE_URL"].rstrip("/")

def request(path, payload=None):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = Request(base_url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=request_timeout) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, None

def prediction(payload):
    status, result = request("/predict", payload)
    weight = result.get("predicted_weight") if isinstance(result, dict) else None
    if status != 200 or type(weight) not in (int, float) or not math.isfinite(weight):
        raise ValueError(f"/predict: 숫자 predicted_weight 필요 (HTTP {status}, {result})")
    return weight

try:
    wait_seconds = float(os.environ["WAIT_SECONDS"])
    request_timeout = float(os.environ["REQUEST_TIMEOUT"])
    samples = int(os.environ["P95_SAMPLES"])
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("WAIT_SECONDS는 0 이상의 유한한 숫자여야 합니다")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("REQUEST_TIMEOUT은 0보다 큰 유한한 숫자여야 합니다")
    if samples < 0:
        raise ValueError("P95_SAMPLES는 0 이상의 정수여야 합니다")

    deadline = time.monotonic() + wait_seconds
    last_health = None
    while True:
        try:
            status, last_health = request("/health")
            if status == 200 and isinstance(last_health, dict) and last_health.get("model_loaded") is True:
                break
            last_health = f"HTTP {status}, {last_health}"
        except (URLError, OSError, ValueError) as exc:
            last_health = str(exc)
        if time.monotonic() >= deadline:
            raise ValueError(f"/health: model_loaded=true 대기 실패 ({last_health})")
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    print("PASS 1/3: /health model_loaded=true", flush=True)

    path = Path("data/drift_shots_23.csv")
    if not path.is_file():
        raise ValueError(f"A 담당 데이터가 필요합니다: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        raise ValueError("드리프트 CSV에 샷이 없습니다")
    curve = [[float(row[f"{channel}_{point:03d}"]) for channel in ("inj", "cav")]
             for point in range(CURVE_LEN)]
    if any(len(point) != N_CHANNELS or any(not math.isfinite(v) or not VALUE_MIN <= v <= VALUE_MAX for v in point) for point in curve):
        raise ValueError("드리프트 CSV 첫 샷이 곡선 계약을 만족하지 않습니다")
    payload = {"curve": curve}
    prediction(payload)
    print("PASS 2/3: /predict 200, numeric predicted_weight", flush=True)

    status, _ = request("/predict", {"curve": curve[:-1]})
    if status != 422:
        raise ValueError(f"길이가 부족한 곡선은 422 필요 (HTTP {status})")
    print("PASS 3/3: short curve returns 422", flush=True)

    if samples:
        durations = []
        for _ in range(samples):
            start = time.perf_counter()
            prediction(payload)
            durations.append((time.perf_counter() - start) * 1000)
        durations.sort()
        p95 = durations[math.ceil(len(durations) * 0.95) - 1]
        print(f"p95_predict_ms={p95:.3f} samples={samples} method=nearest-rank")
except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    sys.exit(1)
SMOKE_PY
