#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BASE_URL="${BASE_URL:-http://localhost:8000}"
"${PYTHON:-python3}" - <<'SMOKE_PY'
import csv
import json
import math
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from serving_app.config import CURVE_LEN, N_CHANNELS, VALUE_MIN, VALUE_MAX

base_url = os.environ["BASE_URL"].rstrip("/")

def request(path, payload=None):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = Request(base_url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, None

try:
    status, health = request("/health")
    if status != 200 or not isinstance(health, dict) or health.get("model_loaded") is not True:
        raise ValueError(f"/health: model_loaded=true 필요 (HTTP {status}, {health})")
    print("PASS 1/3: /health model_loaded=true")

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
    status, result = request("/predict", {"curve": curve})
    weight = result.get("predicted_weight") if isinstance(result, dict) else None
    if status != 200 or type(weight) not in (int, float) or not math.isfinite(weight):
        raise ValueError(f"/predict: 숫자 predicted_weight 필요 (HTTP {status}, {result})")
    print("PASS 2/3: /predict 200, numeric predicted_weight")

    status, _ = request("/predict", {"curve": curve[:-1]})
    if status != 422:
        raise ValueError(f"길이가 부족한 곡선은 422 필요 (HTTP {status})")
    print("PASS 3/3: short curve returns 422")
except (URLError, OSError, ValueError, KeyError) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    sys.exit(1)
SMOKE_PY
