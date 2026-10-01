#!/usr/bin/env bash
# 기본: RULES 10장의 세 검증 모두 통과해야 성공합니다.
# BASE_URL, WAIT_SECONDS, REQUEST_TIMEOUT으로 주소·대기 시간을 조절할 수 있습니다.
# P95_SAMPLES를 지정하면 검증 후 단일 /predict 요청의 응답 시간 p95를 측정합니다.
# INTEGRATION_TEST=1은 새 trained 테스트 서버에서 업로드·드리프트·실제 버전 전환을 검증합니다.
set -euo pipefail
cd "$(dirname "$0")/.."
export BASE_URL="${BASE_URL:-http://localhost:8000}"
export WAIT_SECONDS="${WAIT_SECONDS:-60}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10}"
export P95_SAMPLES="${P95_SAMPLES:-0}"
export INTEGRATION_TEST="${INTEGRATION_TEST:-0}"
export INTEGRATION_REQUEST_TIMEOUT="${INTEGRATION_REQUEST_TIMEOUT:-600}"
"${PYTHON:-python3}" - <<'SMOKE_PY'
import csv
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from serving_app.config import (CURVE_LEN, DRIFT_THRESHOLD, N_CHANNELS, RETRAIN_SHOTS,
                                VALUE_MIN, VALUE_MAX, WINDOW_SIZE)

base_url = os.environ["BASE_URL"].rstrip("/")

def request(path, payload=None, *, timeout=None):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = Request(base_url + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=request_timeout if timeout is None else timeout) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, None

def prediction(payload):
    status, result = request("/predict", payload)
    weight = result.get("predicted_weight") if isinstance(result, dict) else None
    if status != 200 or type(weight) not in (int, float) or not math.isfinite(weight):
        raise ValueError(f"/predict: 숫자 predicted_weight 필요 (HTTP {status}, {result})")
    return weight

def load_shots(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    shots = [{
        "cycle_counter": int(row["cycle_counter"]),
        "curve": [[float(row[f"{channel}_{point:03d}"]) for channel in ("inj", "cav")]
                  for point in range(CURVE_LEN)],
        "actual_weight": float(row["weight"]),
    } for row in rows]
    return sorted(shots, key=lambda shot: shot["cycle_counter"])


def upload_csv(text):
    boundary = "moldweight-smoke-" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="integration.csv"\r\n'
        "Content-Type: text/csv\r\n\r\n"
        + text + f"\r\n--{boundary}--\r\n"
    ).encode("utf-8")
    req = Request(base_url + "/data/upload", data=body,
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urlopen(req, timeout=request_timeout) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, None


def serving_version(curve):
    status, result = request("/predict", {"curve": curve})
    version = result.get("model_version") if isinstance(result, dict) else None
    weight = result.get("predicted_weight") if isinstance(result, dict) else None
    if (status != 200 or not isinstance(version, str) or not version.isdigit()
            or type(weight) not in (int, float) or not math.isfinite(weight)):
        raise ValueError(f"통합 검증에는 실제 MLflow Production 모델이 필요합니다: {result}")
    return version


def integration_checks():
    # 새 trained 컨테이너에서만 실행: 업로드 파일·샷 버퍼·Production 버전을 변경합니다.
    normal_text = Path("data/sample_moldweight.csv").read_text(encoding="utf-8-sig")
    normal = load_shots("data/sample_moldweight.csv")
    drift = [shot for shot in load_shots("data/drift_shots_23.csv")
             if shot["cycle_counter"] >= 21122]  # F의 금형온도 70°C 시연 구간
    if len(normal) < WINDOW_SIZE or len(drift) < WINDOW_SIZE * 2:
        raise ValueError("통합 검증용 정상·70°C 드리프트 샷이 부족합니다")
    probe = normal[-1]["curve"]
    version_before = serving_version(probe)

    # 40행 파일은 저장하지 않고 거부해야 하며, 정상 파일은 /data/status에서 확인합니다.
    short_text = "\n".join(normal_text.splitlines()[:RETRAIN_SHOTS])
    status, _ = upload_csv(short_text)
    if status != 400:
        raise ValueError(f"40행 업로드는 400 필요 (HTTP {status})")
    status, uploaded = upload_csv(normal_text)
    if status != 200 or not isinstance(uploaded, dict) or uploaded.get("rows") != len(normal):
        raise ValueError(f"정상 CSV 업로드 실패: HTTP {status}, {uploaded}")
    status, summary = request("/data/status")
    expected = {
        "exists": True, "filename": uploaded["filename"], "rows": len(normal),
        "first_cycle": normal[0]["cycle_counter"], "last_cycle": normal[-1]["cycle_counter"],
        "weight_min": min(shot["actual_weight"] for shot in normal),
        "weight_max": max(shot["actual_weight"] for shot in normal),
    }
    if status != 200 or summary != expected:
        raise ValueError(f"업로드 요약 불일치: {summary}, expected={expected}")
    print(f"PASS integration upload: 40 rows rejected; {len(normal)} rows stored and summarized", flush=True)

    def batch(shots, label):
        status, result = request("/predict/batch-test", {"shots": shots},
                                 timeout=integration_timeout)
        predictions = result.get("predictions") if isinstance(result, dict) else None
        check = result.get("drift_check") if isinstance(result, dict) else None
        if (status != 200 or not isinstance(predictions, list) or len(predictions) != len(shots)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in predictions) or not isinstance(check, dict)):
            raise ValueError(f"{label}: 배치 응답 계약 불일치 HTTP {status}, {result}")
        print(f"integration {label}: {json.dumps(check, ensure_ascii=False)}", flush=True)
        return check

    warm = normal[-WINDOW_SIZE:]
    check = batch(warm[:-1], "normal20")
    if check != {"status": "collecting", "rmse": None, "window": WINDOW_SIZE - 1}:
        raise ValueError(f"빈 서버에서 정상 20샷은 판정 대기여야 합니다: {check}")
    check = batch(warm[-1:], "normal21")
    if (check.get("status") != "ok" or check.get("window") != WINDOW_SIZE
            or type(check.get("rmse")) not in (int, float)
            or not math.isfinite(check["rmse"]) or check["rmse"] > DRIFT_THRESHOLD):
        raise ValueError(f"정상 21샷 판정 실패: {check}")

    # 총 40샷에서 드리프트가 있어도 41샷 전에는 재학습하면 안 됩니다.
    guard_count = RETRAIN_SHOTS - WINDOW_SIZE - 1
    check = batch(drift[:guard_count], "drift40")
    if (check.get("status") != "collecting" or check.get("window") != WINDOW_SIZE
            or type(check.get("rmse")) not in (int, float)
            or not math.isfinite(check["rmse"]) or check["rmse"] <= DRIFT_THRESHOLD
            or serving_version(probe) != version_before):
        raise ValueError(f"40샷 재학습 보류/기존 서빙 버전 검증 실패: {check}")
    print("PASS integration guard: 20-shot window and 40-shot retraining guard", flush=True)

    promotions = 0
    rejections = 0
    before = version_before
    batches = [drift[guard_count:WINDOW_SIZE], drift[WINDOW_SIZE:WINDOW_SIZE * 2]]
    for index, shots in enumerate(batches, 1):
        check = batch(shots, f"drift_gate{index}")
        after = serving_version(probe)
        score = check.get("rmse")
        if (check.get("window") != WINDOW_SIZE or type(score) not in (int, float)
                or not math.isfinite(score)):
            raise ValueError(f"드리프트 RMSE/윈도우 계약 위반: {check}")
        if score > DRIFT_THRESHOLD:
            if check.get("status") != "retrain_triggered" or type(check.get("promoted")) is not bool:
                raise ValueError(f"41샷 확보 후 드리프트는 재학습 결과가 필요합니다: {check}")
            if check.get("model_version") != after:
                raise ValueError(f"게이트 결과와 실제 /predict 버전 불일치: {check}, serving={after}")
            if check["promoted"]:
                if after == before:
                    raise ValueError("승격 후 실제 서빙 버전이 전환되지 않았습니다")
                promotions += 1
            else:
                if after != before:
                    raise ValueError("게이트 실패 후 기존 서빙 버전이 변경됐습니다")
                rejections += 1
        elif check.get("status") != "ok" or after != before:
            raise ValueError(f"정상 판정에서 버전이 바뀌거나 잘못된 상태를 반환했습니다: {check}")
        print(f"PASS integration gate{index}: serving v{before} -> v{after}", flush=True)
        before = after
    if not promotions:
        raise ValueError("70°C 시연 2배치에서 Production 승격을 확인하지 못했습니다")

    status, logs = request("/logs/aiops.log")
    content = logs.get("content", "") if isinstance(logs, dict) else ""
    markers = ["[WARN] drift detected", "[INFO] retrain triggered", "[OK] gate passed"]
    if rejections:
        markers.append("[FAIL] gate failed")
    if status != 200 or any(marker not in content for marker in markers):
        raise ValueError(f"드리프트·재학습·게이트 로그 누락: HTTP {status}")
    print(f"PASS integration logs: promotions={promotions}, rejections={rejections}; "
          f"serving v{version_before} -> v{before}", flush=True)


try:
    wait_seconds = float(os.environ["WAIT_SECONDS"])
    request_timeout = float(os.environ["REQUEST_TIMEOUT"])
    samples = int(os.environ["P95_SAMPLES"])
    integration = os.environ["INTEGRATION_TEST"]
    if integration not in ("0", "1"):
        raise ValueError("INTEGRATION_TEST는 0 또는 1이어야 합니다")
    integration_timeout = float(os.environ["INTEGRATION_REQUEST_TIMEOUT"])
    if not math.isfinite(integration_timeout) or integration_timeout <= 0:
        raise ValueError("INTEGRATION_REQUEST_TIMEOUT은 0보다 큰 유한한 숫자여야 합니다")
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
    if integration == "1":
        integration_checks()
except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    sys.exit(1)
SMOKE_PY
