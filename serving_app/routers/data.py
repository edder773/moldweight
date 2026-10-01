"""
MoldWeight 데이터 업로드 (RULES.md 4장 /data/upload, /data/status).

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""
import csv
import io
import math
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.storage import latest_upload, save_upload
from serving_app.config import CURVE_LEN, RETRAIN_SHOTS, VALUE_MAX, VALUE_MIN

router = APIRouter(prefix="/data")

# experiment는 기록용이라 있어도 되고 없어도 됨 (RULES 3-1)
REQUIRED_COLUMNS = (
    ["cycle_counter", "weight"]
    + [f"inj_{i:03d}" for i in range(CURVE_LEN)]
    + [f"cav_{i:03d}" for i in range(CURVE_LEN)]
)
MIN_ROWS = RETRAIN_SHOTS  # 재학습 한 번에 필요한 샷 수
CURVE_COLUMNS = REQUIRED_COLUMNS[2:]


def _invalid_value(row: dict) -> str | None:
    """한 행의 값이 학습·요약에서 읽을 수 있는지 확인하고, 문제가 있으면 사유를 돌려준다."""
    try:
        int(row["cycle_counter"])
    except (TypeError, ValueError):
        return f"cycle_counter가 정수가 아닙니다 ({row['cycle_counter']!r})"
    try:
        weight = float(row["weight"])
    except (TypeError, ValueError):
        return f"weight가 숫자가 아닙니다 ({row['weight']!r})"
    if not math.isfinite(weight):
        return f"weight가 숫자가 아닙니다 ({row['weight']!r})"
    for col in CURVE_COLUMNS:
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            return f"{col}이 숫자가 아닙니다 ({row[col]!r})"
        if not (VALUE_MIN <= value <= VALUE_MAX):  # NaN도 여기서 걸림
            return f"{col}={row[col]} 이 허용 범위 {VALUE_MIN}~{VALUE_MAX} 밖입니다"
    return None


@router.post(
    "/upload",
    summary="학습용 CSV 업로드",
    description=(
        "필수 컬럼: cycle_counter, weight, inj_000~inj_127, cav_000~cav_127 "
        "(experiment는 있어도 되고 없어도 됨). 최소 RETRAIN_SHOTS행."
    ),
    responses={
        200: {"content": {"application/json": {"example": {"filename": "moldweight_1727771234.csv", "rows": 526}}}},
        400: {"description": "UTF-8 아님, 필수 컬럼 누락, 행 수 부족, 숫자가 아니거나 범위 밖인 값"},
    },
)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")  # 엑셀에서 저장한 BOM 붙은 UTF-8도 허용
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        shown = ", ".join(missing[:5]) + (f" 외 {len(missing) - 5}개" if len(missing) > 5 else "")
        raise HTTPException(400, f"CSV에 필수 컬럼이 없습니다: {shown}")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다. (현재 {len(rows)}행)")
    for line_no, row in enumerate(rows, start=2):  # 1행은 헤더
        error = _invalid_value(row)
        if error:
            raise HTTPException(400, f"{line_no}행: {error}")

    dest = save_upload(text)
    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get(
    "/status",
    summary="가장 최근 업로드 파일 요약",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "exists": True,
                        "filename": "moldweight_1727771234.csv",
                        "rows": 526,
                        "first_cycle": 20191,
                        "last_cycle": 20897,
                        "weight_min": 114.48,
                        "weight_max": 116.69,
                    }
                }
            }
        }
    },
)
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    # 요약에는 샷 번호와 무게만 필요하므로 곡선은 읽지 않는다
    with open(path, encoding="utf-8-sig", newline="") as f:
        shots = [(int(r["cycle_counter"]), float(r["weight"])) for r in csv.DictReader(f)]
    cycles = [c for c, _ in shots]
    weights = [w for _, w in shots]
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(shots),
        "first_cycle": min(cycles),
        "last_cycle": max(cycles),
        "weight_min": min(weights),
        "weight_max": max(weights),
    }
