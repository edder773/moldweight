"""
업로드된 MoldWeight 데이터 파일 관리.

대시보드에서 CSV 파일을 직접 업로드합니다 (serving_app/routers/data.py 참고).
업로드된 파일은 이 디렉터리(data/uploads/)에 `moldweight_<타임스탬프>.csv` 이름으로
계속 쌓이고(과거 파일을 덮어쓰지 않습니다), 학습(train_and_register.py 등)은 항상
가장 최근에 올라온 파일 하나를 사용합니다.
"""
import glob
import os
import time

UPLOAD_DIR = "data/uploads"
UPLOAD_PREFIX = "moldweight_"


def save_upload(text: str, upload_dir: str = UPLOAD_DIR) -> str:
    """업로드된 CSV 내용을 data/uploads/moldweight_<타임스탬프>.csv 로 저장하고 경로를 반환한다.

    같은 초에 또 올라오면 moldweight_<타임스탬프>_1.csv 처럼 번호를 붙여, 기존 파일을 덮어쓰지 않는다.
    """
    os.makedirs(upload_dir, exist_ok=True)
    stem = f"{UPLOAD_PREFIX}{int(time.time())}"
    suffix = 0
    while True:
        name = f"{stem}.csv" if suffix == 0 else f"{stem}_{suffix}.csv"
        dest = os.path.join(upload_dir, name)
        try:
            with open(dest, "x", encoding="utf-8", newline="") as f:  # "x": 이미 있으면 실패
                f.write(text)
            return dest
        except FileExistsError:
            suffix += 1


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    """data/uploads/ 에 쌓인 CSV 중 가장 최근에 업로드된 파일의 경로를 반환한다."""
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 MoldWeight 데이터가 없습니다. 대시보드에서 CSV 파일을 먼저 업로드하세요 "
            f"(data/sample_moldweight.csv를 예시로 업로드해볼 수 있습니다 -> {upload_dir}/)."
        )
    return files[-1]
