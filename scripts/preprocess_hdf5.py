"""
원본 HDF5 → 학습·드리프트용 CSV 두 개 (RULES.md 3장 형식)

원본: scatimdata dataset2 (https://github.com/sc4t1m/scatimdata, CC BY 4.0)
      dataset2/dynamic_data_versuch_large.h5  — 저장소에 올리지 않음

출력: data/sample_moldweight.csv   실험 15·20 (526샷)  처음 학습·업로드·Docker 시드
      data/drift_shots_23.csv      실험 23    (303샷)  드리프트 주입

한 행 = 샷 하나
  cycle_counter, experiment, weight, inj_000 … inj_127, cav_000 … cav_127
  - inj = 사출압력(Einspritzdruck), cav = 금형 내 압력(Werkzeuginnendruck)
  - 곡선 2,048포인트를 앞에서부터 16개씩 묶어 평균 → 128포인트 (시간 순서 유지)

실행 (저장소 루트에서, 이 스크립트에만 pandas·tables 필요 — 서빙 requirements에는 넣지 않음)
  pip install pandas tables
  python scripts/preprocess_hdf5.py --h5 경로/dynamic_data_versuch_large.h5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving_app.config import CURVE_LEN  # 128

RAW_LEN = 2048
CHANNELS = [("inj", "Einspritzdruck", "einspritzdruck_ist_"),
            ("cav", "Werkzeuginnendruck", "werkzeuginnendruck_ist_")]
ALLOWED_EXPERIMENTS = {15, 20, 23}
SPLITS = [("data/sample_moldweight.csv", [15, 20], 526),   # (출력 경로, 실험 번호, 예상 샷 수)
          ("data/drift_shots_23.csv", [23], 303)]


def downsample(curve: np.ndarray) -> np.ndarray:
    """2,048포인트 → CURVE_LEN 포인트. 연속한 16개씩 평균."""
    return curve.reshape(CURVE_LEN, RAW_LEN // CURVE_LEN).mean(axis=1)


def build_table(h5_path: str) -> pd.DataFrame:
    shots = pd.read_hdf(h5_path, "scalars")[["cycle_counter", "Versuch", "weight"]]
    shots = shots.rename(columns={"Versuch": "experiment"}).sort_values("cycle_counter")
    shots["cycle_counter"] = shots["cycle_counter"].astype(int)
    shots["experiment"] = shots["experiment"].astype(int)

    # 샷 단위 검사: 번호 중복, 허용 실험 번호, 중량 결측·무한대
    dup = shots["cycle_counter"][shots["cycle_counter"].duplicated()].tolist()
    assert not dup, f"샷 번호 중복: {dup[:5]}"
    bad_exp = sorted(set(shots["experiment"]) - ALLOWED_EXPERIMENTS)
    assert not bad_exp, f"허용하지 않은 실험 번호: {bad_exp}"
    bad_w = shots.loc[~np.isfinite(shots["weight"].to_numpy(dtype=float)), "cycle_counter"].tolist()
    assert not bad_w, f"중량이 결측·무한대인 샷: {bad_w[:5]}"

    columns = {}
    for short, key, prefix in CHANNELS:
        curves = pd.read_hdf(h5_path, key)
        assert len(curves) == RAW_LEN, f"{key}: 곡선 길이 {len(curves)} (예상 {RAW_LEN})"
        for cc in shots["cycle_counter"]:
            assert f"{prefix}{cc}" in curves, f"{key}에 샷 {cc}의 곡선이 없음"
            values = curves[f"{prefix}{cc}"].to_numpy(dtype=float)
            assert np.isfinite(values).all(), f"{key} 샷 {cc}에 결측·무한대"
            columns.setdefault(cc, []).extend(np.round(downsample(values), 2))

    names = [f"{short}_{i:03d}" for short, _, _ in CHANNELS for i in range(CURVE_LEN)]
    curve_df = pd.DataFrame.from_dict(columns, orient="index", columns=names)
    table = shots.set_index("cycle_counter").join(curve_df).reset_index()
    table["weight"] = table["weight"].round(3)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="dynamic_data_versuch_large.h5 경로")
    args = parser.parse_args()

    table = build_table(args.h5)
    for path, experiments, expected in SPLITS:
        part = table[table["experiment"].isin(experiments)]
        assert len(part) == expected, f"{path}: {len(part)}샷 (예상 {expected}샷)"
        part.to_csv(path, index=False)
        print(f"{path}: {len(part)}샷, 실험 {experiments}, "
              f"무게 {part['weight'].min():.3f}~{part['weight'].max():.3f} g")


if __name__ == "__main__":
    main()
