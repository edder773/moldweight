"""
사출 성형 샷 데이터를 LSTM 입력용 텐서로 변환하는 공용 유틸리티.

처음 학습(scripts/train_baseline_v1.py), MLflow 학습
(serving_app/train_and_register.py), 드리프트 후 fine-tuning 재학습
(monitoring/retrain_trigger.py)이 모두 이 모듈을 재사용합니다. 입력 정의를
한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는 실무 사고를
방지할 수 있습니다.

입력: 샷 하나의 압력 곡선 (CURVE_LEN=128, N_CHANNELS=2) - 채널 순서는 항상 [사출압력, 금형 내 압력]
타깃: 그 샷의 실측 무게 (g)

주가 예측 스켈레톤과 달리 슬라이딩 윈도우를 쓰지 않습니다. 한 샷이 이미 128포인트
곡선으로 완결된 샘플이고, 타깃은 같은 샷의 무게입니다 (미래 예측이 아닌 동시점 회귀).
"""
import csv
import pickle

from serving_app.config import CURVE_LEN, N_CHANNELS

# C(서빙 스키마)와 D(모니터링)가 곡선 길이를 참조할 때 이 모듈만 보면 되도록 재노출합니다.
__all__ = [
    "CURVE_LEN",
    "N_CHANNELS",
    "load_rows",
    "CurveScaler",
    "build_sequences",
    "train_test_split",
]

INJ_COLUMNS = [f"inj_{i:03d}" for i in range(CURVE_LEN)]
CAV_COLUMNS = [f"cav_{i:03d}" for i in range(CURVE_LEN)]
REQUIRED_COLUMNS = ["cycle_counter", "weight", *INJ_COLUMNS, *CAV_COLUMNS]


def load_rows(csv_path: str = "data/sample_moldweight.csv") -> list[dict]:
    """
    CSV(한 행 = 샷 하나)를 읽어 샷 dict 리스트로 돌려줍니다. cycle_counter 오름차순 정렬.

    반환 형식 (RULES 3-3):
        {"cycle_counter": int, "experiment": int | None, "weight": float,
         "curve": [[inj_0, cav_0], ..., [inj_127, cav_127]]}

    experiment는 기록용이며 모델 입력으로 쓰지 않습니다. 업로드 CSV에는 없을 수 있어
    선택 컬럼으로 다룹니다.
    """
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        rows = [
            {
                "cycle_counter": int(r["cycle_counter"]),
                "experiment": int(r["experiment"]) if r.get("experiment") else None,
                "weight": float(r["weight"]),
                "curve": [[float(r[inj]), float(r[cav])] for inj, cav in zip(INJ_COLUMNS, CAV_COLUMNS)],
            }
            for r in reader
        ]

    rows.sort(key=lambda r: r["cycle_counter"])
    return rows


class CurveScaler:
    """
    압력 곡선을 채널별로, 무게를 따로 [0, 1] 범위로 정규화하는 min-max 스케일러.

    LSTM은 스케일에 민감하므로(트리 기반 모델과 달리) 정규화가 필수입니다.
    sample_moldweight.csv로 한 번만 fit해 serving_app/models/scaler.pkl에 저장하고,
    MLflow 학습과 fine-tuning 모두 같은 스케일러를 재사용합니다.
    (재학습 시 다시 fit하지 않는 이유: 이미 이 스케일로 학습된 모델 가중치와 어긋나면
     fine-tuning 자체가 무의미해지기 때문입니다.)

    채널 min/max는 전 샷·전 시점을 합쳐 채널마다 하나씩 구합니다. 시점별로 따로 잡으면
    곡선의 시간적 모양이 뭉개져 LSTM이 볼 신호가 사라집니다.
    """

    def __init__(self):
        self.channel_min: list[float] | None = None  # [inj_min, cav_min]
        self.channel_max: list[float] | None = None  # [inj_max, cav_max]
        self.weight_min: float | None = None
        self.weight_max: float | None = None

    def fit(self, rows: list[dict]) -> "CurveScaler":
        lows = [float("inf")] * N_CHANNELS
        highs = [float("-inf")] * N_CHANNELS
        for row in rows:
            for point in row["curve"]:
                for ch in range(N_CHANNELS):
                    if point[ch] < lows[ch]:
                        lows[ch] = point[ch]
                    if point[ch] > highs[ch]:
                        highs[ch] = point[ch]

        weights = [r["weight"] for r in rows]
        self.channel_min, self.channel_max = lows, highs
        self.weight_min, self.weight_max = min(weights), max(weights)
        return self

    @staticmethod
    def _scale(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    @staticmethod
    def _unscale(value: float, lo: float, hi: float) -> float:
        return value * (hi - lo) + lo

    def transform_curve(self, curve: list[list[float]]) -> list[list[float]]:
        """곡선 128×2를 채널별로 정규화합니다. 반환 크기도 128×2."""
        return [
            [self._scale(point[ch], self.channel_min[ch], self.channel_max[ch]) for ch in range(N_CHANNELS)]
            for point in curve
        ]

    def scale_weight(self, weight: float) -> float:
        """타깃 무게를 학습용으로 정규화합니다.

        값을 [0, 1]로 자르지 않습니다. 드리프트 데이터(실험 23)는 무게가 학습 범위
        아래로 내려가 음수가 나오는데, 여기서 clip하면 fine-tuning이 범위 밖을 학습할
        길이 막혀 재학습이 무력해집니다.
        """
        return self._scale(weight, self.weight_min, self.weight_max)

    def inverse_weight(self, scaled_weight: float) -> float:
        """모델이 뱉은 정규화된 예측값을 실제 g 단위 무게로 되돌립니다."""
        return self._unscale(scaled_weight, self.weight_min, self.weight_max)

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "CurveScaler":
        scaler = cls()
        with open(path, "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        return scaler


def build_sequences(rows: list[dict], scaler: CurveScaler):
    """
    샷 리스트에서 정규화된 입력 곡선과 무게 타깃을 만듭니다. 샷 하나가 샘플 하나입니다.

    반환: X (n_samples, CURVE_LEN, N_CHANNELS), y (n_samples,) - y는 스케일 안 된 실제 무게(g)
    """
    X = [scaler.transform_curve(r["curve"]) for r in rows]
    y = [r["weight"] for r in rows]
    return X, y


def train_test_split(X: list, y: list, rows: list[dict], test_ratio: float = 0.2):
    """
    실험별로 층화한 뒤 각 실험 안에서 시간순으로 앞을 train, 뒤를 test로 나눕니다.

    단순히 전체를 시간순으로 자르면 검증셋이 뒤쪽 실험 하나로만 채워집니다
    (sample_moldweight.csv는 실험 15가 앞 303샷, 실험 20이 뒤 223샷). 두 실험의 평균
    무게가 0.72 g 차이 나는데 이는 목표 RMSE 0.3 g의 2.4배여서, 그렇게 나누면 검증
    RMSE가 모델 성능이 아니라 실험 간 분포 차이를 재게 됩니다.

    실험 안에서는 생산 순서를 유지하므로 같은 실험 내 미래 데이터 누수는 없습니다.
    experiment가 없는 업로드 CSV는 전체를 한 덩어리로 보고 시간순 분할합니다.
    """
    groups: dict[object, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["experiment"], []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, idx in sorted(groups.items(), key=lambda kv: (kv[0] is not None, kv[0])):
        split_at = int(len(idx) * (1 - test_ratio))
        train_idx += idx[:split_at]
        test_idx += idx[split_at:]

    train_idx.sort()
    test_idx.sort()
    return (
        [X[i] for i in train_idx],
        [y[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in test_idx],
    )
