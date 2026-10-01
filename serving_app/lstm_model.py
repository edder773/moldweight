"""
사출 성형 무게 예측용 LSTM 아키텍처 (처음 학습과 MLflow 학습이 공유).

구조: LSTM(32) -> Dense(1), 파라미터 4,513개.

왜 이 구조인가 (실험 15·20 526샷으로 측정, 실험별 층화 시간순 분할, 시드 42/7/123 평균):

| 후보                        | 파라미터 | 검증 RMSE(g) |
|-----------------------------|---------:|-------------:|
| LSTM32 x3 + Dense16 (원본)  |   16,225 |       0.1687 |
| LSTM(8)                     |      361 |       0.1872 |
| LSTM(16)                    |    1,233 |       0.1530 |
| LSTM(32)  <- 선택           |    4,513 |       0.1344 |
| LSTM(64)                    |   17,089 |       0.1448 |
| LSTM(32) + GlobalAvgPool    |    4,513 |       0.1560 |

- 3층 적층(원본 스켈레톤)은 128스텝 시퀀스에서 오히려 손해입니다. 1층이 더 좋습니다.
- 폭은 32가 최적점입니다. 8은 용량 부족, 64는 분산만 커지고 이득이 없습니다.
- 시드 간 분산폭도 32가 가장 작아(0.010) 시연에서 운에 좌우될 위험이 가장 낮습니다.

학습 길이는 config.BASE_EPOCHS를 따릅니다. 200 epoch 지점에서 train/val 격차가 최소
(-0.009)이고 검증 RMSE도 최소입니다. 400 epoch 이상에서는 train만 내려가고 검증이
되돌아 올라가 과적합이 시작됩니다.

참고: 곡선-무게 관계는 거의 선형입니다(cav 후반부와 무게의 상관계수 r≈0.99, 선형회귀
RMSE 0.045 g). LSTM은 이 과제의 최적 도구는 아니지만 곡선 전체를 입력으로 받는
시계열 서빙 구조를 익히는 것이 실습 목표이므로 LSTM을 씁니다.
"""
from tensorflow import keras

from serving_app.config import CURVE_LEN, N_CHANNELS

LSTM_UNITS = 32


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(CURVE_LEN, N_CHANNELS)),  # (128, 2) = [사출압력, 금형 내 압력]
            keras.layers.LSTM(LSTM_UNITS),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
