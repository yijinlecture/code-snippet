import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hmmlearn: 연속값(가우시안) 기반 HMM
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# =========================================
# 0) 피처 만들기 (간단 파생변수)
# =========================================
def build_features(df: pd.DataFrame, value_col: str = "R", win: int = 6) -> pd.DataFrame:
    """
    R 시계열에서 HMM에 넣을 기본 파생변수를 만듭니다.
    - dR: 한 칸 전 대비 변화량
    - cum_inc: 최근 win개 구간에서 처음과 끝의 차이(=누적상승/하락)
    - roll_std: 최근 win개 표준편차(변동성)
    """
    fdf = df.copy()
    s = fdf[value_col].astype(float)

    fdf["dR"] = s.diff().fillna(0.0)
    fdf["cum_inc"] = s.rolling(win).apply(lambda x: x[-1] - x[0], raw=True)
    fdf["cum_inc"] = fdf["cum_inc"].bfill().ffill()
    fdf["roll_std"] = s.rolling(win).std()
    fdf["roll_std"] = fdf["roll_std"].bfill().ffill()

    return fdf
    # ⟨여기⟩ win(슬라이딩 윈도우 크기): 5분 간격이면 6=30분, 12=60분 등으로 조정해보세요.


def standardize_features(fdf: pd.DataFrame, feat_cols=("R", "dR", "cum_inc", "roll_std")):
    """
    HMM은 값의 스케일에 민감합니다. 표준화(평균0, 분산1)로 맞춰줍니다.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(fdf[list(feat_cols)].values)
    return X, scaler
    # ⟨여기⟩ feat_cols: 처음엔 단순하게 쓰고, 필요하면 조작변수 요약 피처(예: 상위 k개)도 추가해 보세요.


# =========================================
# 1) HMM 학습 + 상태 추론
# =========================================
def fit_hmm(X: np.ndarray, n_states: int = 4, cov_type: str = "full",
            n_iter: int = 200, random_state: int = 42) -> GaussianHMM:
    """
    HMM 학습(EM). 비지도 방식이라 라벨 없이도 상태를 찾아줍니다.
    """
    model = GaussianHMM(
        n_components=n_states,      # 상태 개수(2~4부터 시도 권장) ⟨여기⟩
        covariance_type=cov_type,   # "diag"가 빠르고 안정적일 때도 많음 ⟨여기⟩
        n_iter=n_iter,              # EM 반복 횟수(수렴 안 되면 늘리기) ⟨여기⟩
        random_state=random_state
    )
    model.fit(X)
    return model


def decode_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Viterbi로 가장 그럴듯한 상태 시퀀스를 구합니다.
    """
    return model.predict(X)


# =========================================
# 2) 상태 번호 → 사람이 읽기 쉬운 라벨
# =========================================
def map_states_to_labels(model: GaussianHMM,
                         feat_names=("R", "dR", "cum_inc", "roll_std"),
                         verbose: bool = True):
    """
    상태별 평균(표준화 스케일)을 보고 간단 규칙으로 라벨링합니다.
    - 실제 데이터에 맞춰 경계값을 조금씩 조정하세요.
    """
    means = model.means_

    if verbose:
        print("=== 상태별 평균(표준화) ===")
        for k, mu in enumerate(means):
            print(f"state {k}: ", {n: round(v, 3) for n, v in zip(feat_names, mu)})

    def label_fn(state_id: int) -> str:
        r, dr, ci, rs = means[state_id]
        # ⟨여기⟩ 아래 임계값은 경험적으로 조정하세요(데이터마다 다릅니다)
        if (r > 0.8) and (abs(dr) < 0.2):
            return "비정상"          # 수준이 높고 안정적
        if (ci > 0.6) or (dr > 0.5):
            return "정상→비정상"     # 상승 전이
        if (ci < -0.6) or (dr < -0.5):
            return "비정상→정상"     # 하강 전이
        return "정상"

    return label_fn


# =========================================
# 3) 시각화
# =========================================
def plot_states(df: pd.DataFrame, value_col: str = "R",
                state_col: str = "hmm_state", label_col: str = "hmm_label",
                title: str = "HMM 시계열 상태 구분", colors=None):
    """
    원 시계열 위에 상태 라벨을 색으로 뿌려서 한눈에 보기 좋게 그립니다.
    """
    if colors is None:
        colors = {"정상": "green", "정상→비정상": "orange", "비정상": "red", "비정상→정상": "blue"}

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[value_col], lw=1, alpha=0.5, color="gray", label=value_col)

    for lab, c in colors.items():
        part = df[df[label_col] == lab]
        ax.scatter(part.index, part[value_col], s=10, color=c, label=lab)

    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.legend(loc="upper left", ncol=len(colors))
    plt.tight_layout()
    plt.show()


# =========================================
# 4) 실행 예시 (샘플 시계열)
# =========================================
if __name__ == "__main__":
    # (A) 샘플 데이터: 정상 → 상승전이 → 비정상 → 하강전이
    np.random.seed(7)
    time = pd.date_range("2025-01-01", periods=240, freq="5min")
    R = np.concatenate([
        np.random.normal(100, 2, 60),
        np.linspace(100, 128, 60) + np.random.normal(0, 1, 60),
        np.random.normal(130, 2.5, 60),
        np.linspace(130, 102, 60) + np.random.normal(0, 1, 60),
    ])
    df = pd.DataFrame({"time": time, "R": R}).set_index("time")

    # (B) 피처 만들기 + 표준화
    df_feat = build_features(df, value_col="R", win=6)  # ⟨여기⟩ win=6(30분). 데이터 주기/현장감으로 조정
    X, scaler = standardize_features(df_feat)

    # (C) HMM 학습
    hmm = fit_hmm(X, n_states=4, cov_type="full", n_iter=200, random_state=42)
    # ⟨여기⟩ n_states: 2(정상/비정상) → 3(전이 포함) → 4(상하 전이 분리) 순으로 늘려보기
    # ⟨여기⟩ cov_type: "diag"로 바꾸면 빠르고 과적합 줄 때도 있음
    # ⟨여기⟩ n_iter: 수렴 경고나면 300~500으로 올려보기

    # (D) 상태 추론
    df_feat["hmm_state"] = decode_states(hmm, X)

    # (E) 라벨 매핑
    label_fn = map_states_to_labels(hmm, verbose=True)
    df_feat["hmm_label"] = df_feat["hmm_state"].apply(label_fn)
    # ⟨여기⟩ map_states_to_labels 내부의 임계값(0.8, 0.6 등)을 실제 데이터 통계 보면서 조정

    # (F) 시각화
    plot_states(df_feat, value_col="R", state_col="hmm_state", label_col="hmm_label")

    # (G) 상태별 통계(빠른 감각 잡기)
    print("\n=== 상태별 피처 요약 ===")
    print(df_feat.groupby("hmm_label")[["R", "dR", "cum_inc", "roll_std"]].describe())
