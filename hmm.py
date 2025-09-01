import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hmmlearn 라이브러리에서 가우시안 HMM 모델을 불러옵니다.
# HMM은 관측된 데이터(X) 뒤에 숨겨진(hidden) 상태(state)가 있다고 가정하는 확률 모델입니다.
# 가우시안 HMM은 각 상태의 관측 데이터 분포가 가우시안 분포(정규 분포)를 따른다고 가정합니다.
from hmmlearn.hmm import GaussianHMM

# 데이터 전처리(피처 표준화)를 위해 scikit-learn의 StandardScaler를 사용합니다.
from sklearn.preprocessing import StandardScaler

# =========================
# 0) 피처 엔지니어링 함수
# =========================

def build_features(df: pd.DataFrame,
                   value_col: str = "R",
                   win: int = 6) -> pd.DataFrame:
    """
    원본 시계열 데이터로부터 HMM이 상태를 더 잘 구분할 수 있도록 돕는 파생 피처(특성)들을 생성합니다.

    Args:
        df (pd.DataFrame): 원본 시계열 데이터프레임. 인덱스는 시간 정보여야 합니다.
        value_col (str): 피처를 생성할 기준 컬럼명 (예: "R" 값).
        win (int): 슬라이딩 윈도우의 크기. roll_std, cum_inc 계산에 사용됩니다.

    Returns:
        pd.DataFrame: 원본 데이터에 파생 피처들이 추가된 데이터프레임.
    """
    fdf = df.copy()  # 원본 데이터프레임의 수정을 방지하기 위해 복사본을 생성합니다.
    s = fdf[value_col].astype(float) # 계산을 위해 시리즈(Series) 데이터를 float 타입으로 변환합니다.

    # 1. 1차 차분 (dR): 현재 값과 이전 값의 차이를 계산하여 데이터의 '변화율' 또는 '속도'를 나타냅니다.
    #    - 상태가 급격히 변할 때 dR 값이 커지므로, 전이 상태를 감지하는 데 중요한 피처가 됩니다.
    #    - fillna(0.0)은 첫 번째 데이터의 차분 값이 NaN이 되므로 이를 0으로 채웁니다.
    fdf["dR"] = s.diff().fillna(0.0)

    # 2. 누적 증가량 (cum_inc): 정해진 윈도우(win) 내에서 마지막 값과 첫 값의 차이를 계산합니다.
    #    - 특정 기간 동안의 '추세'를 나타냅니다. 꾸준히 증가하거나 감소하는 상태를 감지하는 데 유용합니다.
    #    - raw=True 옵션은 numpy 배열로 계산하여 속도를 향상시킵니다.
    fdf["cum_inc"] = s.rolling(win).apply(lambda x: x[-1] - x[0], raw=True)
    #    - rolling 연산 초기에 발생하는 NaN 값들을 처리합니다.
    #    - bfill(): 뒤의 값으로 채움. ffill(): 앞의 값으로 채움. 두 개를 연달아 써서 모든 NaN을 제거합니다.
    fdf["cum_inc"] = fdf["cum_inc"].bfill().ffill()

    # 3. 롤링 표준편차 (roll_std): 윈도우 내 데이터의 '변동성' 또는 '안정성'을 측정합니다.
    #    - 정상 상태에서는 변동성이 낮고(작은 값), 비정상 또는 전이 상태에서는 변동성이 클 수 있습니다(큰 값).
    fdf["roll_std"] = s.rolling(win).std()
    #    - 마찬가지로 초반 NaN 값들을 채워줍니다.
    fdf["roll_std"] = fdf["roll_std"].bfill().ffill()

    return fdf


def standardize_features(fdf: pd.DataFrame,
                         feat_cols=("R", "dR", "cum_inc", "roll_std")):
    """
    생성된 피처들을 표준화합니다. (평균=0, 분산=1)
    HMM, 특히 가우시안 HMM은 각 피처의 스케일에 민감합니다.
    예를 들어, 'R' 값(100~130)이 'dR' 값(-2~2)보다 훨씬 크면, 모델이 'R' 피처에만 과도하게 의존할 수 있습니다.
    표준화를 통해 모든 피처가 동등한 스케일로 모델 학습에 기여하도록 만듭니다.

    Args:
        fdf (pd.DataFrame): 피처가 포함된 데이터프레임.
        feat_cols (tuple): 표준화를 적용할 피처 컬럼들의 이름.

    Returns:
        np.ndarray: 표준화된 피처 데이터 (HMM 입력용).
        StandardScaler: 학습된 스케일러 객체 (나중에 새로운 데이터를 변환할 때 필요).
    """
    scaler = StandardScaler()
    # feat_cols에 지정된 컬럼들의 값만 추출하여 numpy 배열로 만듭니다.
    X = scaler.fit_transform(fdf[list(feat_cols)].values)
    return X, scaler


# =========================
# 1) HMM 학습 및 추론
# =========================

def fit_hmm(X: np.ndarray,
            n_states: int = 4,
            cov_type: str = "full",
            n_iter: int = 200,
            random_state: int = 42) -> GaussianHMM:
    """
    주어진 데이터(X)를 사용하여 HMM 모델을 학습(fitting)시킵니다.
    이 과정은 '바움-웰치(Baum-Welch)' 알고리즘을 통해 모델의 파라미터들
    (상태 전이 확률, 각 상태에서의 관측 확률 분포 등)을 데이터에 가장 잘 맞도록 추정합니다.

    Args:
        X (np.ndarray): 표준화된 피처 데이터.
        n_states (int): 모델이 가정할 숨겨진 상태의 개수. (예: 4개 -> 정상, 비정상, 2개의 전이 상태)
        cov_type (str): 각 상태의 가우시안 분포가 가질 공분산 행렬의 타입.
                        - "full": 각 상태가 완전한 공분산 행렬을 가져 피처 간의 복잡한 상관관계를 모델링할 수 있습니다.
                        - "diag": 대각 공분산 행렬을 가정하여 피처 간 독립을 가정합니다. (계산량 감소)
        n_iter (int): 모델 파라미터를 최적화하기 위한 반복 학습 횟수.
        random_state (int): 모델 내부의 무작위 초기화를 제어하여, 실행할 때마다 동일한 결과를 얻도록 합니다.

    Returns:
        GaussianHMM: 학습이 완료된 HMM 모델 객체.
    """
    model = GaussianHMM(
        n_components=n_states,      # 숨겨진 상태의 개수
        covariance_type=cov_type,   # 공분산 타입
        n_iter=n_iter,              # 학습 반복 횟수
        random_state=random_state   # 결과 재현성을 위한 시드
    )
    # fit() 메소드를 호출하여 모델을 학습시킵니다.
    model.fit(X)
    return model


def decode_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    학습된 HMM 모델과 관측 시계열 데이터(X)를 사용하여, 각 시점(time step)에 해당하는
    가장 가능성 높은 숨겨진 상태(hidden state)를 추론합니다.
    이 과정은 '비터비(Viterbi)' 알고리즘을 사용합니다.

    Args:
        model (GaussianHMM): 학습된 HMM 모델.
        X (np.ndarray): 상태를 추론할 표준화된 피처 데이터.

    Returns:
        np.ndarray: 각 시점별로 추론된 상태 번호(0, 1, 2, ...)의 시퀀스.
    """
    # predict() 메소드는 내부적으로 비터비 알고리즘을 실행합니다.
    states = model.predict(X)
    return states


# =========================
# 2) 상태 라벨링
# =========================

def map_states_to_labels(model: GaussianHMM,
                         feat_names=("R", "dR", "cum_inc", "roll_std"),
                         verbose: bool = True):
    """
    HMM이 찾아낸 숫자 상태(0, 1, 2, ...)를 사람이 이해할 수 있는 의미있는 라벨
    (예: "정상", "비정상")로 매핑하는 규칙(함수)을 생성합니다.
    각 상태가 어떤 특징을 가지는지 확인하기 위해, 상태별 피처들의 평균값을 출력하고 이를 기준으로 라벨링 규칙을 정의합니다.

    Args:
        model (GaussianHMM): 학습된 HMM 모델.
        feat_names (tuple): 피처 이름들.
        verbose (bool): 상태별 평균값을 출력할지 여부.

    Returns:
        function: 상태 번호(int)를 입력받아 라벨(str)을 반환하는 함수.
    """
    # model.means_ 속성은 각 상태(행)별 각 피처(열)의 평균값을 담고 있습니다.
    # 이 값들은 표준화된 스케일에서의 평균값입니다.
    means = model.means_

    if verbose:
        # 각 상태가 어떤 물리적 의미를 갖는지 해석하기 위해 평균값을 출력합니다.
        print("=== 상태별 평균 (표준화 스케일) ===")
        for k, mu in enumerate(means):
            print(f"state {k}: ", {name: round(val, 3)
                                   for name, val in zip(feat_names, mu)})

    # 상태 번호를 라벨로 변환하는 함수를 내부에서 정의합니다.
    # 이 조건문들은 `means` 값을 보고 사용자가 직접 정의해야 하는 부분입니다.
    # 예를 들어, R 값이 높고(>0.8) 변화율(dR)이 거의 없는(<0.2) 상태는 '비정상'으로 해석할 수 있습니다.
    def label_fn(state_id: int) -> str:
        # 해당 상태의 평균 피처 값들을 가져옵니다.
        r, dr, ci, rs = means[state_id]
        if (r > 0.8) and (abs(dr) < 0.2):
            return "비정상"
        if (ci > 0.6) or (dr > 0.5):
            return "정상→비정상"
        if (ci < -0.6) or (dr < -0.5):
            return "비정상→정상"
        return "정상" # 위의 어떤 조건에도 해당하지 않으면 '정상' 상태로 간주

    return label_fn


# =========================
# 3) 시각화 함수
# =========================

def plot_states(df: pd.DataFrame,
                value_col: str = "R",
                state_col: str = "hmm_state",
                label_col: str = "hmm_label",
                title: str = "HMM 시계열 상태 구분",
                colors=None):
    """
    HMM에 의해 분류된 상태를 원본 시계열 데이터 위에 색상으로 시각화합니다.
    어떤 구간이 어떤 상태로 분류되었는지 직관적으로 파악할 수 있습니다.

    Args:
        df (pd.DataFrame): 시각화할 데이터 (원본 값과 상태 라벨 포함).
        value_col (str): Y축에 표시할 원본 데이터 컬럼명.
        state_col (str): 상태 번호 컬럼명 (사용되진 않지만, 정보 제공 차원).
        label_col (str): 상태 라벨 컬럼명 (색상 구분에 사용).
        title (str): 그래프 제목.
        colors (dict): 라벨별 색상을 지정하는 딕셔너리.
    """
    if colors is None:
        # 기본 색상 팔레트를 정의합니다.
        colors = {
            "정상": "green",
            "정상→비정상": "orange",
            "비정상": "red",
            "비정상→정상": "blue"
        }

    fig, ax = plt.subplots(figsize=(14, 4))
    # 원본 시계열 데이터를 옅은 회색 선으로 먼저 그립니다.
    ax.plot(df.index, df[value_col], lw=1, alpha=0.5, color='gray', label=f"Original {value_col}")

    # 각 라벨별로 순회하며 해당되는 데이터를 다른 색상의 점(scatter)으로 덧그립니다.
    for lab, c in colors.items():
        part = df[df[label_col] == lab]
        ax.scatter(part.index, part[value_col], s=10, label=lab, color=c)

    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.legend(loc="upper left", ncol=len(colors)) # 범례 표시
    plt.tight_layout() # 그래프 레이아웃 최적화
    plt.show()


# =========================
# 4) 실행 예시
# =========================

# 이 스크립트가 직접 실행될 때만 아래 코드가 동작하도록 합니다.
if __name__ == "__main__":
    # 1. 예시 시계열 데이터 생성
    #    HMM이 잘 동작하는지 테스트하기 위해 의도적으로 4개의 다른 상태를 가진 데이터를 생성합니다.
    np.random.seed(7) # 결과 재현을 위한 시드 고정
    time = pd.date_range("2025-01-01", periods=240, freq="5min")
    R = np.concatenate([
        np.random.normal(100, 2, 60),      # 상태 1: '정상' (평균 100, 낮은 변동성)
        np.linspace(100, 128, 60) + np.random.normal(0, 1, 60), # 상태 2: '정상→비정상' (선형적으로 증가)
        np.random.normal(130, 2.5, 60),    # 상태 3: '비정상' (평균 130, 다소 높은 변동성)
        np.linspace(130, 102, 60) + np.random.normal(0, 1, 60)  # 상태 4: '비정상→정상' (선형적으로 감소)
    ])
    df = pd.DataFrame({"time": time, "R": R}).set_index("time")

    # 2. 피처 생성 및 표준화 (위에서 정의한 함수 사용)
    df_feat = build_features(df, value_col="R", win=6)
    X, scaler = standardize_features(df_feat)

    # 3. HMM 모델 학습
    hmm = fit_hmm(X, n_states=4, cov_type="full", n_iter=200, random_state=42)

    # 4. 숨겨진 상태 추론
    states = decode_states(hmm, X)
    df_feat["hmm_state"] = states # 추론된 상태 번호를 데이터프레임에 추가

    # 5. 상태 번호를 의미있는 라벨로 변환
    label_fn = map_states_to_labels(hmm, verbose=True) # 라벨링 함수 생성
    df_feat["hmm_label"] = df_feat["hmm_state"].apply(label_fn) # 각 상태 번호에 라벨링 함수 적용

    # 6. 결과 시각화
    plot_states(df_feat, value_col="R", state_col="hmm_state", label_col="hmm_label")

    # 7. 상태별 통계량 출력
    #    groupby()를 사용하여 각 라벨별로 피처들이 어떤 통계적 특징을 보이는지 확인합니다.
    #    이는 map_states_to_labels 함수에서 정의한 규칙이 타당한지 검증하는 데 도움을 줍니다.
    summary = df_feat.groupby("hmm_label")[["R", "dR", "cum_inc", "roll_std"]].describe()
    print("\n=== 상태별 피처 요약 ===")
    print(summary)
