
import argparse
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# =========================
# 1. 기본 설정
# =========================
#
# 지금까지의 센서 값으로 10분 뒤 공기질(기본: eCO2, TVOC)을 예측하는 모델을 학습한다.
# 입력은 m701_state_clean.py --all 이 만든 data/labeled/clean_state_all.csv 이다.
#
# 학습: 시뮬레이션 데이터(aug_sim_*)
# 시험: 실제로 측정한 데이터(aug_ 로 시작하지 않는 세션)
#
# 10분 뒤 값을 바로 맞히는 대신 "지금보다 얼마나 오르거나 내릴지"를 맞힌다.
# 센서마다 기준값이 달라도(예: eCO2 470 vs 500) 같은 규칙을 쓸 수 있기 때문이다.
#   예측값 = 지금 값 + 모델이 예측한 변화량

INPUT_FILE = os.path.join("data", "labeled", "clean_state_all.csv")
MODEL_DIR = "models"
RESULT_DIR = "results"

TARGETS = {
    "eco2": "eCO2 (ppm)",
    "tvoc": "TVOC",
    "pm25": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "temperature": "온도 (°C)",
    "humidity": "습도 (%)",
}

INPUTS = ["eco2", "tvoc", "humidity"]  # 특징을 만들 센서
LAGS = [1, 5, 10]                      # 몇 분 전과 비교할지
BASELINE_MINUTES = 30                  # 최근 30분 최저값을 "평소 값"으로 본다

# 10분 동안 이만큼 넘게 변하면 "큰 변화" 구간으로 따로 채점한다.
BIG_CHANGE = {"eco2": 30, "tvoc": 5, "pm25": 3, "pm10": 3,
              "temperature": 0.5, "humidity": 2}

# 그레이디언트 부스팅: 작은 결정 트리를 여러 개 이어 붙여, 앞 트리가 틀린 만큼을
# 다음 트리가 고쳐 나가는 방법. "절대 오차"로 학습하면 평온할 때 괜히
# 변화를 예측하지 않아서, 대부분 평온한 교실 데이터에서 더 잘 맞았다.
MODELS = {
    "linear": ("선형 회귀", lambda: LinearRegression()),
    "tree": ("결정 트리", lambda: DecisionTreeRegressor(
        max_depth=6, min_samples_leaf=20, random_state=0)),
    "boosting": ("그레이디언트 부스팅", lambda: HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=4, random_state=0)),
}


def step(number, title):
    print(f"\n===== [{number}/6] {title} =====")


def pause(args):
    if not args.no_pause:
        input("\n다음 단계로 가려면 Enter를 누르세요...")


# [##########----------]  50% (15/30) 설명  처럼 진행률을 한 줄에 보여 준다.
def progress(done, total, text="", width=20):

    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done == total else ""
    print(f"\r  [{bar}] {done / total * 100:3.0f}% ({done}/{total}) {text:<40}",
          end=end, flush=True)


def show(fig, path, args):

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"저장: {path}")

    if not args.no_pause and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


# =========================
# 2. 특징과 정답 만들기 (세션별)
# =========================

def feature_names(target, with_state):

    names = []

    for name in INPUTS:
        names.append(f"{name}_now")
        names += [f"{name}_change_{lag}m" for lag in LAGS]

    for name in ["eco2", "tvoc"]:
        names.append(f"{name}_rise")

    if target not in INPUTS:
        names.append(f"{target}_now")
        names += [f"{target}_change_{lag}m" for lag in LAGS]

    if with_state:
        names.append("printing")

    return names


def add_features(group, target, horizon):

    group = group.set_index("time").sort_index()
    out = pd.DataFrame(index=group.index)
    out["session"] = group["session"]
    out["printing"] = (group["state"] == "printing").astype(int)

    for name in dict.fromkeys(INPUTS + [target]):
        now = group[f"{name}_smooth"]
        out[f"{name}_now"] = now

        for lag in LAGS:
            before = now.shift(freq=f"{lag}min").reindex(now.index)
            out[f"{name}_change_{lag}m"] = now - before

    for name in ["eco2", "tvoc"]:
        now = group[f"{name}_smooth"]
        out[f"{name}_rise"] = now - now.rolling(f"{BASELINE_MINUTES}min").min()

    # 정답: 10분 뒤 값과 지금 값의 차이
    now = group[f"{target}_smooth"]
    future = now.shift(freq=f"-{horizon}min").reindex(now.index)
    out["future"] = future
    out["target_change"] = future - now

    return out.reset_index()


# =========================
# 3. 평가와 그래프
# =========================

def scores(actual, predict):
    mae = mean_absolute_error(actual, predict)
    rmse = np.sqrt(mean_squared_error(actual, predict))
    return mae, rmse


def plot_test(test, target, horizon, model_title, path, args):

    sessions = [s for s in test["session"].unique()
                if (test["session"] == s).sum() > 30]
    fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2.8 * len(sessions)),
                             squeeze=False)

    for ax, session in zip(axes[:, 0], sessions):
        part = test[test["session"] == session]
        # 예측은 "10분 뒤" 값이므로 10분 뒤 시각에 그려야 실제 값과 겹쳐 볼 수 있다.
        later = part["time"] + pd.Timedelta(minutes=horizon)
        ax.plot(later, part["future"], color="black", lw=1, label="실제 값")
        ax.plot(later, part["predict_persist"], color="gray", lw=1, ls="--",
                label="지속 예측 (10분 전 값)")
        ax.plot(later, part["predict"], color="C3", lw=1.2,
                label=f"{model_title} 예측")
        ax.set_title(session, fontsize=10)
        ax.set_ylabel(TARGETS[target])

    axes[0, 0].legend(loc="upper right", fontsize=8)
    show(fig, path, args)


# =========================
# 4. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE,
                        help="전처리된 라벨링 데이터 (m701_state_clean.py --all 결과)")
    parser.add_argument("--target", nargs="+", default=["eco2", "tvoc"],
                        choices=list(TARGETS),
                        help="예측할 값 (여러 개 가능, 예: --target eco2 tvoc humidity)")
    parser.add_argument("--horizon", type=int, default=10,
                        help="몇 분 뒤를 예측할지")
    parser.add_argument("--model", default="boosting", choices=list(MODELS),
                        help="저장할 모델 종류 (세 가지 모두 학습해 비교는 함)")
    parser.add_argument("--with-state", action="store_true",
                        help="프린터 가동 여부(라벨)도 특징으로 쓰기")
    parser.add_argument("--with-variants", action="store_true",
                        help="실제 데이터를 바꾼 aug_real_* 도 학습에 넣기 "
                             "(시험 데이터와 겹쳐 점수가 부풀려질 수 있음)")
    parser.add_argument("--no-pause", action="store_true",
                        help="멈추지 않고 끝까지 실행")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"입력 파일이 없습니다: {args.input}")
        print("먼저 python m701_state_clean.py --all 을 실행하세요.")
        return

    # 그래프의 한글 표시용. 라즈베리파이: sudo apt install fonts-nanum
    plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 같은 항목을 두 번 적어도 한 번만 학습한다.
    targets = list(dict.fromkeys(args.target))

    # 1) 불러오기
    step(1, "전처리된 데이터 불러오기")
    df = pd.read_csv(args.input, parse_dates=["time"])
    print(f"파일: {args.input}")
    print(f"크기: {len(df)}행, 세션 {df['session'].nunique()}개")
    print(f"예측할 값: {args.horizon}분 뒤 " + ", ".join(TARGETS[t] for t in targets))
    pause(args)

    # 항목마다 2~6단계를 반복해 models/forecast_<항목>.joblib 을 하나씩 만든다.
    for index, target in enumerate(targets, start=1):
        if len(targets) > 1:
            print(f"\n########## 항목 {index}/{len(targets)}: {TARGETS[target]} ##########")
        train_target(df, target, args)


def train_target(df, target, args):

    features = feature_names(target, args.with_state)
    name = TARGETS[target]

    # 2) 특징과 정답 만들기
    step(2, f"특징과 정답({args.horizon}분 뒤 변화량) 만들기")
    groups = list(df.groupby("session"))
    parts = []

    for index, (session, group) in enumerate(groups, start=1):
        parts.append(add_features(group, target, args.horizon))
        progress(index, len(groups), session)

    data = pd.concat(parts, ignore_index=True)
    before = len(data)
    data = data.dropna(subset=features + ["target_change"])
    print(f"특징: {features}")
    print(f"처음 {max(LAGS)}분(과거가 없음)과 마지막 {args.horizon}분(정답이 없음)은 "
          f"뺐습니다: {before - len(data)}행")
    print(data[["session", f"{target}_now", "future", "target_change"]]
          .head(3).round(1).to_string())
    pause(args)

    # 3) 학습/시험 나누기
    step(3, "세션 단위로 학습 데이터와 시험 데이터 나누기")
    is_sim = data["session"].str.startswith("aug_sim")
    is_variant = data["session"].str.startswith("aug_real")
    is_real = ~data["session"].str.startswith("aug_")

    train = data[is_sim | (is_variant if args.with_variants else False)]
    test = data[is_real].copy()

    if train.empty or test.empty:
        print("학습 또는 시험 데이터가 없습니다. m701_augment.py 와 "
              "m701_state_clean.py --all 을 먼저 실행하세요.")
        return

    print(f"학습: 세션 {train['session'].nunique()}개, {len(train)}행")
    print(f"시험: 세션 {test['session'].nunique()}개, {len(test)}행")
    print("시험 세션:", ", ".join(test["session"].unique()))
    print(f"(시험 세션이 {max(LAGS) + args.horizon}분보다 짧으면 예측할 구간이 없어 빠집니다.)")
    pause(args)

    # 4) 학습
    step(4, "모델 세 가지 학습해서 비교하기")
    trained = {}

    for index, (key, (title, make)) in enumerate(MODELS.items(), start=1):
        progress(index - 1, len(MODELS), f"{title} 학습 중...")
        start = time.time()
        model = make()
        model.fit(train[features], train["target_change"])
        trained[key] = model
        progress(index, len(MODELS), f"{title} 완료 ({time.time() - start:.1f}초)")

    pause(args)

    # 5) 실제 데이터로 시험
    step(5, "실제로 측정한 데이터로 시험하기")
    now = test[f"{target}_now"]
    test["predict_persist"] = now

    # 큰 변화가 있던 구간(프린터 가동 등)과 평온한 구간을 나눠서도 본다.
    big = (test["future"] - now).abs() >= BIG_CHANGE[target]

    def row(title, predict):
        mae, rmse = scores(test["future"], predict)
        error = (test["future"] - predict).abs()
        return (title, mae, rmse, error[big].mean(), error[~big].mean())

    rows = [row("지속 예측 (변화 없음)", now)]

    for key, model in trained.items():
        predict = now + model.predict(test[features])
        test[f"predict_{key}"] = predict
        rows.append(row(MODELS[key][0], predict))

    result = pd.DataFrame(
        rows, columns=["모델", "MAE", "RMSE", "큰 변화 MAE", "평온 MAE"]
    ).set_index("모델")
    result["지속 예측보다"] = (1 - result["MAE"] / result["MAE"].iloc[0]) * 100
    print(f"{args.horizon}분 뒤 {name} 예측 오차 (작을수록 좋음)")
    print(result.round(2).to_string(
        formatters={"지속 예측보다": lambda v: f"{v:+.0f}%"}))
    print(f"\n큰 변화: {args.horizon}분 동안 {BIG_CHANGE[target]} 넘게 변한 구간 "
          f"({int(big.sum())}행) / 평온: 나머지 ({int((~big).sum())}행)")
    print("MAE: 평균적으로 몇만큼 틀렸나 / RMSE: 크게 틀린 것을 더 무겁게 본 오차")
    print("지속 예측: '10분 뒤에도 지금과 같다'고 보는 가장 단순한 방법."
          " 이것보다 좋아야 모델을 쓸 의미가 있다.")

    chosen = trained[args.model]

    if hasattr(chosen, "feature_importances_"):
        print(f"\n특징 중요도 ({MODELS[args.model][0]}):")
        importance = pd.Series(chosen.feature_importances_, index=features)
        print(importance.sort_values(ascending=False).head(8).round(3).to_string())

    pause(args)

    # 6) 저장
    step(6, "그래프와 모델 저장하기")
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    test["predict"] = test[f"predict_{args.model}"]
    plot_test(test, target, args.horizon, MODELS[args.model][0],
              os.path.join(RESULT_DIR, f"forecast_{target}_test.png"), args)

    model_file = os.path.join(MODEL_DIR, f"forecast_{target}.joblib")
    joblib.dump({
        "model": chosen,
        "features": features,
        "target": target,
        "horizon_minutes": args.horizon,
        "inputs": INPUTS,
        "lags_minutes": LAGS,
        "baseline_minutes": BASELINE_MINUTES,
        "with_state": args.with_state,
    }, model_file)
    print(f"저장: {model_file} ({MODELS[args.model][0]})")


if __name__ == "__main__":
    main()
