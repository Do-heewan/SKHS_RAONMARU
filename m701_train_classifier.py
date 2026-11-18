
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# =========================
# 1. 기본 설정
# =========================
#
# 3D 프린터가 가동 중(printing)인지 평시(idle)인지 맞히는 결정 트리를 학습한다.
# 입력은 m701_state_clean.py --all 이 만든 data/labeled/clean_state_all.csv 이다.
#
# 학습: 시뮬레이션 데이터(aug_sim_*)
# 시험: 실제로 측정한 데이터(aug_ 로 시작하지 않는 세션)
#
# 행을 무작위로 섞어 나누면 바로 옆 10초끼리 학습/시험에 갈라져 들어가
# 점수가 부풀려진다. 그래서 세션(측정한 날, 센서) 단위로 나눈다.

INPUT_FILE = os.path.join("data", "labeled", "clean_state_all.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "printer_tree.joblib")
RESULT_DIR = "results"

CHANGE_SECONDS = 300     # 최근 5분 동안의 변화량
BASELINE_SECONDS = 1800  # 최근 30분의 최저값을 "평소 값"으로 본다

FEATURES = [
    "eco2_smooth",      # 지금 eCO2
    "tvoc_smooth",      # 지금 TVOC
    "humidity_smooth",  # 지금 습도
    "eco2_change",      # 5분 전보다 eCO2가 얼마나 올랐나
    "tvoc_change",      # 5분 전보다 TVOC가 얼마나 올랐나
    "humidity_change",  # 5분 전보다 습도가 얼마나 올랐나
    "eco2_rise",        # 평소(30분 최저값)보다 eCO2가 얼마나 높나
    "tvoc_rise",        # 평소(30분 최저값)보다 TVOC가 얼마나 높나
]


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
# 2. 특징 만들기 (세션별)
# =========================

def add_features(group):

    group = group.set_index("time").sort_index()

    for name in ["eco2", "tvoc", "humidity"]:
        now = group[f"{name}_smooth"]
        before = now.shift(freq=f"{CHANGE_SECONDS}s").reindex(now.index)
        group[f"{name}_change"] = now - before

    for name in ["eco2", "tvoc"]:
        now = group[f"{name}_smooth"]
        lowest = now.rolling(f"{BASELINE_SECONDS}s").min()
        group[f"{name}_rise"] = now - lowest

    return group.reset_index()


# =========================
# 3. 그래프
# =========================

def plot_test(test, path, args):

    sessions = list(test["session"].unique())
    fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2.6 * len(sessions)),
                             squeeze=False)

    for ax, session in zip(axes[:, 0], sessions):
        part = test[test["session"] == session]
        ax.plot(part["time"], part["eco2_smooth"], color="C0", lw=1, label="eCO2")
        top = part["eco2_smooth"].max()
        bottom = part["eco2_smooth"].min()

        ax.fill_between(part["time"], bottom, top, where=part["label"] == 1,
                        color="C3", alpha=0.25, label="정답: 가동")
        ax.fill_between(part["time"], bottom, bottom + (top - bottom) * 0.12,
                        where=part["predict"] == 1, color="C2", alpha=0.8,
                        label="예측: 가동")
        ax.set_title(session, fontsize=10)
        ax.set_ylabel("eCO2")

    axes[0, 0].legend(loc="upper right", fontsize=8)
    show(fig, path, args)


def plot_model(model, path, args):

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(model, feature_names=FEATURES, class_names=["idle", "printing"],
              filled=True, rounded=True, fontsize=7, ax=ax)
    show(fig, path, args)


# =========================
# 4. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE,
                        help="전처리된 라벨링 데이터 (m701_state_clean.py --all 결과)")
    parser.add_argument("--depth", type=int, default=4,
                        help="결정 트리의 최대 깊이")
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

    # 1) 불러오기
    step(1, "전처리된 데이터 불러오기")
    df = pd.read_csv(args.input, parse_dates=["time"])
    df["label"] = (df["state"] == "printing").astype(int)
    print(f"파일: {args.input}")
    print(f"크기: {len(df)}행, 세션 {df['session'].nunique()}개")
    pause(args)

    # 2) 특징 만들기
    step(2, "특징 만들기 (지금 값, 5분 변화량, 평소보다 높은 정도)")
    groups = list(df.groupby("session"))
    parts = []

    for index, (session, group) in enumerate(groups, start=1):
        parts.append(add_features(group))
        progress(index, len(groups), session)

    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.dropna(subset=FEATURES)
    print(f"특징: {FEATURES}")
    print(f"각 세션 처음 5분은 변화량을 잴 수 없어 뺐습니다: {before - len(df)}행")
    print(df[["session"] + FEATURES].head(3).round(1).to_string())
    pause(args)

    # 3) 학습/시험 나누기
    step(3, "세션 단위로 학습 데이터와 시험 데이터 나누기")
    is_sim = df["session"].str.startswith("aug_sim")
    is_variant = df["session"].str.startswith("aug_real")
    is_real = ~df["session"].str.startswith("aug_")

    train = df[is_sim | (is_variant if args.with_variants else False)]
    test = df[is_real].copy()

    if train.empty or test.empty:
        print("학습 또는 시험 데이터가 없습니다. m701_augment.py 와 "
              "m701_state_clean.py --all 을 먼저 실행하세요.")
        return

    for title, part in [("학습", train), ("시험", test)]:
        counts = part["label"].value_counts()
        print(f"{title}: 세션 {part['session'].nunique()}개, "
              f"평시 {counts.get(0, 0)}개 / 가동 {counts.get(1, 0)}개")
    print("시험 세션:", ", ".join(test["session"].unique()))
    pause(args)

    # 4) 학습
    step(4, f"결정 트리 학습 (최대 깊이 {args.depth})")
    # 결정 트리는 한 번에 학습이 끝나므로, 깊이를 1부터 하나씩 늘려 가며
    # 트리가 자랄수록 학습 데이터를 얼마나 더 잘 맞히는지 보여 준다.
    # 마지막(최대 깊이) 트리를 최종 모델로 쓴다.
    history = []

    for depth in range(1, args.depth + 1):
        model = DecisionTreeClassifier(max_depth=depth, random_state=0)
        model.fit(train[FEATURES], train["label"])
        score = accuracy_score(train["label"], model.predict(train[FEATURES]))
        history.append((depth, model.get_n_leaves(), score))
        progress(depth, args.depth, f"깊이 {depth}: 정확도 {score:.3f}")

    print("\n깊이  잎 수  학습 데이터 정확도")
    for depth, leaves, score in history:
        print(f"{depth:>4}  {leaves:>5}  {score:.3f}")
    print("\n트리 규칙:")
    print(export_text(model, feature_names=FEATURES, decimals=1))
    print("특징 중요도:")
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    print(importance.sort_values(ascending=False).round(3).to_string())
    pause(args)

    # 5) 실제 데이터로 시험
    step(5, "실제로 측정한 데이터로 시험하기")
    test["predict"] = model.predict(test[FEATURES])
    print(f"정확도: {accuracy_score(test['label'], test['predict']):.3f}\n")
    print(classification_report(test["label"], test["predict"],
                                target_names=["idle", "printing"], zero_division=0))
    matrix = confusion_matrix(test["label"], test["predict"], labels=[0, 1])
    print(pd.DataFrame(matrix, index=["정답 idle", "정답 printing"],
                       columns=["예측 idle", "예측 printing"]).to_string())
    print("\n주의: 실제 데이터의 가동 시간은 추정한 것이라 정답 자체가 정확하지 않을 수 있습니다.")
    pause(args)

    # 6) 저장
    step(6, "그래프와 모델 저장하기")
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_model(model, os.path.join(RESULT_DIR, "classifier_tree.png"), args)
    plot_test(test, os.path.join(RESULT_DIR, "classifier_test.png"), args)
    joblib.dump({"model": model, "features": FEATURES,
                 "change_seconds": CHANGE_SECONDS,
                 "baseline_seconds": BASELINE_SECONDS}, MODEL_FILE)
    print(f"저장: {MODEL_FILE}")


if __name__ == "__main__":
    main()
