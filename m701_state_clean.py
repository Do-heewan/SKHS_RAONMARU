
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. 기본 설정
# =========================

LABELED_DIR = os.path.join("data", "labeled")
RESULT_DIR = "results"

SENSOR_COLUMNS = {
    "eco2_raw": "eco2",
    "ech2o_raw": "ech2o",
    "tvoc_raw": "tvoc",
    "pm25_raw": "pm25",
    "pm10_raw": "pm10"
}

STATES = ["idle", "printing"]
STATE_COLORS = {"idle": "C0", "printing": "C3"}

PLOT_COLUMNS = [
    ("eCO2", "eco2"),
    ("TVOC", "tvoc"),
    ("PM2.5", "pm25")
]

MAX_NORMAL_GAP = 5  # 초


# =========================
# 2. 공통 함수
# =========================

def find_inputs(args):

    if args.input:
        return [args.input]

    files = sorted(
        glob.glob(os.path.join(LABELED_DIR, "state_*.csv"))
    )

    if args.all:
        return files

    return files[-1:]


def load(path):

    df = pd.read_csv(path)
    name = os.path.splitext(os.path.basename(path))[0]
    df["session"] = name.replace("state_", "")

    return df


def step(number, title):
    print(f"\n===== [{number}/6] {title} =====")


def pause(args):
    if not args.no_pause:
        input("\n다음 단계로 가려면 Enter를 누르세요...")


def show(fig, path, args):

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"저장: {path}")

    if not args.no_pause and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


# "19 00" -> 25.0 (첫 바이트: 정수, 둘째 바이트: 소수 첫째 자리)
def decode_hex(text):
    whole, tenth = text.split()
    return int(whole, 16) + int(tenth, 16) / 10


# =========================
# 3. 시간 단위 정리 (세션별)
# =========================

def resample_session(group, sensors, args):

    group = group.set_index("time")

    clean = group[sensors].resample(f"{args.resample}s").mean()
    smooth = clean.rolling(f"{args.window}s", min_periods=1).mean()
    smooth.columns = [f"{c}_smooth" for c in sensors]

    # 구간 안의 상태가 하나로 같을 때만 그 상태를 붙인다.
    state = group["state"].resample(f"{args.resample}s").agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else None
    )

    result = pd.concat([clean, smooth], axis=1)
    result.insert(0, "state", state)

    has_data = clean[sensors[0]].notna()
    mixed = int((has_data & result["state"].isna()).sum())
    empty = int((~has_data).sum())

    result = result[has_data & result["state"].notna()]

    return result, mixed, empty


# =========================
# 4. 그래프
# =========================

def plot_by_state(result, path, args):

    states = [s for s in STATES if s in set(result["state"])]

    fig, axes = plt.subplots(1, len(PLOT_COLUMNS), figsize=(11, 4))

    for ax, (title, column) in zip(axes, PLOT_COLUMNS):
        data = [result.loc[result["state"] == s, column] for s in states]
        box = ax.boxplot(data, patch_artist=True)

        for patch, state in zip(box["boxes"], states):
            patch.set_facecolor(STATE_COLORS[state])
            patch.set_alpha(0.4)

        ax.set_xticks(range(1, len(states) + 1), states)
        ax.set_title(title)

    fig.suptitle("Sensor values by state")
    show(fig, path, args)


def plot_timeline(result, path, args):

    fig, axes = plt.subplots(
        len(PLOT_COLUMNS), 1, figsize=(11, 2.4 * len(PLOT_COLUMNS)),
        sharex=True
    )

    printing = result["state"] == "printing"

    for ax, (title, column) in zip(axes, PLOT_COLUMNS):
        ax.plot(result.index, result[column], color="C0", alpha=0.3)
        ax.plot(result.index, result[f"{column}_smooth"], color="C0")

        # 음영 때문에 세로축이 0부터 시작하지 않도록 범위를 고정한다.
        ylim = ax.get_ylim()
        ax.fill_between(
            result.index, 0, 1, where=printing,
            transform=ax.get_xaxis_transform(),
            color=STATE_COLORS["printing"], alpha=0.15, linewidth=0
        )
        ax.set_ylim(ylim)
        ax.set_ylabel(title)

    axes[0].set_title("Shaded = printing")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    show(fig, path, args)


# =========================
# 5. 전처리 단계
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="전처리할 라벨링 CSV 파일")
    parser.add_argument("--all", action="store_true",
                        help="data/labeled 의 모든 파일을 합쳐서 전처리")
    parser.add_argument("--resample", type=int, default=10,
                        help="평균을 낼 시간 간격(초)")
    parser.add_argument("--window", type=int, default=60,
                        help="이동평균 구간(초)")
    parser.add_argument("--no-pause", action="store_true",
                        help="멈추지 않고 끝까지 실행")
    args = parser.parse_args()

    inputs = find_inputs(args)

    if not inputs:
        print("라벨링 데이터가 없습니다. 먼저 python m701_state_logger.py 로 수집하세요.")
        return

    # 1) 불러오기
    step(1, "라벨링 데이터 불러오기")
    df = pd.concat([load(path) for path in inputs], ignore_index=True)

    for path in inputs:
        print(f"파일: {path}")

    print(f"크기: {len(df)}행")
    print("상태별 행 수:")
    print(df["state"].value_counts().to_string())
    pause(args)

    # 2) 시간 변환
    step(2, "글자로 된 시간을 진짜 시간으로 바꾸기")
    df["time"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    print(f"측정 기간: {df['time'].min()} ~ {df['time'].max()}")
    pause(args)

    # 3) 온습도 변환
    step(3, "16진수 온도와 습도를 숫자로 바꾸기")
    df["temperature"] = df["temperature_hex"].map(decode_hex)
    df["humidity"] = df["humidity_hex"].map(decode_hex)
    print(df[["temperature_hex", "temperature",
              "humidity_hex", "humidity"]].head(3))
    pause(args)

    # 4) 수신 간격 확인
    step(4, "데이터가 빠진 구간이 있는지 확인하기")
    for session, group in df.groupby("session"):
        gaps = group["time"].diff().dt.total_seconds()
        print(
            f"{session}: 보통 {gaps.median():.1f}초, "
            f"가장 긴 간격 {gaps.max():.1f}초, "
            f"{MAX_NORMAL_GAP}초 초과 {(gaps > MAX_NORMAL_GAP).sum()}번"
        )
    pause(args)

    # 5) 품질 점검
    step(5, "잘못된 값과 잘못된 라벨 제거하기")
    before = len(df)
    df = df.drop_duplicates(subset=["session", "time"])
    df = df[df["temperature"].between(-20, 60)
            & df["humidity"].between(0, 100)]
    df = df[df["state"].isin(STATES)]
    print(f"제거한 행(중복, 불가능한 온습도, 알 수 없는 라벨): {before - len(df)}개")

    df = df.rename(columns=SENSOR_COLUMNS)
    sensors = list(SENSOR_COLUMNS.values()) + ["temperature", "humidity"]

    # 여러 날의 데이터를 합칠 수 있도록 열은 지우지 않고 알려만 준다.
    constant = [c for c in sensors if df[c].nunique() <= 1]
    print(f"값이 변하지 않는 열: {constant if constant else '없음'} (학습 전에 뺄지 판단)")
    print("주의: 값이 튀는 구간은 실제 변화일 수 있으므로 지우지 않습니다.")
    pause(args)

    # 6) 시간 단위 정리
    step(6, f"{args.resample}초 평균과 {args.window}초 이동평균 만들기 (라벨 유지)")
    parts = []
    mixed = empty = 0

    for session, group in df.groupby("session"):
        part, part_mixed, part_empty = resample_session(group, sensors, args)
        part.insert(0, "session", session)
        parts.append(part)
        mixed += part_mixed
        empty += part_empty

    result = pd.concat(parts).round(2)
    print(f"행 수: {len(df)}개 -> {len(result)}개")
    print(f"평시와 가동이 섞여 뺀 구간: {mixed}개")
    print(f"센서가 끊겨 비어 있어 뺀 구간: {empty}개")
    print("상태별 구간 수:")
    print(result["state"].value_counts().to_string())

    # 저장
    os.makedirs(RESULT_DIR, exist_ok=True)
    name = "all" if len(inputs) > 1 else result["session"].iloc[0]

    output_file = os.path.join(LABELED_DIR, f"clean_state_{name}.csv")
    os.makedirs(LABELED_DIR, exist_ok=True)
    result.to_csv(output_file, index_label="time")
    print(f"\n저장: {output_file}")

    # 상태별 비교
    print("\n상태별 평균:")
    print(result.groupby("state")[[c for _, c in PLOT_COLUMNS]].mean().round(1))
    plot_by_state(
        result, os.path.join(RESULT_DIR, f"state_{name}_boxplot.png"), args
    )

    # 시간에 따른 변화 (파일 하나일 때만)
    if len(inputs) == 1:
        plot_timeline(
            result, os.path.join(RESULT_DIR, f"state_{name}_timeline.png"),
            args
        )


if __name__ == "__main__":
    main()

