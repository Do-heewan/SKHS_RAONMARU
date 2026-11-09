
import argparse
import glob
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. 기본 설정
# =========================

DATA_DIR = "data"
RESULT_DIR = "results"

PANELS = [
    ("eCO2", ["eco2"]),
    ("TVOC", ["tvoc"]),
    ("PM", ["pm25", "pm10"]),
    ("Temperature", ["temperature"]),
    ("Humidity", ["humidity"])
]

BEFORE_SECONDS = 60
AFTER_SECONDS = 180


# =========================
# 2. 공통 함수
# =========================

def latest_clean():

    files = glob.glob(os.path.join(DATA_DIR, "clean_*.csv"))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def load_events(input_file):

    name = os.path.basename(input_file).replace("clean_", "events_")
    path = os.path.join(os.path.dirname(input_file), name)

    if not os.path.exists(path):
        return None

    events = pd.read_csv(path)
    events["time"] = pd.to_datetime(
        events["timestamp"], format="ISO8601"
    ).dt.tz_localize(None)

    return events


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


# =========================
# 3. 항목별 시계열 그래프
# =========================

def plot_timeseries(df, events, path, args):

    panels = [
        (title, [c for c in columns if c in df.columns])
        for title, columns in PANELS
    ]
    panels = [(title, columns) for title, columns in panels if columns]

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 2.2 * len(panels)), sharex=True
    )

    for ax, (title, columns) in zip(axes, panels):

        for index, column in enumerate(columns):
            color = f"C{index}"
            ax.plot(df.index, df[column], color=color, alpha=0.3)
            ax.plot(df.index, df[f"{column}_smooth"], color=color,
                    label=column)

        ax.set_ylabel(title)

        if len(columns) > 1:
            ax.legend(loc="upper right")

        if events is not None:
            for time in events["time"]:
                ax.axvline(time, color="black", linestyle="--", linewidth=1)

    if events is not None:
        for _, event in events.iterrows():
            axes[0].annotate(
                event["event"],
                xy=(mdates.date2num(event["time"]), 1),
                xycoords=axes[0].get_xaxis_transform(),
                xytext=(0, 4),
                textcoords="offset points",
                rotation=30
            )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time")
    show(fig, path, args)


# =========================
# 4. 사건 전후 비교
# =========================

def compare_events(df, events):

    if events is None:
        print("기록된 사건이 없습니다. (예시 데이터에는 사건 기록이 없습니다)")
        return

    for _, event in events.iterrows():

        start = event["time"]
        before = df[(df.index >= start - pd.Timedelta(seconds=BEFORE_SECONDS))
                    & (df.index < start)]
        after = df[(df.index >= start)
                   & (df.index <= start + pd.Timedelta(seconds=AFTER_SECONDS))]

        print(f"\n{event['event']} ({start:%H:%M:%S})")

        for column in ["eco2", "tvoc"]:
            print(
                f"  {column}: 직전 평균 {before[column].mean():.1f}"
                f" -> 이후 최고 {after[column].max():.1f}"
            )


# =========================
# 5. 산점도
# =========================

def plot_scatter(df, path, args):

    r = df["eco2"].corr(df["tvoc"])
    print(f"eCO2와 TVOC의 상관계수: {r:.2f} (1에 가까울수록 함께 움직임)")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["eco2"], df["tvoc"], alpha=0.5)
    ax.set_title(f"eCO2 vs TVOC (r = {r:.2f})")
    ax.set_xlabel("eCO2")
    ax.set_ylabel("TVOC")
    show(fig, path, args)


# =========================
# 6. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="전처리된 CSV 파일")
    parser.add_argument("--no-pause", action="store_true",
                        help="멈추지 않고 끝까지 실행")
    args = parser.parse_args()

    input_file = args.input or latest_clean()

    if input_file is None:
        print("전처리된 파일이 없습니다. 먼저 python m701_clean.py 를 실행하세요.")
        return

    df = pd.read_csv(input_file, parse_dates=["time"], index_col="time")
    events = load_events(input_file)

    os.makedirs(RESULT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_file))[0]
    name = name.replace("clean_", "")

    print(f"파일: {input_file}")

    print("\n===== [1/3] 시간에 따른 공기질 변화 =====")
    print("옅은 선은 10초 평균, 진한 선은 이동평균, 점선은 기록한 사건입니다.")
    plot_timeseries(
        df, events,
        os.path.join(RESULT_DIR, f"{name}_timeseries.png"), args
    )
    pause(args)

    print("\n===== [2/3] 사건 전후 비교 =====")
    compare_events(df, events)
    pause(args)

    print("\n===== [3/3] eCO2와 TVOC의 관계 =====")
    plot_scatter(
        df, os.path.join(RESULT_DIR, f"{name}_scatter.png"), args
    )


if __name__ == "__main__":
    main()

