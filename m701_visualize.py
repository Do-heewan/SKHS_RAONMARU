
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


# =========================
# 2. 공통 함수
# =========================

def latest_clean():

    files = glob.glob(os.path.join(DATA_DIR, "clean_*.csv"))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


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

def plot_timeseries(df, path, args):

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

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time")
    show(fig, path, args)


# =========================
# 4. 산점도
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
# 5. 실행
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

    os.makedirs(RESULT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_file))[0]
    name = name.replace("clean_", "")

    print(f"파일: {input_file}")

    print("\n===== [1/2] 시간에 따른 공기질 변화 =====")
    print("옅은 선은 10초 평균, 진한 선은 이동평균입니다.")
    plot_timeseries(
        df, os.path.join(RESULT_DIR, f"{name}_timeseries.png"), args
    )
    pause(args)

    print("\n===== [2/2] eCO2와 TVOC의 관계 =====")
    plot_scatter(
        df, os.path.join(RESULT_DIR, f"{name}_scatter.png"), args
    )


if __name__ == "__main__":
    main()

