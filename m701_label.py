
import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. 기본 설정
# =========================

# 모든 팀이 같은 데이터를 보고 판단을 비교하도록 예시 데이터를 기본으로 쓴다.
DUMMY_FILE = "m701_raw_data.csv"
RESULT_DIR = "results"

PANELS = [
    ("eCO2", "eco2_raw"),
    ("TVOC", "tvoc_raw"),
    ("PM2.5", "pm25_raw")
]

EDGE_SECONDS = 10  # 변화량 계산에 쓰는 구간 처음/마지막 시간(초)


# =========================
# 2. 구간 나누기
# =========================

def make_blocks(df, seconds):

    offset = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["block"] = (offset // seconds).astype(int) + 1
    df["offset"] = offset - (df["block"] - 1) * seconds

    # 데이터가 절반도 안 되는 짧은 구간(보통 마지막)은 뺀다.
    counts = df["block"].value_counts()
    full = counts[counts >= counts.median() / 2].index

    return df[df["block"].isin(full)]


def summarize(df, seconds):

    rows = []

    for number, block in df.groupby("block"):

        first = block[block["offset"] < EDGE_SECONDS]
        last = block[block["offset"] >= seconds - EDGE_SECONDS]

        row = {
            "block": number,
            "start": block["time"].iloc[0],
            "end": block["time"].iloc[-1]
        }

        for name, column in PANELS:
            row[name] = block[column].mean()

        # 구간 마지막 10초 평균 - 처음 10초 평균
        for name, column in PANELS[:2]:
            row[f"{name} change"] = last[column].mean() - first[column].mean()

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# 3. 표 출력
# =========================

# 한글은 터미널에서 두 칸을 차지하므로 너비를 따로 계산한다.
def width(text):
    return sum(2 if ord(ch) > 0x1100 else 1 for ch in text)


def cell(text, size):
    text = str(text)
    return " " * (size - width(text)) + text


def number(value, signed=False, digits=0):
    if pd.isna(value):
        return "-"
    sign = "+" if signed else ""
    value = round(value, digits) + 0.0  # -0 이 표시되지 않게 한다.
    return f"{value:{sign}.{digits}f}"


def print_table(table):

    sizes = [4, 19, 7, 7, 7, 7, 7]
    headers = ["구간", "시간", "eCO2", "변화", "TVOC", "변화", "PM2.5"]

    print("".join(cell(h, s) for h, s in zip(headers, sizes)))
    print("-" * sum(sizes))

    for _, row in table.iterrows():
        values = [
            row["block"],
            f"{row['start']:%H:%M:%S}~{row['end']:%H:%M:%S}",
            number(row["eCO2"]),
            number(row["eCO2 change"], signed=True),
            number(row["TVOC"]),
            number(row["TVOC change"], signed=True),
            number(row["PM2.5"], digits=1)
        ]
        print("".join(cell(v, s) for v, s in zip(values, sizes)))


# =========================
# 4. 구간 번호 그래프
# =========================

def plot_blocks(df, table, seconds, path, args):

    start = df["time"].iloc[0]
    step = pd.Timedelta(seconds=seconds)
    left = mdates.date2num(start)
    right = mdates.date2num(start + step * len(table))

    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(11, 2.4 * len(PANELS)), sharex=True
    )

    for ax, (name, column) in zip(axes, PANELS):

        for index in range(len(table)):
            if index % 2 == 0:
                ax.axvspan(
                    mdates.date2num(start + step * index),
                    mdates.date2num(start + step * (index + 1)),
                    color="0.92"
                )

        ax.plot(df["time"], df[column], color="C0", linewidth=1)
        ax.set_ylabel(name)

    for index, row in table.iterrows():
        center = start + step * index + step / 2
        axes[0].text(
            mdates.date2num(center), 1.03, str(row["block"]),
            transform=axes[0].get_xaxis_transform(),
            ha="center", fontsize=11, fontweight="bold"
        )

    axes[0].set_xlim(left, right)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time")
    fig.suptitle("Judge each block: Good / Caution / Ventilate")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"저장: {path}")

    if not args.no_pause and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


# =========================
# 5. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DUMMY_FILE,
                        help="판단할 원본 CSV 파일")
    parser.add_argument("--block", type=int, default=60,
                        help="구간 길이(초)")
    parser.add_argument("--no-pause", action="store_true",
                        help="그래프 창을 띄우지 않음")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["time"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    df = make_blocks(df, args.block)
    table = summarize(df, args.block)

    print("===== 데이터를 읽고 판단하기 =====")
    print(f"파일: {args.input}")
    print(f"구간: {args.block}초씩 {len(table)}개")
    print("구간마다 좋음 / 주의 / 환기 필요 중 하나를 골라 활동지에 적어 보세요.")
    print("판단한 이유도 함께 적어 보세요.\n")
    print("변화 = 구간 마지막 10초 평균 - 처음 10초 평균 (+는 증가, -는 감소)\n")

    print_table(table)
    print()

    os.makedirs(RESULT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    name = name.replace("session_", "")

    table.round(1).to_csv(
        os.path.join(RESULT_DIR, f"{name}_blocks.csv"), index=False
    )
    plot_blocks(
        df, table, args.block,
        os.path.join(RESULT_DIR, f"{name}_blocks.png"), args
    )


if __name__ == "__main__":
    main()

