
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. 기본 설정
# =========================

DATA_DIR = "data"
RESULT_DIR = "results"
DUMMY_FILE = "m701_raw_data.csv"

SENSOR_COLUMNS = {
    "eco2_raw": "eco2",
    "ech2o_raw": "ech2o",
    "tvoc_raw": "tvoc",
    "pm25_raw": "pm25",
    "pm10_raw": "pm10"
}

MAX_NORMAL_GAP = 5  # 초


# =========================
# 2. 공통 함수
# =========================

def latest_session():

    sessions = sorted(
        glob.glob(os.path.join(DATA_DIR, "session_*.csv"))
    )

    if sessions:
        return sessions[-1]

    print("수집한 데이터가 없어 예시 데이터를 사용합니다.")
    return DUMMY_FILE


def step(number, title):
    print(f"\n===== [{number}/6] {title} =====")


def pause(args):
    if not args.no_pause:
        input("\n다음 단계로 가려면 Enter를 누르세요...")


# "19 00" -> 25.0 (첫 바이트: 정수, 둘째 바이트: 소수 첫째 자리)
def decode_hex(text):
    whole, tenth = text.split()
    return int(whole, 16) + int(tenth, 16) / 10


# =========================
# 3. 전처리 단계
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="전처리할 CSV 파일")
    parser.add_argument("--resample", type=int, default=10,
                        help="평균을 낼 시간 간격(초)")
    parser.add_argument("--window", type=int, default=60,
                        help="이동평균 구간(초)")
    parser.add_argument("--no-pause", action="store_true",
                        help="멈추지 않고 끝까지 실행")
    args = parser.parse_args()

    input_file = args.input or latest_session()
    name = os.path.splitext(os.path.basename(input_file))[0]
    name = name.replace("session_", "")

    # 1) 불러오기
    step(1, "원본 데이터 불러오기")
    df = pd.read_csv(input_file)
    print(f"파일: {input_file}")
    print(f"크기: {len(df)}행, {len(df.columns)}열")
    print(df[["timestamp", "eco2_raw", "tvoc_raw", "pm25_raw"]].head())
    pause(args)

    # 2) 시간 변환
    step(2, "글자로 된 시간을 진짜 시간으로 바꾸기")
    print(f"변환 전: {df['timestamp'].iloc[0]} ({type(df['timestamp'].iloc[0]).__name__})")
    df["time"] = pd.to_datetime(
        df["timestamp"], format="ISO8601"
    ).dt.tz_localize(None)
    print(f"변환 후: {df['time'].iloc[0]} ({type(df['time'].iloc[0]).__name__})")
    print("이제 시간끼리 빼거나 10초 단위로 묶을 수 있습니다.")
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
    gaps = df["time"].diff().dt.total_seconds()
    print(f"보통 간격: {gaps.median():.1f}초")
    print(f"가장 긴 간격: {gaps.max():.1f}초")
    print(f"{MAX_NORMAL_GAP}초보다 긴 간격: {(gaps > MAX_NORMAL_GAP).sum()}번")
    pause(args)

    # 5) 품질 점검
    step(5, "잘못된 값과 쓸모없는 열 제거하기")
    before = len(df)
    df = df.drop_duplicates(subset="time")
    df = df[df["temperature"].between(-20, 60)
            & df["humidity"].between(0, 100)]
    print(f"제거한 행(중복, 불가능한 온습도): {before - len(df)}개")

    df = df.rename(columns=SENSOR_COLUMNS)
    sensors = list(SENSOR_COLUMNS.values()) + ["temperature", "humidity"]

    # eco2, tvoc는 시각화에 꼭 필요하므로 측정이 짧아 값이 같아도 남긴다.
    constant = [
        c for c in sensors
        if c not in ("eco2", "tvoc") and df[c].nunique() <= 1
    ]
    sensors = [c for c in sensors if c not in constant]
    print(f"값이 변하지 않아 제외한 열: {constant if constant else '없음'}")
    print("주의: 입김처럼 값이 튀는 구간은 실제 사건이므로 지우지 않습니다.")
    pause(args)

    # 6) 시간 단위 정리
    step(6, f"{args.resample}초 평균과 {args.window}초 이동평균 만들기")
    raw = df.set_index("time")[sensors]
    clean = raw.resample(f"{args.resample}s").mean()
    smooth = clean.rolling(f"{args.window}s", min_periods=1).mean()
    smooth.columns = [f"{c}_smooth" for c in sensors]
    result = pd.concat([clean, smooth], axis=1).round(2)
    print(f"행 수: {len(raw)}개 -> {len(result)}개")
    print(f"센서가 끊겨 비어 있는 구간: {clean[sensors[0]].isna().sum()}개")
    print(result[["tvoc", "tvoc_smooth"]].head())

    # 저장
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    output_file = os.path.join(DATA_DIR, f"clean_{name}.csv")
    result.to_csv(output_file)
    print(f"\n저장: {output_file}")

    # 전후 비교 그래프
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw.index, raw["tvoc"], color="lightgray", label="Raw")
    ax.plot(result.index, result["tvoc"], color="C0", alpha=0.6,
            label=f"{args.resample}s average")
    ax.plot(result.index, result["tvoc_smooth"], color="C3",
            label=f"{args.window}s moving average")
    ax.set_title("TVOC: before and after preprocessing")
    ax.set_xlabel("Time")
    ax.set_ylabel("TVOC")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    figure_file = os.path.join(RESULT_DIR, f"{name}_preprocess.png")
    fig.savefig(figure_file, dpi=150)
    print(f"저장: {figure_file}")

    if not args.no_pause and plt.get_backend().lower() != "agg":
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()

