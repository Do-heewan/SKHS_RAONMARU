
import argparse
import glob
import os

import pandas as pd

# =========================
# 1. 기본 설정
# =========================

DATA_DIR = "data"
LABELED_DIR = os.path.join("data", "labeled")
PRINT_LOG = os.path.join("data", "print_log.csv")

LOG_EXAMPLE = """start,end
2026-11-17 13:05,2026-11-17 15:40
2026-11-18 09:30,2026-11-18 11:00"""


# =========================
# 2. 프린터 가동 기록 읽기
# =========================

def load_print_log(path):

    if not os.path.exists(path):
        print(f"프린터 가동 기록 파일이 없습니다: {path}")
        print("아래처럼 시작, 끝 시간을 한 줄에 하나씩 적어 저장하세요.\n")
        print(LOG_EXAMPLE)
        return None

    log = pd.read_csv(path)
    log["start"] = pd.to_datetime(log["start"])
    log["end"] = pd.to_datetime(log["end"])

    wrong = log[log["end"] <= log["start"]]

    if not wrong.empty:
        print("끝 시간이 시작 시간보다 빠르거나 같은 줄이 있습니다. 고친 뒤 다시 실행하세요.")
        print(wrong.to_string())
        return None

    return log


# =========================
# 3. 세션 파일에 라벨 붙이기
# =========================

def label_session(path, log):

    df = pd.read_csv(path)
    time = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # 가동 기록의 시작~끝 사이에 있으면 가동 중(1), 아니면 평시(0)
    printing = pd.Series(False, index=df.index)

    for _, job in log.iterrows():
        printing |= time.between(job["start"], job["end"])

    df.insert(1, "state", printing.map({True: "printing", False: "idle"}))
    df.insert(2, "label", printing.astype(int))

    return df, time


# =========================
# 4. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+",
                        help="라벨을 붙일 수집 파일 (기본: data/session_*.csv 전부)")
    parser.add_argument("--log", default=PRINT_LOG,
                        help="프린터 가동 기록 CSV 파일")
    args = parser.parse_args()

    log = load_print_log(args.log)

    if log is None:
        return

    inputs = args.input or sorted(
        glob.glob(os.path.join(DATA_DIR, "session_*.csv"))
    )

    if not inputs:
        print("수집한 데이터가 없습니다. 먼저 python m701_csv.py 로 수집하세요.")
        return

    print(f"프린터 가동 기록: {args.log} ({len(log)}건)\n")

    os.makedirs(LABELED_DIR, exist_ok=True)
    job_rows = [0] * len(log)

    for path in inputs:

        df, time = label_session(path, log)

        name = os.path.splitext(os.path.basename(path))[0]
        name = name.replace("session_", "")
        output_file = os.path.join(LABELED_DIR, f"state_{name}.csv")
        df.to_csv(output_file, index=False)

        counts = df["label"].value_counts()
        print(
            f"{path}\n"
            f"  기간: {time.min():%Y-%m-%d %H:%M} ~ {time.max():%H:%M}\n"
            f"  평시(0): {counts.get(0, 0)}개 / 가동(1): {counts.get(1, 0)}개\n"
            f"  저장: {output_file}"
        )

        for index, job in log.iterrows():
            job_rows[index] += int(time.between(job["start"], job["end"]).sum())

    # 수집 데이터와 겹치지 않는 가동 기록은 알려 준다.
    missing = [
        job for (_, job), rows in zip(log.iterrows(), job_rows) if rows == 0
    ]

    if missing:
        print("\n주의: 아래 가동 시간에는 수집한 데이터가 없습니다. 로거가 꺼져 있었는지 확인하세요.")
        for job in missing:
            print(f"  {job['start']:%Y-%m-%d %H:%M} ~ {job['end']:%H:%M}")

    print("\n다음 단계: python m701_state_clean.py --all")


if __name__ == "__main__":
    main()

