
# 실행: python m701_live_classify.py
# 센서 값을 수집해 CSV로 저장하면서, 학습한 결정 트리로
# 3D 프린터가 가동 중인지 평시인지 10초마다 판단해 보여 준다.
#
# 먼저 python m701_train_classifier.py 로 models/printer_tree.joblib 을 만들어야 한다.
# 센서가 없을 때는 저장된 파일로 연습할 수 있다:
#   python m701_live_classify.py --replay data/labeled/state_20261117_084512_grapepi.csv

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import joblib
import pandas as pd

# =========================
# 1. 기본 설정
# =========================

PORT = "/dev/ttyAMA0"
BAUD = 9600
FRAME_SIZE = 17
HEADER = b"\x3c\x02"

DATA_DIR = "data"
MODEL_FILE = os.path.join("models", "printer_tree.joblib")

COLUMNS = [
    "timestamp",
    "raw_hex",
    "eco2_raw",
    "ech2o_raw",
    "tvoc_raw",
    "pm25_raw",
    "pm10_raw",
    "temperature_hex",
    "humidity_hex"
]

# 학습 데이터를 만들 때(m701_state_clean.py)와 똑같이 정리해야 모델이 제대로 판단한다.
RESAMPLE = "10s"     # 10초 평균
WINDOW = "60s"       # 60초 이동평균
HISTORY_ROWS = 1000  # 판단에 쓸 최근 데이터 개수 (2초 간격이면 약 33분)


# =========================
# 2. 17바이트 데이터 해석
# =========================

def parse_frame(data):

    if len(data) != FRAME_SIZE:
        return None

    # 시작 바이트 확인
    if data[:2] != HEADER:
        return None

    # 체크섬 확인
    checksum = sum(data[:16]) & 0xFF

    if checksum != data[16]:
        return None

    # 2바이트 정수 변환
    def uint16(start):
        return int.from_bytes(
            data[start:start + 2],
            byteorder="big"
        )

    return {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        ),
        "raw_hex": data.hex(" "),
        "eco2_raw": uint16(2),
        "ech2o_raw": uint16(4),
        "tvoc_raw": uint16(6),
        "pm25_raw": uint16(8),
        "pm10_raw": uint16(10),
        "temperature_hex": data[12:14].hex(" "),
        "humidity_hex": data[14:16].hex(" ")
    }


# "19 00" -> 25.0 (첫 바이트: 정수, 둘째 바이트: 소수 첫째 자리)
def decode_hex(text):
    whole, tenth = text.split()
    return int(whole, 16) + int(tenth, 16) / 10


# =========================
# 3. 특징 만들기와 판단
# =========================

def predict_states(history, bundle, confirm):

    frame = pd.DataFrame(history).set_index("time")

    # 10초 평균. 아직 10초가 다 차지 않은 마지막 구간은 뺀다.
    clean = frame[["eco2", "tvoc", "humidity"]].resample(RESAMPLE).mean()
    clean = clean[clean.index + pd.Timedelta(RESAMPLE) <= frame.index[-1]]
    smooth = clean.rolling(WINDOW, min_periods=1).mean()
    smooth = smooth[clean["eco2"].notna()]

    if smooth.empty:
        return smooth

    # 학습할 때와 같은 특징 (m701_train_classifier.py 참고)
    change = f"{bundle['change_seconds']}s"
    baseline = f"{bundle['baseline_seconds']}s"
    features = pd.DataFrame(index=smooth.index)

    for name in ["eco2", "tvoc", "humidity"]:
        features[f"{name}_smooth"] = smooth[name]

    for name in ["eco2", "tvoc", "humidity"]:
        before = smooth[name].shift(freq=change).reindex(smooth.index)
        features[f"{name}_change"] = smooth[name] - before

    for name in ["eco2", "tvoc"]:
        features[f"{name}_rise"] = smooth[name] - smooth[name].rolling(baseline).min()

    # 5분 변화량을 잴 수 없는 처음 5분은 판단하지 않는다.
    features = features.dropna()

    if features.empty:
        return features

    model = bundle["model"]
    printing = list(model.classes_).index(1)
    features["probability"] = model.predict_proba(features[bundle["features"]])[:, printing]

    # 한 구간만 튀어도 상태가 바뀌지 않도록 최근 몇 구간 중 다수결로 정한다.
    vote = (features["probability"] >= 0.5).astype(float)
    features["predict"] = (vote.rolling(confirm, min_periods=1).mean() > 0.5).astype(int)

    return features


# =========================
# 4. 판단 결과 처리 (화면 출력 + 파일 저장)
# =========================

class Judge:

    def __init__(self, bundle, confirm, predict_file):
        self.bundle = bundle
        self.confirm = confirm
        self.history = deque(maxlen=HISTORY_ROWS)
        self.last_bin = None
        self.last_reported = None
        self.state = None
        self.predict_file = predict_file
        self.file = None
        self.writer = None

    def add(self, result, when):

        self.history.append({
            "time": when,
            "eco2": result["eco2_raw"],
            "tvoc": result["tvoc_raw"],
            "humidity": decode_hex(result["humidity_hex"])
        })

        # 10초 구간이 새로 시작될 때만 판단한다.
        current_bin = pd.Timestamp(when).floor(RESAMPLE)

        if current_bin == self.last_bin:
            return

        self.last_bin = current_bin
        features = predict_states(list(self.history), self.bundle, self.confirm)

        if features.empty:
            minutes = (when - self.history[0]["time"]).total_seconds() / 60
            print(f"[{when:%H:%M:%S}] 판단 준비 중... "
                  f"(5분 이상 데이터가 필요합니다, 지금 {minutes:.1f}분)")
            return

        latest = features.index[-1]

        if latest == self.last_reported:
            return

        self.last_reported = latest
        row = features.iloc[-1]
        state = "printing" if row["predict"] == 1 else "idle"
        text = "가동 중" if state == "printing" else "평시"
        mark = "  <== 상태 바뀜" if self.state is not None and state != self.state else ""
        self.state = state

        print(
            f"[{latest:%H:%M:%S}] {text:<4} (가동 확률 {row['probability'] * 100:3.0f}%) | "
            f"eCO2 {row['eco2_smooth']:5.0f} ({row['eco2_change']:+5.0f}/5분) | "
            f"TVOC {row['tvoc_smooth']:4.0f} ({row['tvoc_change']:+4.0f}/5분){mark}"
        )

        self.save(latest, state, row)

    def save(self, latest, state, row):

        # 나중에 실제 프린터 기록과 비교해 검증할 수 있도록 판단 결과도 저장한다.
        if self.file is None:
            os.makedirs(DATA_DIR, exist_ok=True)
            self.file = open(self.predict_file, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(["time", "state", "label", "probability"]
                                 + self.bundle["features"])

        self.writer.writerow(
            [f"{latest:%Y-%m-%d %H:%M:%S}", state, int(state == "printing"),
             round(row["probability"], 3)]
            + [round(row[name], 2) for name in self.bundle["features"]]
        )
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            print(f"판단 결과 저장: {self.predict_file}")


# =========================
# 5. 센서에서 읽기 / 저장된 파일로 연습하기
# =========================

def run_sensor(judge, session):

    # --replay 연습은 pyserial 없이도 되도록 센서를 쓸 때만 불러온다.
    import serial

    csv_file = os.path.join(DATA_DIR, f"session_{session}.csv")
    file = None
    writer = None
    buffer = bytearray()
    count = 0

    print(f"PORT: {PORT}")
    print(f"FILE: {csv_file}")
    print("Ctrl+C to stop\n")

    try:
        with serial.Serial(
            port=PORT,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        ) as ser:

            while True:

                chunk = ser.read(64)

                if not chunk:
                    continue

                buffer.extend(chunk)

                while True:

                    # 프레임 시작 위치 탐색
                    pos = buffer.find(HEADER)

                    if pos == -1:
                        if buffer[-1:] == b"\x3c":
                            buffer[:] = b"\x3c"
                        else:
                            buffer.clear()
                        break

                    # 헤더 이전 불필요한 바이트 제거
                    if pos > 0:
                        del buffer[:pos]

                    # 프레임이 완성될 때까지 대기
                    if len(buffer) < FRAME_SIZE:
                        break

                    result = parse_frame(bytes(buffer[:FRAME_SIZE]))

                    if result is None:
                        del buffer[0]
                        continue

                    del buffer[:FRAME_SIZE]

                    # 첫 데이터가 들어올 때 파일을 만든다. (빈 파일 방지)
                    if file is None:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        file = open(csv_file, "w", newline="", encoding="utf-8")
                        writer = csv.DictWriter(file, fieldnames=COLUMNS)
                        writer.writeheader()

                    writer.writerow(result)
                    file.flush()
                    count += 1

                    judge.add(result, datetime.now())

    except KeyboardInterrupt:
        print("\n수집을 멈췄습니다.")

    except serial.SerialException as error:
        print(f"\n센서에 연결할 수 없습니다: {error}")
        print("배선과 포트를 확인하세요. 센서 없이 연습하려면 --replay 옵션을 쓰세요.")

    finally:
        if file is not None:
            file.close()
        print(f"저장한 데이터: {count}개")


def run_replay(judge, path, speed):

    df = pd.read_csv(path)
    times = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    print(f"연습용 파일: {path} ({len(df)}행, {speed}배속)" if speed > 0
          else f"연습용 파일: {path} ({len(df)}행, 최대 속도)")
    print("Ctrl+C to stop\n")

    try:
        for index, row in df.iterrows():
            if speed > 0 and index > 0:
                gap = (times[index] - times[index - 1]).total_seconds()
                time.sleep(max(gap, 0) / speed)
            judge.add(row, times[index].to_pydatetime())

    except KeyboardInterrupt:
        print("\n연습을 멈췄습니다.")


# =========================
# 6. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_FILE,
                        help="학습한 모델 파일")
    parser.add_argument("--confirm", type=int, default=3,
                        help="최근 몇 개 구간(10초)의 다수결로 상태를 정할지")
    parser.add_argument("--replay",
                        help="센서 대신 저장된 수집 파일(CSV)을 읽어 연습하기")
    parser.add_argument("--speed", type=float, default=0,
                        help="--replay 재생 배속 (0이면 기다리지 않고 바로)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"모델 파일이 없습니다: {args.model}")
        print("먼저 python m701_train_classifier.py 로 모델을 학습하세요.")
        return

    bundle = joblib.load(args.model)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("M701B 3D 프린터 상태 실시간 판단")
    print(f"모델: {args.model} (특징 {len(bundle['features'])}개)")

    if args.replay:
        name = os.path.splitext(os.path.basename(args.replay))[0]
        predict_file = os.path.join(DATA_DIR, f"predict_replay_{name}.csv")
        judge = Judge(bundle, args.confirm, predict_file)
        run_replay(judge, args.replay, args.speed)
    else:
        predict_file = os.path.join(DATA_DIR, f"predict_{session}.csv")
        judge = Judge(bundle, args.confirm, predict_file)
        run_sensor(judge, session)

    judge.close()


if __name__ == "__main__":
    main()
