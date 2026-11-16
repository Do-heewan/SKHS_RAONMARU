
import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime

import serial

# =========================
# 1. 기본 설정
# =========================

PORT = "/dev/ttyAMA0"
BAUD = 9600
FRAME_SIZE = 17
HEADER = b"\x3c\x02"

DATA_DIR = os.path.join("data", "labeled")
SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = os.path.join(DATA_DIR, f"state_{SESSION}.csv")

# 숫자 + Enter 로 상태를 바꾼다. 파일에는 영어 라벨로 저장한다.
STATES = {
    "0": "idle",
    "1": "printing"
}

STATE_NAMES = {
    "idle": "평시",
    "printing": "3D프린터 가동"
}

FIELDS = [
    "timestamp",
    "state",
    "raw_hex",
    "eco2_raw",
    "ech2o_raw",
    "tvoc_raw",
    "pm25_raw",
    "pm10_raw",
    "temperature_hex",
    "humidity_hex"
]

RETRY_SECONDS = 5  # 센서 연결이 끊겼을 때 다시 연결하기까지 기다리는 시간

current_state = "idle"


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


# =========================
# 3. 상태 바꾸기 (키보드 입력)
# =========================

def watch_keyboard():

    global current_state

    while True:

        try:
            key = input().strip()
        except (EOFError, OSError):
            # 백그라운드 실행처럼 키보드가 없으면 처음 상태를 유지한다.
            return

        if key not in STATES:
            print("0(평시) 또는 1(3D프린터 가동)을 입력하고 Enter를 누르세요.")
            continue

        current_state = STATES[key]
        print(f"\n*** 상태 변경: {STATE_NAMES[current_state]} ***\n")


# =========================
# 4. 센서 프레임 읽기 (끊기면 다시 연결)
# =========================

def read_frames():

    while True:

        try:
            with serial.Serial(
                port=PORT,
                baudrate=BAUD,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            ) as ser:

                buffer = bytearray()

                while True:

                    chunk = ser.read(64)

                    if not chunk:
                        continue

                    buffer.extend(chunk)

                    while True:

                        pos = buffer.find(HEADER)

                        if pos == -1:
                            if buffer[-1:] == b"\x3c":
                                buffer[:] = b"\x3c"
                            else:
                                buffer.clear()
                            break

                        if pos > 0:
                            del buffer[:pos]

                        if len(buffer) < FRAME_SIZE:
                            break

                        result = parse_frame(bytes(buffer[:FRAME_SIZE]))

                        if result is None:
                            del buffer[0]
                            continue

                        del buffer[:FRAME_SIZE]
                        yield result

        except serial.SerialException as error:
            print(f"센서 연결 끊김: {error}")
            print(f"{RETRY_SECONDS}초 후 다시 연결합니다. (종료: Ctrl+C)")
            time.sleep(RETRY_SECONDS)


# =========================
# 5. 실행
# =========================

def main():

    global current_state

    # 백그라운드에서 로그 파일로 보낼 때도 출력이 바로 기록되게 한다.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["idle", "printing"],
                        default="idle", help="시작할 때의 상태")
    args = parser.parse_args()

    current_state = args.state

    print("M701B 라벨링 데이터 수집")
    print(f"FILE: {CSV_FILE}")
    print(f"시작 상태: {STATE_NAMES[current_state]}")
    print("상태 바꾸기: 0 = 평시 / 1 = 3D프린터 가동 (숫자 + Enter)")
    print("Ctrl+C to stop\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    file = open(CSV_FILE, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()

    counts = {"idle": 0, "printing": 0}

    threading.Thread(target=watch_keyboard, daemon=True).start()

    try:
        for result in read_frames():

            row = {**result, "state": current_state}
            writer.writerow(row)
            file.flush()

            counts[current_state] += 1

            print(
                f"[{sum(counts.values())}] "
                f"{result['timestamp']} | "
                f"{STATE_NAMES[current_state]} | "
                f"eCO2={result['eco2_raw']} | "
                f"TVOC={result['tvoc_raw']} | "
                f"PM2.5={result['pm25_raw']}"
            )

    except KeyboardInterrupt:
        print("\n수집을 멈췄습니다.")

    finally:
        file.close()

        for state, count in counts.items():
            print(f"{STATE_NAMES[state]}: {count}개")

        # 데이터가 없는 빈 파일은 남기지 않는다.
        if sum(counts.values()) == 0:
            os.remove(CSV_FILE)


if __name__ == "__main__":
    main()

