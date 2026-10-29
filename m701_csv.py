
import serial
import csv
import os
from datetime import datetime

# =========================
# 1. 기본 설정
# =========================

PORT = "/dev/ttyAMA0"
BAUD = 9600
FRAME_SIZE = 17
HEADER = b"\x3c\x02"

CSV_FILE = "m701_raw_data.csv"

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
# 3. CSV 파일 준비
# =========================

def prepare_csv():

    needs_header = (
        not os.path.exists(CSV_FILE)
        or os.path.getsize(CSV_FILE) == 0
    )

    file = open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    )

    writer = csv.DictWriter(
        file,
        fieldnames=COLUMNS
    )

    if needs_header:
        writer.writeheader()
        file.flush()

    return file, writer


# =========================
# 4. UART 수신 및 저장
# =========================

def main():

    print("M701B CSV Data Logger")
    print(f"PORT: {PORT}")
    print(f"FILE: {CSV_FILE}")
    print("Ctrl+C to stop\n")

    buffer = bytearray()
    count = 0

    file, writer = prepare_csv()

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

                    frame = bytes(buffer[:FRAME_SIZE])
                    result = parse_frame(frame)

                    if result is None:
                        print("Invalid frame")
                        del buffer[0]
                        continue

                    # 정상 프레임 처리
                    del buffer[:FRAME_SIZE]

                    writer.writerow(result)
                    file.flush()

                    count += 1

                    print(
                        f"[{count}] "
                        f"{result['timestamp']} | "
                        f"PM2.5={result['pm25_raw']} | "
                        f"PM10={result['pm10_raw']} | "
                        f"TVOC={result['tvoc_raw']}"
                    )

    except KeyboardInterrupt:
        print("\nData logging stopped.")

    finally:
        file.close()
        print(f"Total saved frames: {count}")


if __name__ == "__main__":
    main()

