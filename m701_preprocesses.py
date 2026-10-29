
import serial
from datetime import datetime

PORT = "/dev/ttyAMA0"
BAUD = 9600
FRAME_LENGTH = 17
HEADER = b"\x3c\x02"


def parse_frame(data):
    # 1. 프레임 길이 검사
    if len(data) != FRAME_LENGTH:
        return None

    # 2. 시작 바이트 검사
    if data[:2] != HEADER:
        return None

    # 3. 체크섬 검사
    # 앞의 16바이트를 합산한 뒤 하위 8비트 비교
    checksum = sum(data[:16]) & 0xFF

    if checksum != data[16]:
        return None

    # 4. 센서 데이터 변환 (Big-endian)
    def read_uint16(start):
        return int.from_bytes(
            data[start:start + 2],
            byteorder="big"
        )

    return {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "eco2_raw": read_uint16(2),
        "ech2o_raw": read_uint16(4),
        "tvoc_raw": read_uint16(6),
        "pm25_raw": read_uint16(8),
        "pm10_raw": read_uint16(10),
        "temperature_bytes": data[12:14].hex(" "),
        "humidity_bytes": data[14:16].hex(" ")
    }


def main():
    print("M701B Data Preprocessing")
    print("Ctrl+C to stop\n")

    buffer = bytearray()

    with serial.Serial(
        port=PORT,
        baudrate=BAUD,
        bytesize=8,
        parity=serial.PARITY_NONE,
        stopbits=1,
        timeout=1
    ) as ser:

        try:
            while True:
                chunk = ser.read(64)

                if not chunk:
                    continue

                buffer.extend(chunk)

                while True:
                    pos = buffer.find(HEADER)

                    if pos == -1:
                        # 다음 수신에서 헤더가 완성될 수 있음
                        if buffer[-1:] == b"\x3c":
                            buffer[:] = b"\x3c"
                        else:
                            buffer.clear()
                        break

                    if pos > 0:
                        del buffer[:pos]

                    if len(buffer) < FRAME_LENGTH:
                        break

                    frame = bytes(buffer[:FRAME_LENGTH])
                    result = parse_frame(frame)

                    if result is None:
                        print("Invalid frame")
                        del buffer[0]
                        continue

                    del buffer[:FRAME_LENGTH]

                    print("-------------------------")
                    for key, value in result.items():
                        print(f"{key}: {value}")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
