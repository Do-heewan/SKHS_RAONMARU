
# 실행: streamlit run m701_dashboard.py
# 센서 값을 수집해 CSV로 저장하면서, 같은 값을 웹 대시보드에 바로 보여 준다.
# 같은 Wi-Fi의 다른 기기에서는 http://<라즈베리파이 IP>:8501 로 접속한다.
# 페이지 두 개(그래프 / 요약)가 같은 수집기를 함께 쓴다. 왼쪽 메뉴에서 고른다.

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import serial
import streamlit as st

# =========================
# 1. 기본 설정
# =========================

PORT = "/dev/ttyAMA0"
BAUD = 9600
FRAME_SIZE = 17
HEADER = b"\x3c\x02"

DATA_DIR = "data"
RETRY_SECONDS = 5
MAX_ROWS = 1800          # 화면용으로 기억할 개수 (2초 간격이면 약 1시간)
REFRESH = "2s"           # 화면을 다시 그리는 간격

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

SHOW_MINUTES = 10        # 그래프에 보여 줄 최근 시간(분)

# 토글 버튼에 보일 이름과 그래프 색 (항목마다 색을 고정한다)
SERIES = {
    "eCO2": ("eco2", "#4e79a7"),
    "TVOC": ("tvoc", "#e15759"),
    "eCH2O": ("ech2o", "#76b7b2"),
    "PM2.5": ("pm25", "#f28e2b"),
    "PM10": ("pm10", "#59a14f"),
    "Temp": ("temperature", "#b07aa1"),
    "Humidity": ("humidity", "#9c755f")
}
DEFAULT_SERIES = ["eCO2"]

VALUE = "값"
PERCENT = "변화율(%)"

# 요약 페이지: 학교 교실 공기질 유지 기준 (학교보건법 시행규칙)
STANDARDS = {
    "eco2": ("eCO2", "ppm", 1000),
    "pm25": ("PM2.5", "µg/m³", 35),
    "pm10": ("PM10", "µg/m³", 75)
}
CAUTION_RATIO = 0.8      # 기준의 80%를 넘으면 '주의'


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
# 3. 수집기 (화면과 따로 계속 돈다)
# =========================

class Collector:

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = deque(maxlen=MAX_ROWS)
        self.status = "센서 연결 중..."
        self.connected = False
        self.count = 0

        session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(DATA_DIR, f"session_{session}.csv")
        self.file = None
        self.writer = None

        threading.Thread(target=self.run, daemon=True).start()

    def save(self, result):

        # 첫 데이터가 들어올 때 파일을 만든다. (빈 파일 방지)
        if self.file is None:
            os.makedirs(DATA_DIR, exist_ok=True)
            self.file = open(self.csv_file, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=COLUMNS)
            self.writer.writeheader()

        self.writer.writerow(result)
        self.file.flush()

        row = {
            "time": datetime.now(),
            "eco2": result["eco2_raw"],
            "tvoc": result["tvoc_raw"],
            "ech2o": result["ech2o_raw"],
            "pm25": result["pm25_raw"],
            "pm10": result["pm10_raw"],
            "temperature": decode_hex(result["temperature_hex"]),
            "humidity": decode_hex(result["humidity_hex"])
        }

        with self.lock:
            self.rows.append(row)
            self.count += 1

    def read_frames(self, ser):

        buffer = bytearray()

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
                self.save(result)

    def run(self):

        # 센서가 끊기면 몇 초 뒤 다시 연결한다.
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
                    self.connected = True
                    self.status = "수집 중"
                    self.read_frames(ser)

            except serial.SerialException as error:
                self.connected = False
                self.status = (
                    f"센서에 연결할 수 없습니다 ({error}). "
                    f"{RETRY_SECONDS}초 뒤 다시 시도합니다."
                )
                time.sleep(RETRY_SECONDS)

    def snapshot(self):
        with self.lock:
            return list(self.rows), self.count


# 브라우저를 여러 개 열거나 화면이 다시 그려져도 수집기는 하나만 돈다.
@st.cache_resource
def get_collector():
    return Collector()


# =========================
# 4. 그래프 페이지
# =========================

@st.fragment(run_every=REFRESH)
def live_chart(selected, mode):

    if not collector.connected:
        st.warning(collector.status)

    rows, _ = collector.snapshot()

    if not rows:
        st.info("데이터를 기다리는 중입니다.")
        return

    if not selected:
        st.info("위에서 보고 싶은 항목을 누르세요.")
        return

    df = pd.DataFrame(rows)
    df = df[df["time"] >= df["time"].iloc[-1] - timedelta(minutes=SHOW_MINUTES)]

    # 선택한 항목만 골라 '항목, 값' 형태의 긴 표로 바꾼다.
    long = df.rename(
        columns={column: name for name, (column, _) in SERIES.items()}
    ).melt(id_vars="time", value_vars=selected, var_name="item", value_name="value")

    if mode == PERCENT:
        # 그래프 맨 왼쪽(처음) 값을 0%로 두고 얼마나 늘고 줄었는지 계산한다.
        # 크기가 다른 항목(eCO2 약 900, PM2.5 약 4)도 한 그래프에서 비교할 수 있다.
        first = long.groupby("item")["value"].transform("first")
        long["change"] = (long["value"] - first) / first * 100

        # 처음 값이 0이면 나눌 수 없으므로 그 항목은 뺀다.
        zero = sorted(set(long.loc[first == 0, "item"]))
        long = long[first != 0]

        if zero:
            st.caption(f"처음 값이 0이라 변화율을 계산할 수 없는 항목: {', '.join(zero)}")

        y = alt.Y("change:Q", title="Change from first value (%)")
        tooltip = ["time:T", "item:N", "value:Q",
                   alt.Tooltip("change:Q", format="+.1f", title="change %")]
    else:
        y = alt.Y("value:Q", title=None)
        tooltip = ["time:T", "item:N", "value:Q"]

    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
        y=y,
        color=alt.Color(
            "item:N", title=None,
            # 켜진 항목만 범례에 보이되, 항목별 색은 항상 같게
            scale=alt.Scale(
                domain=selected,
                range=[SERIES[name][1] for name in selected]
            ),
            legend=alt.Legend(orient="top")
        ),
        tooltip=tooltip
    )

    chart = lines

    if mode == PERCENT:
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="gray", strokeDash=[4, 4]
        ).encode(y="y:Q")
        chart = zero_line + lines

    st.altair_chart(chart.properties(height=650), use_container_width=True)


def chart_page():

    # 누를 때마다 켜지고 꺼지는 토글 버튼
    selected = st.pills(
        "항목", list(SERIES), selection_mode="multi",
        default=DEFAULT_SERIES, label_visibility="collapsed"
    )

    # 실제 값으로 볼지, 처음 값 대비 변화율로 볼지 고른다.
    mode = st.segmented_control(
        "보기", [VALUE, PERCENT], default=PERCENT,
        label_visibility="collapsed"
    ) or PERCENT

    live_chart(selected, mode)


# =========================
# 5. 요약 페이지 (상태 판단, 값 카드, 그래프 4개)
# =========================

def judge(latest):

    over, near = [], []

    for key, (name, unit, limit) in STANDARDS.items():
        value = latest[key]
        text = f"{name} {value:,.0f}{unit} (기준 {limit:,}{unit})"

        if value >= limit:
            over.append(text)
        elif value >= limit * CAUTION_RATIO:
            near.append(text)

    if over:
        st.error("환기 필요: " + ", ".join(over))
    elif near:
        st.warning("주의: " + ", ".join(near))
    else:
        st.success("좋음: 모든 값이 기준보다 충분히 낮습니다.")


@st.fragment(run_every=REFRESH)
def live_view(minutes):

    rows, count = collector.snapshot()

    if collector.connected:
        st.caption(f"상태: {collector.status} · 저장한 데이터 {count:,}개")
    else:
        st.warning(collector.status)

    if not rows:
        st.info("데이터를 기다리는 중입니다. 센서는 약 2초마다 값을 보냅니다.")
        return

    df = pd.DataFrame(rows).set_index("time")
    latest = df.iloc[-1]

    # 1분 전 값과 비교해 얼마나 변했는지 보여 준다.
    before = df[df.index <= df.index[-1] - timedelta(minutes=1)]
    previous = before.iloc[-1] if not before.empty else df.iloc[0]

    judge(latest)

    cards = [
        ("eCO2 (ppm)", "eco2", "{:,.0f}", "inverse"),
        ("TVOC", "tvoc", "{:,.0f}", "inverse"),
        ("PM2.5 (µg/m³)", "pm25", "{:,.0f}", "inverse"),
        ("PM10 (µg/m³)", "pm10", "{:,.0f}", "inverse"),
        ("온도 (°C)", "temperature", "{:.1f}", "off"),
        ("습도 (%)", "humidity", "{:.1f}", "off")
    ]

    for col, (label, key, fmt, color) in zip(st.columns(len(cards)), cards):
        change = latest[key] - previous[key]
        col.metric(
            label, fmt.format(latest[key]),
            delta=f"{change:+.1f} (1분)", delta_color=color
        )

    recent = df[df.index >= df.index[-1] - timedelta(minutes=minutes)]

    left, right = st.columns(2)
    left.subheader("eCO2")
    left.line_chart(recent[["eco2"]])
    right.subheader("TVOC")
    right.line_chart(recent[["tvoc"]])

    left.subheader("미세먼지 PM2.5 · PM10")
    left.line_chart(recent[["pm25", "pm10"]])
    right.subheader("온도 · 습도")
    right.line_chart(recent[["temperature", "humidity"]])

    st.caption(f"마지막 측정: {df.index[-1]:%H:%M:%S}")


def overview_page():

    st.title("라온마루 실시간 공기질")

    with st.sidebar:
        st.header("설정")
        minutes = st.slider("그래프에 보여 줄 시간(분)", 1, 60, 10)
        st.caption(f"저장 파일: {collector.csv_file}")
        st.caption("판단 기준: 학교 교실 공기질 유지 기준")
        for name, unit, limit in STANDARDS.values():
            st.caption(f"- {name}: {limit:,} {unit}")

    live_view(minutes)


# =========================
# 6. 실행
# =========================

st.set_page_config(page_title="라온마루 공기질", layout="wide")

collector = get_collector()

page = st.navigation([
    st.Page(chart_page, title="그래프", url_path="chart", default=True),
    st.Page(overview_page, title="요약", url_path="overview")
])
page.run()
