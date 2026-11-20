
# 실행: streamlit run m701_dashboard.py
# 센서 값을 수집해 CSV로 저장하면서, 같은 값을 웹 대시보드에 바로 보여 준다.
# 같은 Wi-Fi의 다른 기기에서는 http://<라즈베리파이 IP>:8501 로 접속한다.
# 페이지 두 개(그래프 / 요약)가 같은 수집기를 함께 쓴다. 왼쪽 메뉴에서 고른다.
# models/printer_tree.joblib 이 있으면 3D 프린터 가동 여부도 함께 보여 준다.
# (모델은 python m701_train_classifier.py 로 만든다.)
# models/forecast_*.joblib 이 있으면 10분 뒤 예상 값도 보여 준다.
# (모델은 python m701_train_forecast.py 로 만든다.)

import csv
import glob
import json
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

# 3D 프린터 상태 판단 (m701_live_classify.py 와 같은 방법)
MODEL_FILE = os.path.join("models", "printer_tree.joblib")
RESAMPLE = "10s"         # 학습 데이터와 같은 10초 평균
WINDOW = "60s"           # 학습 데이터와 같은 60초 이동평균
CONFIRM_BINS = 3         # 최근 3구간(30초)의 다수결로 상태를 정한다
PRINTING_COLOR = "#e15759"

# 10분 뒤 예측 (m701_train_forecast.py 로 만든 모델을 모두 불러온다)
FORECAST_FILES = os.path.join("models", "forecast_*.joblib")
FORECAST_NAMES = {
    "eco2": ("eCO2", "ppm", "{:,.0f}"),
    "tvoc": ("TVOC", "", "{:,.0f}"),
    "pm25": ("PM2.5", "µg/m³", "{:,.0f}"),
    "pm10": ("PM10", "µg/m³", "{:,.0f}"),
    "temperature": ("온도", "°C", "{:.1f}"),
    "humidity": ("습도", "%", "{:.1f}")
}

# LED/부저 알림 상태 (m701_alert.py 가 판단할 때마다 이 파일에 적는다)
ALERT_STATUS_FILE = os.path.join(DATA_DIR, "alert_status.json")
ALERT_STALE_SECONDS = 40  # 이보다 오래 갱신되지 않으면 알림 프로그램이 멈춘 것으로 본다
ALERT_LIGHTS = {
    0: (":green-background[좋음]", "초록 LED"),
    1: (":orange-background[주의]", "노란 LED"),
    2: (":red-background[미리 환기]", "빨간 LED 깜빡임 · 부저 짧게 2번"),
    3: (":red-background[환기 필요]", "빨간 LED · 부저 길게 3번")
}


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
# 4. 3D 프린터 상태 판단
# =========================

# 모델 파일이 바뀌면(다시 학습하면) 새로 불러온다.
@st.cache_resource
def load_model(path, modified):
    import joblib  # scikit-learn 이 없어도 대시보드는 열리도록 여기서 불러온다.
    return joblib.load(path)


def get_model():

    if not os.path.exists(MODEL_FILE):
        return None, "모델이 없습니다. python m701_train_classifier.py 로 먼저 학습하세요."

    try:
        return load_model(MODEL_FILE, os.path.getmtime(MODEL_FILE)), None
    except Exception as error:
        return None, f"모델을 불러오지 못했습니다: {error}"


def predict_states(rows, bundle):

    frame = pd.DataFrame(rows).set_index("time")

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

    vote = (features["probability"] >= 0.5).astype(float)
    features["predict"] = (vote.rolling(CONFIRM_BINS, min_periods=1).mean() > 0.5).astype(int)

    return features


def printer_status(rows):

    bundle, message = get_model()

    if bundle is None:
        st.caption(f"3D 프린터 판단: {message}")
        return None

    states = predict_states(rows, bundle)

    if states.empty:
        minutes = (rows[-1]["time"] - rows[0]["time"]).total_seconds() / 60
        st.markdown(f"**3D 프린터:** :gray-background[판단 준비 중] "
                    f"(5분 이상 데이터가 필요합니다, 지금 {minutes:.1f}분)")
        return None

    latest = states.iloc[-1]
    probability = latest["probability"] * 100

    if latest["predict"] == 1:
        st.markdown(f"### 3D 프린터: :red-background[가동 중] "
                    f"<small>가동 확률 {probability:.0f}%</small>", unsafe_allow_html=True)
    else:
        st.markdown(f"### 3D 프린터: :green-background[평시] "
                    f"<small>가동 확률 {probability:.0f}%</small>", unsafe_allow_html=True)

    return states


# =========================
# 5. 10분 뒤 예측
# =========================

def get_forecast_models():

    models = {}

    for path in sorted(glob.glob(FORECAST_FILES)):
        try:
            bundle = load_model(path, os.path.getmtime(path))
        except Exception:
            continue
        models[bundle["target"]] = bundle

    # 화면에 보일 순서를 eCO2, TVOC, ... 순서로 맞춘다.
    return {t: models[t] for t in FORECAST_NAMES if t in models}


def predict_future(rows, bundle, states):

    target = bundle["target"]
    columns = list(dict.fromkeys(bundle["inputs"] + [target]))
    frame = pd.DataFrame(rows).set_index("time")

    # 학습 데이터와 같은 10초 평균 + 60초 이동평균
    clean = frame[columns].resample(RESAMPLE).mean()
    clean = clean[clean.index + pd.Timedelta(RESAMPLE) <= frame.index[-1]]
    smooth = clean.rolling(WINDOW, min_periods=1).mean()
    smooth = smooth[clean[columns[0]].notna()]

    if smooth.empty:
        return None

    # 학습할 때와 같은 특징 (m701_train_forecast.py 참고)
    features = pd.DataFrame(index=smooth.index)

    for name in columns:
        now = smooth[name]
        features[f"{name}_now"] = now

        for lag in bundle["lags_minutes"]:
            before = now.shift(freq=f"{lag}min").reindex(now.index)
            features[f"{name}_change_{lag}m"] = now - before

    for name in ["eco2", "tvoc"]:
        now = smooth[name]
        lowest = now.rolling(f"{bundle['baseline_minutes']}min").min()
        features[f"{name}_rise"] = now - lowest

    if bundle["with_state"]:
        if states is None:
            return None
        features["printing"] = states["predict"].reindex(features.index)

    # 10분 전 값이 아직 없으면(수집 시작 10분 이내) 예측하지 않는다.
    if features.iloc[[-1]][bundle["features"]].isna().any(axis=None):
        return None

    # 10초 구간마다 "이 시점에서 본 10분 뒤 값"을 모두 계산해 그래프로 잇는다.
    # 지난 부분은 10분 전에 예상했던 값이라 실제 값과 비교해 볼 수 있다.
    ready = features.dropna(subset=bundle["features"])
    now = ready[f"{target}_now"]
    change = bundle["model"].predict(ready[bundle["features"]])
    horizon = pd.Timedelta(minutes=bundle["horizon_minutes"])

    path = pd.DataFrame({"time": ready.index + horizon, "value": now.values + change})

    return {
        "time": ready.index[-1],
        "now": now.iloc[-1],
        "value": path["value"].iloc[-1],
        "horizon": bundle["horizon_minutes"],
        "path": path
    }


def future_values(rows, states):

    models = get_forecast_models()
    results = {target: predict_future(rows, bundle, states)
               for target, bundle in models.items()}

    return results, models


def forecast_note(results, models):

    if not models:
        st.caption("10분 뒤 예측: 모델이 없습니다. "
                   "python m701_train_forecast.py 로 먼저 학습하세요.")
        return

    names = ", ".join(FORECAST_NAMES[target][0] for target in models)
    horizon = next(iter(models.values()))["horizon_minutes"]

    if any(result is None for result in results.values()):
        st.caption(f"{horizon}분 뒤 예측({names}): 준비 중입니다. "
                   f"예측하려면 {horizon}분 이상 데이터가 필요합니다.")
    else:
        st.caption(f"점선은 모델이 예상한 {horizon}분 뒤 값입니다({names}). "
                   f"오른쪽 끝이 지금부터 {horizon}분 뒤 예상이고, "
                   f"지난 부분은 {horizon}분 전에 예상했던 값이라 실선(실제 값)과 비교해 볼 수 있습니다.")


# 요약 페이지의 작은 그래프: 실제 값은 실선, 예상 값은 점선
def small_chart(container, title, recent, columns, results):

    container.subheader(title)
    start = recent.index[0]
    parts = []

    for column in columns:
        name = FORECAST_NAMES[column][0]
        actual = recent[column].reset_index()
        actual.columns = ["time", "value"]
        parts.append(actual.assign(item=name, kind="실제"))

        result = results.get(column)

        if result is not None:
            path = result["path"]
            parts.append(path[path["time"] >= start].assign(item=name, kind="10분 뒤 예상"))

    data = pd.concat(parts, ignore_index=True)

    chart = alt.Chart(data).mark_line().encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("item:N", title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash(
            "kind:N", title=None, legend=alt.Legend(orient="top"),
            scale=alt.Scale(domain=["실제", "10분 뒤 예상"], range=[[1, 0], [6, 4]])
        ),
        tooltip=["time:T", "item:N", "kind:N", alt.Tooltip("value:Q", format=",.1f")]
    )

    container.altair_chart(chart.properties(height=260), use_container_width=True)


def forecast_alert(results):

    # 학교 기준을 넘을 것으로 예상되면 미리 알려 준다.
    for key, (name, unit, limit) in STANDARDS.items():
        result = results.get(key)

        if result is None:
            continue

        value = result["value"]
        text = (f"{result['horizon']}분 뒤 {name}: {value:,.0f}{unit} 예상 "
                f"(기준 {limit:,}{unit})")

        if value >= limit:
            st.error("미리 환기하세요. " + text)
        elif value >= limit * CAUTION_RATIO:
            st.warning("곧 기준에 가까워집니다. " + text)


# LED/부저 알림 상태. 대시보드가 따로 판단하지 않고
# m701_alert.py 가 실제로 켠 LED 상태를 그대로 보여 준다.
def alert_status():

    if not os.path.exists(ALERT_STATUS_FILE):
        st.caption("LED 알림: 꺼져 있습니다. (python m701_alert.py 로 켭니다)")
        return

    try:
        with open(ALERT_STATUS_FILE, encoding="utf-8") as file:
            status = json.load(file)
        updated = datetime.fromisoformat(status["updated"])
    except (OSError, ValueError, KeyError):
        return  # 파일을 바꿔 끼우는 순간이면 다음 갱신 때 보여 준다.

    age = (datetime.now() - updated).total_seconds()

    if not status.get("running", True):
        st.caption(f"LED 알림: 꺼져 있습니다. (마지막 실행 {updated:%H:%M:%S}, "
                   "python m701_alert.py 로 켭니다)")
        return

    if age > ALERT_STALE_SECONDS:
        st.caption(f"LED 알림: 응답이 없습니다. (마지막 갱신 {age:.0f}초 전) "
                   "m701_alert.py 가 실행 중인지 확인하세요.")
        return

    level = status.get("level")
    reasons = status.get("reasons") or []

    if level is None:
        st.markdown("**LED 알림:** :gray-background[LED 꺼짐] " + ", ".join(reasons))
        return

    badge, light = ALERT_LIGHTS[level]
    notes = []

    if status.get("mute"):
        notes.append("부저 끔")
    if not status.get("gpio", True):
        notes.append("GPIO 없음, 화면에만 표시")

    extra = f" ({', '.join(notes)})" if notes else ""
    st.markdown(f"**LED 알림:** {badge} {light}{extra}")

    if reasons:
        st.caption("이유: " + ", ".join(reasons))


# =========================
# 6. 그래프 페이지
# =========================

@st.fragment(run_every=REFRESH)
def live_chart(selected, mode):

    if not collector.connected:
        st.warning(collector.status)

    rows, _ = collector.snapshot()

    if not rows:
        st.info("데이터를 기다리는 중입니다.")
        return

    states = printer_status(rows)
    results, models = future_values(rows, states)
    forecast_note(results, models)
    forecast_alert(results)
    alert_status()

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

    x = alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M:%S"))
    color = alt.Color(
        "item:N", title=None,
        # 켜진 항목만 범례에 보이되, 항목별 색은 항상 같게
        scale=alt.Scale(
            domain=selected,
            range=[SERIES[name][1] for name in selected]
        ),
        legend=alt.Legend(orient="top")
    )

    lines = alt.Chart(long).mark_line().encode(
        x=x, y=y, color=color, tooltip=tooltip
    )

    chart = lines

    # 예측 모델이 있는 항목은 예상 값을 같은 색 점선으로 그린다.
    # 오른쪽 끝이 10분 뒤라서 가로축이 지금보다 10분 더 길어진다.
    future = []
    start = df["time"].iloc[0]

    for name in selected:
        column = SERIES[name][0]
        result = results.get(column)

        if result is None:
            continue

        first = df[column].iloc[0]

        if mode == PERCENT and first == 0:
            continue

        path = result["path"]
        path = path[path["time"] >= start]

        for when, value in zip(path["time"], path["value"]):
            future.append({
                "time": when, "item": name, "value": value,
                "change": (value - first) / first * 100 if first else 0
            })

    if future:
        dashed = alt.Chart(pd.DataFrame(future)).mark_line(
            strokeDash=[6, 4]
        ).encode(x=x, y=y, color=color, tooltip=tooltip)
        chart = chart + dashed

    if mode == PERCENT:
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="gray", strokeDash=[4, 4]
        ).encode(y="y:Q")
        chart = zero_line + chart

    # 모델이 가동 중이라고 판단한 10초 구간은 배경을 붉게 칠한다.
    if states is not None:
        busy = states[(states["predict"] == 1) & (states.index >= df["time"].iloc[0])]

        if not busy.empty:
            spans = pd.DataFrame({"start": busy.index,
                                  "end": busy.index + pd.Timedelta(RESAMPLE)})
            shade = alt.Chart(spans).mark_rect(
                color=PRINTING_COLOR, opacity=0.12
            ).encode(x="start:T", x2="end:T")
            chart = shade + chart

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
# 7. 요약 페이지 (상태 판단, 값 카드, 그래프 4개)
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
    states = printer_status(rows)

    # 10분 뒤 예상 값 (기준을 넘을 것 같으면 먼저 알려 준다)
    results, models = future_values(rows, states)
    forecast_alert(results)
    alert_status()

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

    forecast_note(results, models)

    recent = df[df.index >= df.index[-1] - timedelta(minutes=minutes)]

    left, right = st.columns(2)
    small_chart(left, "eCO2", recent, ["eco2"], results)
    small_chart(right, "TVOC", recent, ["tvoc"], results)
    small_chart(left, "미세먼지 PM2.5 · PM10", recent, ["pm25", "pm10"], results)
    small_chart(right, "온도 · 습도", recent, ["temperature", "humidity"], results)

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
# 8. 실행
# =========================

st.set_page_config(page_title="라온마루 공기질", layout="wide")

collector = get_collector()

page = st.navigation([
    st.Page(chart_page, title="그래프", url_path="chart", default=True),
    st.Page(overview_page, title="요약", url_path="overview")
])
page.run()
