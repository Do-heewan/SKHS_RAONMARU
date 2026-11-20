
# 실행: python m701_alert.py
# 대시보드(또는 m701_csv.py)가 저장하고 있는 data/session_*.csv 를 따라 읽으면서
# 지금 값과 10분 뒤 예상 값을 학교 기준과 비교해 LED와 부저로 알린다.
# 센서를 직접 열지 않으므로 대시보드와 동시에 켜 둘 수 있다.
#
#   초록 LED  : 좋음
#   노란 LED  : 주의 (기준의 80% 이상이거나 3D 프린터 가동 중)
#   빨간 LED 깜빡임 + 짧은 삐 2번 : 미리 환기 (10분 뒤 기준을 넘을 것으로 예상)
#   빨간 LED 켜짐   + 긴 삐 3번   : 환기 필요 (지금 기준을 넘음, 몇 분마다 다시 울림)
#
# 배선 (BCM 번호, 바꾸려면 --green --yellow --red --buzzer 옵션)
#   GPIO17 -> 220Ω 저항 -> 초록 LED(+) , LED(-) -> GND
#   GPIO27 -> 220Ω 저항 -> 노란 LED(+) , LED(-) -> GND
#   GPIO22 -> 220Ω 저항 -> 빨간 LED(+) , LED(-) -> GND
#   GPIO18 -> 능동 부저(+) , 부저(-) -> GND  (3.3V로 울리는 능동 부저)
#
# 배선 확인:   python m701_alert.py --test
# 센서 없이 연습: python m701_alert.py --replay data/labeled/state_20261117_084512_grapepi.csv --speed 60
#
# 라즈베리파이 5는 gpiozero + lgpio 로 핀을 쓴다. (RPi.GPIO 는 라즈베리파이 5에서 동작하지 않는다.)
# venv 에서 핀을 못 잡으면 venv 를 python -m venv --system-site-packages 로 만들거나
# sudo apt install python3-lgpio 를 확인한다. GPIO 가 없는 PC 에서는 화면에만 표시한다.
#
# 모델이 있으면 함께 쓴다. (없어도 지금 값만으로 알림은 동작한다.)
#   models/printer_tree.joblib     : python m701_train_classifier.py
#   models/forecast_<항목>.joblib  : python m701_train_forecast.py

import argparse
import csv
import glob
import os
import time
from collections import deque
from datetime import datetime

import pandas as pd

# =========================
# 1. 기본 설정
# =========================

DATA_DIR = "data"
SESSION_FILES = os.path.join(DATA_DIR, "session_*.csv")
CLASSIFIER_FILE = os.path.join("models", "printer_tree.joblib")
FORECAST_FILES = os.path.join("models", "forecast_*.joblib")

# 학교 교실 공기질 유지 기준 (대시보드와 같음). TVOC 는 단위가 확인되지 않아 판단하지 않는다.
STANDARDS = {
    "eco2": ("eCO2", "ppm", 1000),
    "pm25": ("PM2.5", "µg/m³", 35),
    "pm10": ("PM10", "µg/m³", 75)
}
CAUTION_RATIO = 0.8

LEVELS = ["좋음", "주의", "미리 환기", "환기 필요"]
GOOD, CAUTION, SOON, VENTILATE = range(4)

CHECK_SECONDS = 10    # 몇 초마다 판단할지
AVERAGE_SECONDS = 60  # 지금 값은 최근 1분 평균으로 본다 (한 번 튀는 값에 울리지 않도록)
NO_DATA_SECONDS = 30  # 이 시간 동안 새 데이터가 없으면 경고

# 학습 데이터를 만들 때(m701_state_clean.py)와 똑같이 정리해야 모델이 제대로 판단한다.
RESAMPLE = "10s"
WINDOW = "60s"
CONFIRM_BINS = 3
HISTORY_ROWS = 1800   # 2초 간격이면 약 1시간


# "19 00" -> 25.0 (첫 바이트: 정수, 둘째 바이트: 소수 첫째 자리)
def decode_hex(text):
    whole, tenth = text.split()
    return int(whole, 16) + int(tenth, 16) / 10


# CSV 한 줄(m701_csv.py 형식)을 계산하기 쉬운 값으로 바꾼다.
def to_row(record):
    return {
        "time": datetime.fromisoformat(record["timestamp"]).replace(tzinfo=None),
        "eco2": int(record["eco2_raw"]),
        "tvoc": int(record["tvoc_raw"]),
        "pm25": int(record["pm25_raw"]),
        "pm10": int(record["pm10_raw"]),
        "temperature": decode_hex(record["temperature_hex"]),
        "humidity": decode_hex(record["humidity_hex"])
    }


# =========================
# 2. LED 와 부저
# =========================

class Signal:

    def __init__(self, args):
        self.mute = args.mute
        self.real = False

        try:
            from gpiozero import LED, Buzzer
            self.green = LED(args.green)
            self.yellow = LED(args.yellow)
            self.red = LED(args.red)
            self.buzzer = Buzzer(args.buzzer)
            self.real = True
            print(f"GPIO: 초록 {args.green}, 노랑 {args.yellow}, 빨강 {args.red}, "
                  f"부저 {args.buzzer}" + (" (부저 끔)" if self.mute else ""))
        except Exception as error:
            print(f"GPIO를 쓸 수 없어 화면에만 표시합니다: {error}")

    def show(self, level):

        if not self.real:
            return

        # on()/off() 를 부르면 깜빡이던 것도 멈춘다.
        for led in [self.green, self.yellow, self.red]:
            led.off()

        if level == GOOD:
            self.green.on()
        elif level == CAUTION:
            self.yellow.on()
        elif level == SOON:
            self.red.blink(on_time=0.5, off_time=0.5)
        elif level == VENTILATE:
            self.red.on()

    def beep(self, count, seconds):

        if self.mute:
            return

        if not self.real:
            print("  (부저) " + " ".join(["삐-" if seconds > 0.3 else "삐"] * count))
            return

        # background=True: 울리는 동안 판단을 멈추지 않는다.
        self.buzzer.beep(on_time=seconds, off_time=0.2, n=count, background=True)

    def close(self):
        if self.real:
            for device in [self.green, self.yellow, self.red, self.buzzer]:
                device.off()
                device.close()


def wiring_test(signal):

    print("배선 확인: 초록 -> 노랑 -> 빨강 -> 부저 순서로 1초씩 켭니다.")

    for level, name in [(GOOD, "초록"), (CAUTION, "노랑"), (VENTILATE, "빨강")]:
        print(f"  {name} LED")
        signal.show(level)
        time.sleep(1)

    signal.show(None)
    print("  부저")
    signal.beep(1, 1.0)
    time.sleep(1.5)
    print("끝. 켜지지 않은 것이 있으면 배선과 핀 번호를 확인하세요.")


# =========================
# 3. 모델 (프린터 상태, 10분 뒤 예측)
# =========================

def load_models(args):

    classifier = None
    forecasts = {}

    try:
        import joblib  # scikit-learn 이 없어도 지금 값 알림은 되도록 여기서 불러온다.
    except ImportError:
        print("joblib/scikit-learn 이 없어 모델 없이 지금 값만으로 알립니다.")
        return classifier, forecasts

    if os.path.exists(args.classifier):
        classifier = joblib.load(args.classifier)
        print(f"프린터 상태 모델: {args.classifier}")
    else:
        print("프린터 상태 모델이 없습니다. (python m701_train_classifier.py)")

    if not args.no_forecast:
        for path in sorted(glob.glob(FORECAST_FILES)):
            bundle = joblib.load(path)
            # 기준이 있는 항목만 알림에 쓴다.
            if bundle["target"] in STANDARDS:
                forecasts[bundle["target"]] = bundle
                print(f"예측 모델: {path}")

        if not forecasts:
            print("eCO2/PM 예측 모델이 없습니다. (python m701_train_forecast.py)")

    return classifier, forecasts


def smooth_frame(history, columns):

    frame = pd.DataFrame(history).set_index("time")

    # 10초 평균 + 60초 이동평균. 아직 10초가 다 차지 않은 마지막 구간은 뺀다.
    clean = frame[columns].resample(RESAMPLE).mean()
    clean = clean[clean.index + pd.Timedelta(RESAMPLE) <= frame.index[-1]]
    smooth = clean.rolling(WINDOW, min_periods=1).mean()
    return smooth[clean[columns[0]].notna()]


# m701_train_classifier.py / m701_live_classify.py 와 같은 특징
def predict_states(history, bundle):

    smooth = smooth_frame(history, ["eco2", "tvoc", "humidity"])

    if smooth.empty:
        return smooth

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

    features = features.dropna()

    if features.empty:
        return features

    model = bundle["model"]
    printing = list(model.classes_).index(1)
    features["probability"] = model.predict_proba(features[bundle["features"]])[:, printing]

    vote = (features["probability"] >= 0.5).astype(float)
    features["predict"] = (vote.rolling(CONFIRM_BINS, min_periods=1).mean() > 0.5).astype(int)

    return features


# m701_train_forecast.py 와 같은 특징으로 가장 최근 시점의 10분 뒤 값을 예측한다.
def predict_future(history, bundle, states):

    target = bundle["target"]
    columns = list(dict.fromkeys(bundle["inputs"] + [target]))
    smooth = smooth_frame(history, columns)

    if smooth.empty:
        return None

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
        if states is None or states.empty:
            return None
        features["printing"] = states["predict"].reindex(features.index)

    latest = features.iloc[[-1]]

    # 10분 전 값이 아직 없으면(데이터 10분 이내) 예측하지 않는다.
    if latest[bundle["features"]].isna().any(axis=None):
        return None

    now = latest[f"{target}_now"].iloc[0]
    return now + bundle["model"].predict(latest[bundle["features"]])[0]


# =========================
# 4. 판단
# =========================

def evaluate(history, classifier, forecasts):

    level = GOOD
    reasons = []
    parts = []

    # 1) 지금 값 (최근 1분 평균)
    frame = pd.DataFrame(history).set_index("time")
    recent = frame[frame.index >= frame.index[-1] - pd.Timedelta(seconds=AVERAGE_SECONDS)]
    current = recent.mean()

    for key, (name, unit, limit) in STANDARDS.items():
        value = current[key]

        if value >= limit:
            level = max(level, VENTILATE)
            reasons.append(f"지금 {name} {value:,.0f}{unit} (기준 {limit:,})")
        elif value >= limit * CAUTION_RATIO:
            level = max(level, CAUTION)
            reasons.append(f"지금 {name} 값이 기준의 {value / limit * 100:.0f}%")

    # 2) 3D 프린터 상태
    states = predict_states(history, classifier) if classifier else None

    if states is not None and not states.empty:
        printing = states["predict"].iloc[-1] == 1
        parts.append("프린터 " + ("가동 중" if printing else "평시"))

        if printing:
            level = max(level, CAUTION)
            reasons.append("3D 프린터 가동 중")

    # 3) 10분 뒤 예상
    for key, bundle in forecasts.items():
        name, unit, limit = STANDARDS[key]
        value = predict_future(history, bundle, states)

        if value is None:
            parts.append(f"{name} 예측 준비 중")
            continue

        parts.append(f"{bundle['horizon_minutes']}분 뒤 {name} {value:,.0f}")

        if value >= limit:
            level = max(level, SOON)
            reasons.append(f"{bundle['horizon_minutes']}분 뒤 {name} {value:,.0f}{unit} 예상 "
                           f"(기준 {limit:,})")
        elif value >= limit * CAUTION_RATIO:
            level = max(level, CAUTION)
            reasons.append(f"{bundle['horizon_minutes']}분 뒤 {name} 값이 기준의 "
                           f"{value / limit * 100:.0f}% 예상")

    values = (f"eCO2 {current['eco2']:,.0f} | PM2.5 {current['pm25']:.0f} | "
              f"PM10 {current['pm10']:.0f}")

    return level, reasons, " | ".join([values] + parts)


class Alarm:

    def __init__(self, signal, classifier, forecasts, repeat_minutes):
        self.signal = signal
        self.classifier = classifier
        self.forecasts = forecasts
        self.repeat = pd.Timedelta(minutes=repeat_minutes)
        self.level = None
        self.last_beep = {}  # 단계별로 마지막에 부저를 울린 시각

    def check(self, history, now):

        level, reasons, text = evaluate(list(history), self.classifier, self.forecasts)
        changed = level != self.level
        mark = "  <== 바뀜" if self.level is not None and changed else ""

        print(f"[{now:%H:%M:%S}] {LEVELS[level]:<5} | {text}{mark}")

        if changed and reasons:
            print("  이유: " + ", ".join(reasons))

        if changed:
            self.signal.show(level)

        # 부저는 단계가 올라갈 때만 울린다. 수업 중에 계속 울리면 시끄럽기 때문이다.
        # 값이 기준선 근처에서 오르내려도 --repeat 분 안에는 같은 단계로 다시 울리지 않는다.
        rising = changed and (self.level is None or level > self.level)
        last = self.last_beep.get(level)
        waited = last is None or now - last >= self.repeat

        if level == SOON and rising and waited:
            self.signal.beep(2, 0.2)
            self.last_beep[level] = now

        elif level == VENTILATE:
            # 환기 필요가 계속되면 --repeat 분마다 다시 울린다. (0이면 처음 한 번만)
            again = (not changed and self.repeat > pd.Timedelta(0) and last is not None
                     and now - last >= self.repeat)

            if (rising and waited) or again:
                self.signal.beep(3, 0.6)
                self.last_beep[level] = now

        self.level = level


# =========================
# 5. 데이터 따라 읽기 / 저장된 파일로 연습하기
# =========================

class Follower:

    # 수집 프로그램이 쓰고 있는 CSV 파일에 새로 추가된 줄만 읽는다.
    # --input 을 주지 않으면 가장 최근 session_*.csv 를 따라가고,
    # 수집을 다시 시작해 새 파일이 생기면 그 파일로 옮겨 간다.

    def __init__(self, path):
        self.fixed = path
        self.path = None
        self.file = None
        self.header = None
        self.buffer = ""

    def newest(self):
        if self.fixed:
            return self.fixed
        files = glob.glob(SESSION_FILES)
        return max(files, key=os.path.getmtime) if files else None

    def read(self):

        path = self.newest()

        if path is None:
            return []

        if path != self.path:
            if self.file is not None:
                self.file.close()
            print(f"따라 읽는 파일: {path}")
            self.path = path
            self.file = open(path, newline="", encoding="utf-8")
            self.header = None
            self.buffer = ""

        # 쓰는 중인 마지막 줄은 다음에 마저 읽는다.
        self.buffer += self.file.read()
        *lines, self.buffer = self.buffer.split("\n")
        rows = []

        for line in lines:
            if not line.strip():
                continue

            values = next(csv.reader([line]))

            if self.header is None:
                self.header = values
                continue

            try:
                rows.append(to_row(dict(zip(self.header, values))))
            except (KeyError, ValueError):
                continue

        return rows


def run_follow(alarm, path):

    follower = Follower(path)
    history = deque(maxlen=HISTORY_ROWS)
    last_new = time.time()
    warned = False

    print("Ctrl+C to stop\n")

    try:
        while True:
            rows = follower.read()

            if rows:
                # 새 데이터가 들어왔을 때만 판단한다.
                history.extend(rows)
                last_new = time.time()
                warned = False
                alarm.check(history, history[-1]["time"])

            elif not history:
                if not warned:
                    print("데이터를 기다리는 중입니다. 대시보드나 m701_csv.py 를 먼저 켜세요.")
                    warned = True
            elif time.time() - last_new > NO_DATA_SECONDS:
                if not warned:
                    print(f"{NO_DATA_SECONDS}초 넘게 새 데이터가 없습니다. "
                          "수집 프로그램이 켜져 있는지 확인하세요. (LED 끔)")
                    alarm.signal.show(None)
                    alarm.level = None
                    warned = True

            time.sleep(CHECK_SECONDS)

    except KeyboardInterrupt:
        print("\n알림을 멈췄습니다.")


def run_replay(alarm, path, speed):

    df = pd.read_csv(path, dtype=str)
    history = deque(maxlen=HISTORY_ROWS)
    last_check = None

    print(f"연습용 파일: {path} ({len(df)}행, "
          + (f"{speed}배속)" if speed > 0 else "최대 속도)"))
    print("Ctrl+C to stop\n")

    try:
        for _, record in df.iterrows():
            row = to_row(record)

            if speed > 0 and history:
                gap = (row["time"] - history[-1]["time"]).total_seconds()
                time.sleep(max(gap, 0) / speed)

            history.append(row)

            if last_check is None or (row["time"] - last_check).total_seconds() >= CHECK_SECONDS:
                alarm.check(history, row["time"])
                last_check = row["time"]

        time.sleep(2)  # 마지막 부저 소리가 끝날 때까지 기다린다.

    except KeyboardInterrupt:
        print("\n연습을 멈췄습니다.")


# =========================
# 6. 실행
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        help="따라 읽을 수집 파일 (없으면 가장 최근 data/session_*.csv)")
    parser.add_argument("--replay",
                        help="저장된 수집 파일(CSV)을 처음부터 읽어 연습하기")
    parser.add_argument("--speed", type=float, default=0,
                        help="--replay 재생 배속 (0이면 기다리지 않고 바로)")
    parser.add_argument("--test", action="store_true",
                        help="LED와 부저를 하나씩 켜서 배선 확인")
    parser.add_argument("--green", type=int, default=17, help="초록 LED 핀 (BCM)")
    parser.add_argument("--yellow", type=int, default=27, help="노란 LED 핀 (BCM)")
    parser.add_argument("--red", type=int, default=22, help="빨간 LED 핀 (BCM)")
    parser.add_argument("--buzzer", type=int, default=18, help="부저 핀 (BCM)")
    parser.add_argument("--mute", action="store_true", help="부저를 울리지 않기 (LED만)")
    parser.add_argument("--repeat", type=float, default=5,
                        help="환기 필요가 계속될 때 몇 분마다 다시 울릴지 (0이면 한 번만)")
    parser.add_argument("--classifier", default=CLASSIFIER_FILE,
                        help="프린터 상태 모델 파일")
    parser.add_argument("--no-forecast", action="store_true",
                        help="10분 뒤 예측을 쓰지 않고 지금 값만으로 알리기")
    args = parser.parse_args()

    print("M701B LED/부저 환기 알림")
    signal = Signal(args)

    try:
        if args.test:
            wiring_test(signal)
            return

        classifier, forecasts = load_models(args)
        alarm = Alarm(signal, classifier, forecasts, args.repeat)

        if args.replay:
            run_replay(alarm, args.replay, args.speed)
        else:
            run_follow(alarm, args.input)

    finally:
        signal.close()


if __name__ == "__main__":
    main()
