
import argparse
import glob
import os

import numpy as np
import pandas as pd

# =========================
# 1. 기본 설정
# =========================
#
# 실제로 모은 데이터가 몇 시간뿐이라 모델을 학습하기에는 부족하다.
# 이 프로그램은 data/labeled/state_*.csv (라벨이 붙은 실제 데이터)를 참고해서
# 같은 형식의 데이터를 더 만든다. 만드는 방법은 두 가지다.
#
#   1) 변형(aug_real): 실제 세션을 복사한 뒤 센서 차이(값 이동), 반응 크기,
#      작은 잡음을 조금씩 바꾼다. 라벨은 그대로 둔다.
#   2) 시뮬레이션(aug_sim): 실제 데이터에서 잰 값(기준 농도, 잡음,
#      프린터 가동 때 오르고 내리는 빠르기)으로 새 세션을 처음부터 만든다.
#      가동 시간과 횟수를 마음대로 정할 수 있어 긴 출력도 들어간다.
#
# 결과 파일은 실제 데이터와 같은 형식이라
# python m701_state_clean.py --all 로 바로 전처리할 수 있다.
# 만든 데이터는 진짜 측정값이 아니므로, 모델을 평가할 때는 실제 데이터로 확인한다.

LABELED_DIR = os.path.join("data", "labeled")
AUG_PREFIX = "state_aug_"

INTERVAL = 2.002          # 센서가 한 줄을 보내는 간격(초)
START_DATE = "2026-09-01"  # 만든 데이터에 붙일 첫 날짜 (실제 데이터와 겹치지 않게)

GASES = ["eco2_raw", "ech2o_raw", "tvoc_raw", "pm25_raw", "pm10_raw"]
NOISE_COLUMNS = ["eco2_raw", "tvoc_raw", "pm25_raw", "pm10_raw",
                 "temperature", "humidity"]

# 실제 데이터(11/13, 11/17)의 프린터 가동 구간에 맞춰 본 값
#   - 가동을 시작하면 eCO2가 평시보다 150~400, TVOC는 그 0.12~0.2배만큼 오른다.
#     (실제로는 10분 남짓 만에 130~340 올랐다.)
#   - 처음 15~40분이 가장 많이 나오고, 그 뒤에는 그 40~80% 수준으로 이어진다고 가정한다.
#   - 오르는 빠르기(시간 상수)는 6~16분, 끝난 뒤 내려가는 빠르기는 20~50분
#   - 습도는 0~3%, 온도는 0~0.4도 정도 함께 오른다.
PRINT_ECO2 = (150, 400)
PRINT_TVOC_RATIO = (0.12, 0.20)
PRINT_KEEP = (0.4, 0.8)
PRINT_BURST = (15, 40)
PRINT_RISE = (6, 16)
PRINT_DECAY = (20, 50)

# 11/17 데이터에는 30~45분마다 습도가 4~6% 갑자기 오르고 TVOC만 조금
# 따라 오르는 일이 있었다(가습기 등으로 추정). 프린터와 헷갈리기 쉬운
# 사례라서 평시(idle) 데이터에 일부러 넣는다.
HUMID_GAP = (28, 45)


# =========================
# 2. 실제 데이터 읽기
# =========================

def decode_hex(text):
    whole, tenth = text.split()
    return int(whole, 16) + int(tenth, 16) / 10


def encode_hex(value):
    value = round(float(value), 1)
    whole = int(value)
    tenth = int(round((value - whole) * 10))
    return f"{whole:02x} {tenth:02x}"


def load_real(paths):

    sessions = []

    for path in paths:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["timestamp"])
        df["temperature"] = df["temperature_hex"].map(decode_hex)
        df["humidity"] = df["humidity_hex"].map(decode_hex)
        name = os.path.splitext(os.path.basename(path))[0].replace("state_", "")
        sessions.append((name, df))

    return sessions


# 평시 구간에서 "5분 중앙값과의 차이"를 잡음으로 모은다.
# 여러 센서의 잡음을 같은 행끼리 묶어서 가져오므로 센서끼리의 관계가 유지된다.
def collect_noise(sessions):

    pieces = []

    for _, df in sessions:
        values = df[NOISE_COLUMNS].astype(float)
        trend = values.rolling(151, center=True, min_periods=30).median()
        noise = (values - trend)[df["label"] == 0].dropna()

        if len(noise) > 300:
            pieces.append(noise.to_numpy())

    return pieces


def sample_noise(pieces, length, rng, block=150):

    out = []
    total = 0

    while total < length:
        piece = pieces[rng.integers(len(pieces))]
        start = rng.integers(0, len(piece) - block)
        out.append(piece[start:start + block])
        total += block

    return np.concatenate(out)[:length]


# =========================
# 3. 저장 형식 만들기 (m701_csv.py와 같은 열 + state, label)
# =========================

def to_frame(times, values, printing):

    eco2 = np.clip(np.round(values["eco2_raw"]), 400, 65535).astype(int)
    ech2o = np.clip(np.round(values["ech2o_raw"]), 0, 65535).astype(int)
    tvoc = np.clip(np.round(values["tvoc_raw"]), 0, 65535).astype(int)
    pm25 = np.clip(np.round(values["pm25_raw"]), 0, 65535).astype(int)
    pm10 = np.maximum(np.clip(np.round(values["pm10_raw"]), 0, 65535), pm25).astype(int)
    temp = np.clip(values["temperature"], 0, 60)
    hum = np.clip(values["humidity"], 1, 99)

    temp_hex = [encode_hex(v) for v in temp]
    hum_hex = [encode_hex(v) for v in hum]

    raw_hex = []

    for i in range(len(times)):
        body = [0x3C, 0x02]
        for number in (eco2[i], ech2o[i], tvoc[i], pm25[i], pm10[i]):
            body += [number >> 8, number & 0xFF]
        body += [int(x, 16) for x in temp_hex[i].split()]
        body += [int(x, 16) for x in hum_hex[i].split()]
        body.append(sum(body) & 0xFF)
        raw_hex.append(" ".join(f"{b:02x}" for b in body))

    printing = np.asarray(printing, dtype=bool)

    return pd.DataFrame({
        "timestamp": [t.isoformat(timespec="milliseconds") for t in times],
        "state": np.where(printing, "printing", "idle"),
        "label": printing.astype(int),
        "raw_hex": raw_hex,
        "eco2_raw": eco2,
        "ech2o_raw": ech2o,
        "tvoc_raw": tvoc,
        "pm25_raw": pm25,
        "pm10_raw": pm10,
        "temperature_hex": temp_hex,
        "humidity_hex": hum_hex,
    })


# =========================
# 4. 방법 1: 실제 세션 변형
# =========================

def make_variant(df, day, rng):

    # 날짜만 바꾸고 하루 중 시각은 그대로 둔다.
    shift = pd.Timestamp(day).date() - df["time"].iloc[0].date()
    times = list(df["time"] + pd.Timedelta(days=shift.days))

    values = {}
    noise_size = {"eco2_raw": 3, "tvoc_raw": 0.5, "pm25_raw": 0,
                  "pm10_raw": 0, "temperature": 0.05, "humidity": 0.1}
    offset = {"eco2_raw": rng.uniform(-20, 20), "tvoc_raw": rng.uniform(-3, 3),
              "pm25_raw": rng.integers(0, 2), "pm10_raw": 0,
              "temperature": rng.uniform(-2, 2), "humidity": rng.uniform(-5, 5)}
    offset["pm10_raw"] = offset["pm25_raw"]
    gain = rng.uniform(0.8, 1.4)  # 가동 때 오르는 크기를 키우거나 줄인다

    for column in NOISE_COLUMNS:
        series = df[column].astype(float)
        base = series.rolling(1800, min_periods=1).min()  # 1시간 최저값을 기준으로
        changed = base + (series - base) * (gain if column in ("eco2_raw", "tvoc_raw") else 1)
        values[column] = (changed + offset[column]
                          + rng.normal(0, noise_size[column], len(df))).to_numpy()

    values["ech2o_raw"] = df["ech2o_raw"].to_numpy()

    return to_frame(times, values, df["label"] == 1)


# =========================
# 5. 방법 2: 시뮬레이션
# =========================

# 1차 반응: 목표값을 향해 tau(분)의 빠르기로 따라간다.
def follow(target, tau, dt):

    x = np.zeros(len(target))

    for i in range(1, len(target)):
        x[i] = x[i - 1] + (target[i] - x[i - 1]) * min(1.0, dt / tau[i])

    return x


def random_jobs(minutes, rng):

    jobs = []
    t = rng.uniform(10, 90)

    while t < minutes - 15:
        # 짧은 출력(실제 데이터와 비슷한 10~30분)과 긴 출력(30분~3시간)을 섞는다.
        if rng.random() < 0.4:
            length = rng.uniform(10, 30)
        else:
            length = rng.uniform(30, 180)
        end = min(t + length, minutes - 1)
        jobs.append((t, end))
        t = end + rng.uniform(20, 120)

    # 프린터를 한 번도 안 쓴 날도 있도록 가끔 비운다.
    if rng.random() < 0.1:
        jobs = []

    return jobs


def simulate(day, noise, rng):

    hours = rng.uniform(4, 8)
    n = int(hours * 3600 / INTERVAL)
    dt = INTERVAL / 60
    minute = np.arange(n) * dt

    start = pd.Timestamp(f"{day} 08:00", tz="Asia/Seoul") + pd.Timedelta(
        minutes=float(rng.uniform(20, 120)), milliseconds=int(rng.integers(0, 1000))
    )
    times = list(start + pd.to_timedelta(minute * 60, unit="s"))

    # (1) 평시 기준값 + 천천히 변하는 흐름
    def drift(amplitude):
        total = np.zeros(n)
        for _ in range(3):
            period = rng.uniform(40, 240)
            total += amplitude * rng.uniform(0.3, 1) * np.sin(
                2 * np.pi * minute / period + rng.uniform(0, 2 * np.pi))
        return total / 2

    eco2 = rng.uniform(460, 510) + drift(8)
    tvoc = rng.uniform(11, 21) + drift(2) - rng.uniform(0, 1.5) * minute / 60
    pm = rng.choice([2, 2, 2, 3, 4]) + np.zeros(n)
    temp = rng.uniform(19, 26) + rng.uniform(-0.1, 0.6) * minute / 60 + drift(0.2)
    hum = rng.uniform(33, 45) + drift(1.0) - rng.uniform(-0.5, 1.0) * minute / 60

    # (2) 프린터 가동
    jobs = random_jobs(minute[-1], rng)
    printing = np.zeros(n, dtype=bool)
    target_eco2 = np.zeros(n)
    target_tvoc = np.zeros(n)
    target_hum = np.zeros(n)
    target_temp = np.zeros(n)

    for begin, end in jobs:
        inside = (minute >= begin) & (minute <= end)
        printing |= inside
        keep = rng.uniform(*PRINT_KEEP)
        burst = rng.uniform(*PRINT_BURST)
        level = keep + (1 - keep) * np.exp(-(minute[inside] - begin) / burst)
        a = rng.uniform(*PRINT_ECO2) * level
        target_eco2[inside] = a
        target_tvoc[inside] = a * rng.uniform(*PRINT_TVOC_RATIO)
        target_hum[inside] = rng.uniform(0, 3)
        target_temp[inside] = rng.uniform(0, 0.4)

    rise = rng.uniform(*PRINT_RISE)
    decay = rng.uniform(*PRINT_DECAY)
    tau = np.where(printing, rise, decay)
    tau_tvoc = np.where(printing, rise, decay * rng.uniform(0.5, 0.9))

    eco2 += follow(target_eco2, tau, dt)
    tvoc += follow(target_tvoc, tau_tvoc, dt)
    hum += follow(target_hum, tau, dt)
    temp += follow(target_temp, tau, dt)

    # (3) 프린터와 헷갈리는 습도 상승 (평시 라벨)
    if rng.random() < 0.6:
        humid_on = np.zeros(n)
        t = rng.uniform(0, 40)
        while t < minute[-1]:
            length = rng.uniform(4, 8)
            humid_on[(minute >= t) & (minute < t + length)] = rng.uniform(4, 7)
            t += rng.uniform(*HUMID_GAP)
        humid = follow(humid_on, np.where(humid_on > 0, 3, rng.uniform(15, 25)), dt)
        hum += humid
        temp -= 0.05 * humid
        tvoc += follow(humid * rng.uniform(1.2, 2.0), np.full(n, 6.0), dt)

    # (4) 실제 데이터에서 가져온 잡음
    sampled = sample_noise(noise, n, rng)
    values = {
        "eco2_raw": eco2 + sampled[:, 0],
        "ech2o_raw": np.full(n, 10.0),
        "tvoc_raw": tvoc + sampled[:, 1],
        "pm25_raw": pm + sampled[:, 2],
        "pm10_raw": pm + sampled[:, 3],
        "temperature": temp + sampled[:, 4],
        "humidity": hum + sampled[:, 5],
    }

    return to_frame(times, values, printing), jobs, times[0]


# =========================
# 6. 실행
# =========================

def next_weekdays(count):
    days = pd.bdate_range(START_DATE, periods=count)
    return [d.strftime("%Y-%m-%d") for d in days]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+",
                        help="참고할 라벨링 파일 (기본: data/labeled/state_*.csv 중 실제 데이터)")
    parser.add_argument("--variants", type=int, default=3,
                        help="실제 세션 하나당 만들 변형 개수")
    parser.add_argument("--sim", type=int, default=30,
                        help="시뮬레이션으로 만들 세션 개수")
    parser.add_argument("--seed", type=int, default=42,
                        help="난수 시드 (같으면 같은 데이터가 나온다)")
    parser.add_argument("--output", default=LABELED_DIR,
                        help="저장할 폴더")
    args = parser.parse_args()

    inputs = args.input or sorted(
        path for path in glob.glob(os.path.join(LABELED_DIR, "state_*.csv"))
        if not os.path.basename(path).startswith(AUG_PREFIX)
    )

    if not inputs:
        print("참고할 라벨링 데이터가 없습니다. 먼저 m701_time_label.py 나 "
              "m701_state_logger.py 로 라벨링 데이터를 만드세요.")
        return

    rng = np.random.default_rng(args.seed)
    sessions = load_real(inputs)
    noise = collect_noise(sessions)

    if not noise:
        print("평시 구간이 너무 짧아 잡음을 잴 수 없습니다. 더 긴 데이터를 넣으세요.")
        return

    print("참고한 실제 데이터:")
    for name, df in sessions:
        print(f"  {name}: {len(df)}행 (가동 {int(df['label'].sum())}행)")

    os.makedirs(args.output, exist_ok=True)
    days = next_weekdays(len(sessions) * args.variants + args.sim)
    day_index = 0
    summary = []
    jobs_log = []

    # 방법 1
    for name, df in sessions:
        for k in range(1, args.variants + 1):
            out = make_variant(df, days[day_index], rng)
            day_index += 1
            path = os.path.join(args.output, f"{AUG_PREFIX}real_{name}_v{k}.csv")
            out.to_csv(path, index=False)
            summary.append((path, out))

    # 방법 2
    for k in range(1, args.sim + 1):
        out, jobs, start = simulate(days[day_index], noise, rng)
        day_index += 1
        path = os.path.join(args.output, f"{AUG_PREFIX}sim_{k:03d}.csv")
        out.to_csv(path, index=False)
        summary.append((path, out))

        for begin, end in jobs:
            jobs_log.append({
                "file": os.path.basename(path),
                "start": (start + pd.Timedelta(minutes=begin)).strftime("%Y-%m-%d %H:%M:%S"),
                "end": (start + pd.Timedelta(minutes=end)).strftime("%Y-%m-%d %H:%M:%S"),
            })

    jobs_file = os.path.join(args.output, "aug_print_jobs.csv")
    pd.DataFrame(jobs_log, columns=["file", "start", "end"]).to_csv(jobs_file, index=False)

    rows = sum(len(out) for _, out in summary)
    printing = sum(int(out["label"].sum()) for _, out in summary)
    print(f"\n만든 파일: {len(summary)}개 ({rows}행, 약 {rows * INTERVAL / 3600:.0f}시간)")
    print(f"평시(0): {rows - printing}행 / 가동(1): {printing}행 "
          f"({printing / rows * 100:.0f}%)")
    print(f"저장 폴더: {args.output}")
    print(f"시뮬레이션 가동 기록: {jobs_file}")
    print("\n다음 단계: python m701_state_clean.py --all")


if __name__ == "__main__":
    main()
