import csv, random, string
from datetime import datetime, timedelta

PLANTED_PER_TYPE = 30
OUT = f"citibike-test-{PLANTED_PER_TYPE}.csv"
TOTAL_ROWS = 10_000
CHAIN_LEN = 3
HOT = {"6822.09", "5779.09", "5905.12"}

OTHER = ["5492.05","5351.03","6230.04","6602.03","5788.12","6743.06","6004.06","6912.01","5593.04","5905.12","6824.07"]
STATIONS = sorted(set(OTHER) | HOT)

def station_name(sid): return f"Station {sid}"
def fake_lat(sid):     return 40.7 + (hash(sid) % 100) * 1e-3
def fake_lng(sid):     return -74.0 + (hash(sid[::-1]) % 100) * 1e-3

HEADERS = [
    "ride_id","rideable_type","started_at","ended_at",
    "start_station_name","start_station_id",
    "end_station_name","end_station_id",
    "start_lat","start_lng","end_lat","end_lng","member_casual"
]

def rid():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=16))

def mk_trip(rideable_type, start_ts, dur_s, start_sid, end_sid, member=True):
    started_at = start_ts
    ended_at   = started_at + timedelta(seconds=dur_s)
    return {
        "ride_id": rid(),
        "rideable_type": rideable_type,
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "ended_at":   ended_at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "start_station_name": station_name(start_sid),
        "start_station_id": start_sid,
        "end_station_name": station_name(end_sid),
        "end_station_id": end_sid,
        "start_lat": f"{fake_lat(start_sid):.6f}",
        "start_lng": f"{fake_lng(start_sid):.6f}",
        "end_lat":   f"{fake_lat(end_sid):.6f}",
        "end_lng":   f"{fake_lng(end_sid):.6f}",
        "member_casual": "member" if member else "casual",
    }

def plant_chain(now, bike_type, chain_len=3):
    # build a contiguous path over non-hot stations, then end 'b' at HOT
    # ensure we can finish within 1h
    remaining = 50 * 60  # reserve some time for b
    a_trips = []
    cur_start = random.choice(STATIONS)
    t = now

    for _ in range(chain_len):
        # trip duration 2–8 minutes
        dur = random.randint(120, 480)
        remaining -= dur
        # pick an end different from start to avoid too many trivial self-loops
        end = random.choice(STATIONS)
        if end == cur_start:
            end = random.choice(STATIONS)
        a_trips.append(mk_trip(bike_type, t, dur, cur_start, end))
        t = t + timedelta(seconds=dur + random.randint(10, 60))  # short turnover
        cur_start = end

    # final 'b' — same type (classifier), contiguity not required to a[last],
    # only tail condition is required in the pattern (and within 1h overall).
    # But to be closer to "hot paths", we’ll keep the station flow by starting b anywhere
    # and ending at a HOT station.
    dur_b = random.randint(120, 480)
    b_end = random.choice(sorted(HOT))
    b_start = random.choice(STATIONS)

    # keep whole sequence within 1 hour of a[1].started_at
    a1_start = datetime.strptime(a_trips[0]["started_at"], "%Y-%m-%d %H:%M:%S.%f")
    if (t + timedelta(seconds=dur_b)) - a1_start > timedelta(hours=1):
        # pull back 't' to satisfy the window
        t = a1_start + timedelta(minutes=55)

    b_trip = mk_trip(bike_type, t, dur_b, b_start, b_end)
    return a_trips + [b_trip]

def generate():
    random.seed(42)
    rows = []

    # base timeline start
    now = datetime(2020, 12, 1, 7, 0, 0)

    # background noise
    bg_count = TOTAL_ROWS - (PLANTED_PER_TYPE * (CHAIN_LEN + 1) * 2)
    for _ in range(max(0, bg_count)):
        typ = random.choice(["classic_bike", "electric_bike"])
        dur = random.randint(60, 1800)
        s = random.choice(STATIONS)
        e = random.choice(STATIONS)
        rows.append(mk_trip(typ, now, dur, s, e, member=bool(random.getrandbits(1))))
        now += timedelta(seconds=random.randint(5, 20))

    # planted matches
    for typ in ["classic_bike", "electric_bike"]:
        for _ in range(PLANTED_PER_TYPE):
            seq = plant_chain(now, typ, chain_len=CHAIN_LEN)
            rows.extend(seq)
            # space chains a bit
            now += timedelta(minutes=random.randint(3, 8))

    # sort by start time
    rows.sort(key=lambda r: r["started_at"])

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT}")
    print(f"Planted matches per type: {PLANTED_PER_TYPE}, chain_len={CHAIN_LEN}, HOT={sorted(HOT)}")

if __name__ == "__main__":
    generate()