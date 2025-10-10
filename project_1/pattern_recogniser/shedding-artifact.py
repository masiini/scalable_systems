import time, random, threading
import hashlib, struct
from datetime import timedelta
from collections import defaultdict, deque
from prometheus_client import Gauge, Counter, Histogram
import pika

admit_p_g = Gauge('admission_probability', 'Global keep probability [0..1]')
p95_latency_g = Gauge('controller_p95_latency_seconds', 'Rolling p95 used by shedding controller')
target_latency_g = Gauge('controller_target_p95_seconds', 'Target p95 (baseline * pct)')
shed_level_g  = Gauge('shed_level', 'Shedding level (0=off,1,2)')
shed_cause_g  = Gauge('shed_cause', '0=none,1=latency,2=depth,3=both')
events_fwd    = Counter('events_forwarded_total', 'Events forwarded to CEP')
events_drop   = Counter('events_dropped_total', 'Events dropped by shedding', ['reason'])
matches_total = Counter('matches_total', 'Pattern matches emitted')
queue_depth_g = Gauge('rabbitmq_queue_depth', 'Queue depth (messages)')

ADMIT_ATTACK = 0.6
ADMIT_RELEASE = 0.05
ADMIT_MIN = 0.01
PREFETCH = 100
P95_CACHE_MS = 250

def _stable_keep(key_bytes: bytes, p: float) -> bool:
    if p >= 0.999: return True
    if p <= 0.0:   return False
    h = hashlib.blake2b(key_bytes, digest_size=8).digest()
    u = struct.unpack('!Q', h)[0] / 2**64
    return u < p

def sampling_key(evt) -> bytes:
    typ = (evt.get("rideable_type") or "").strip().lower()
    s_start = str(evt.get("start_station_id") or "").strip()
    s_end   = str(evt.get("end_station_id") or "").strip()
    t_end = evt.get("ended_at")
    if hasattr(t_end, "timestamp"):
        bucket = int(t_end.timestamp() // 10)
    else:
        bucket = 0
    key_tuple = (typ, s_start, s_end, bucket)
    return "|".join(map(str, key_tuple)).encode()

TARGET = {"6822.09","5779.09", "5905.12"}

# queue thresholds
HIGH = 10000
LOW  = 3000

# # latency thresholds (seconds)
# P95_L1 = 0.1   # 100 ms → go to level 1
# P95_L2 = 0.3   # 200 ms → go to level 2

TAU1 = 0.7
TAU2 = 2.5
SAMPLE_MED = 0.3

class LatencyWindow:
    def __init__(self, max_samples=5000, window_sec=30):
        self.max = max_samples
        self.win = window_sec
        self.samples = deque()  # (ts, value_s)

    def add(self, value_s: float):
        now = time.time()
        self.samples.append((now, value_s))
        self._gc(now)

    def p95(self):
        now = time.time(); self._gc(now)
        vals = [v for _, v in self.samples]
        if not vals: return None
        vals.sort()
        idx = int(0.95 * (len(vals) - 1))
        return vals[idx]

    def _gc(self, now: float):
        cutoff = now - self.win
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while len(self.samples) > self.max:
            self.samples.popleft()

# global instance
latency_window = LatencyWindow()

class SheddingState:
    def __init__(self, conn_params: pika.ConnectionParameters, queue_name: str,
                 high:int=HIGH, low:int=LOW, poll_period:float=0.5,
                 target_p95: float = 2.0):
        self.q = queue_name
        self.high, self.low = high, low
        self.level = 0
        self._depth = 0
        self._poll_period = poll_period
        self._stop = False

        self.target_p95 = target_p95
        self.admit_p = 1.0
        self._last_p95 = None
        self._last_p95_ts = 0.0

        self._conn_params = conn_params
        t = threading.Thread(target=self._poller, daemon=True)
        t.start()

    def _p95_cached(self):
        now = time.monotonic()
        if (now - self._last_p95_ts) * 1000.0 < P95_CACHE_MS:
            return self._last_p95
        v = latency_window.p95()
        self._last_p95 = v
        self._last_p95_ts = now
        return v

    def _poller(self):
        conn = pika.BlockingConnection(self._conn_params)
        ch = conn.channel()
        try:
            while not self._stop:
                try:
                    m = ch.queue_declare(queue=self.q, passive=True)
                    self._depth = m.method.message_count
                    queue_depth_g.set(self._depth)
                except Exception:
                    pass
                time.sleep(self._poll_period)
        finally:
            try: ch.close()
            except Exception: pass
            try: conn.close()
            except Exception: pass

    def stop(self):
        self._stop = True

    def update(self) -> int:
        depth = self._depth
        p95   = self._p95_cached()
        depth_level = 2 if depth >= self.high else (1 if depth >= self.low else 0)

        # Latency-based level
        lat_level = 0
        if p95 is not None:
            p95_latency_g.set(p95); target_latency_g.set(self.target_p95)
            if self.target_p95 is not None:
                target_latency_g.set(self.target_p95)
                # hysteresis band to avoid flapping
                hi = 1.02 * self.target_p95
                lo = 0.95 * self.target_p95
                if   p95 >= hi:          lat_level = 2
                elif p95 >= self.target_p95: lat_level = 1
                elif p95 <= lo:          lat_level = 0
                else:                    lat_level = 1
            else:
                # fixed steps if no target provided
                P95_L1 = 0.10  # 100ms
                P95_L2 = 0.30  # 300ms
                lat_level = 2 if p95 >= P95_L2 else (1 if p95 >= P95_L1 else 0)

        # Aggressive admission
        admit = self.admit_p
        if p95 is not None and self.target_p95 is not None:
            if p95 > self.target_p95:
                ratio = min(5.0, p95 / max(1e-6, self.target_p95))
                desired = max(ADMIT_MIN, 1.0 / (ratio ** 2))
                admit = min(admit, desired) # fast attack
            elif p95 < 0.95 * self.target_p95:
                admit = min(1.0, admit + ADMIT_RELEASE) # slow release

        # Depth safety caps
        if depth >= HIGH:
            admit = min(admit, 0.05)      # at most 5% through
        elif depth >= LOW:
            admit = min(admit, 0.30)      # at most 30% through

        # Persist & export
        if abs(admit - self.admit_p) > 1e-6:
            self.admit_p = admit
            admit_p_g.set(admit)

        # Choose stricter
        if   depth_level and lat_level: cause = 3
        elif depth_level:               cause = 2
        elif lat_level:                 cause = 1
        else:                           cause = 0

        self.level = max(depth_level, lat_level)
        shed_level_g.set(self.level)
        shed_cause_g.set(cause)
        return self.level
    
    def budget_admit(self, key_bytes: bytes) -> bool:
        return _stable_keep(key_bytes, self.admit_p)

class RecentEnds:
    def __init__(self, ttl_sec=1800):
        self.ttl = ttl_sec
        self.deq = deque()  # (ts, station_id)
        self.set = set()

    def add(self, sid):
        if not sid: return
        now = time.time()
        s = str(sid).strip()
        self.deq.append((now, s))
        self.set.add(s)
        cutoff = now - self.ttl
        while self.deq and self.deq[0][0] < cutoff:
            _, old = self.deq.popleft()
            if all(x != old for _, x in self.deq):
                self.set.discard(old)

    def seen(self, sid) -> bool:
        return str(sid or "").strip() in self.set

class TypeChainTracker:
    def __init__(self, max_gap: timedelta = timedelta(minutes=45)):
        self.max_gap = max_gap
        # tails[type][station] -> deque[(last_end_time, length, first_start_time)]
        self.tails = defaultdict(lambda: defaultdict(deque)) # type: ignore

    def on_event(self, evt):
        typ = (evt.get("rideable_type") or "").strip().lower()
        s_start = str(evt.get("start_station_id") or "").strip()
        s_end   = str(evt.get("end_station_id") or "").strip()
        t_start = evt["started_at"]
        t_end   = evt["ended_at"]

        dq = self.tails[typ][s_start]
        cutoff = t_start - self.max_gap
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        best = None  # (idx, last_end, length, first_start)
        for i, (last_end, length, first_start) in enumerate(dq):
            if last_end <= t_start and (best is None or last_end > best[1]):
                best = (i, last_end, length, first_start)

        if best is None:
            prev_len, first_start, length = 0, t_start, 1
        else:
            _, last_end, length0, first_start = best
            prev_len = length0
            dq.remove((last_end, length0, first_start))
            length = length0 + 1

        self.tails[typ][s_end].append((t_end, length, first_start))
        return prev_len, first_start

def utility_score_type(evt, chains: TypeChainTracker, recents: RecentEnds, targets: set[str]) -> float:
    """
    Prioritize:
      - longer chains (prev_len)
      - closeness to 1h window (from a[1].start to current ended_at)
      - ending at target stations
      - likely handoff (start station was a recent end)
    """
    prev_len, first_start = chains.on_event(evt)

    near = 0.0
    if first_start:
        elapsed = (evt["ended_at"] - first_start).total_seconds()
        near = max(0.0, min(1.0, elapsed / 3600.0))

    tail = 1.0 if str(evt.get("end_station_id") or "").strip() in targets else 0.0
    hand = 1.0 if recents.seen(evt.get("start_station_id")) else 0.0

    # Weights emphasize nearing 1h window & tails at target stations
    wL, wN, wT, wH = 0.6, 2.2, 3.2, 0.5
    return wL*prev_len + wN*near + wT*tail + wH*hand

def should_forward(level: int, score: float) -> bool:
    if level == 0:
        return True
    if level == 1:
        return score >= TAU1
    # level >= 2
    return score >= TAU2
