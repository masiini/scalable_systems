from prometheus_client import Gauge, Counter
import collections, time, random

shed_level_g  = Gauge('shed_level', 'Shedding level (0=off,1,2)')
events_fwd    = Counter('events_forwarded_total', 'Events forwarded to CEP')
events_drop   = Counter('events_dropped_total', 'Events dropped by shedding', ['reason'])
matches_total = Counter('matches_total', 'Pattern matches emitted')
queue_depth_g = Gauge('rabbitmq_queue_depth', 'Queue depth (messages)')

class SheddingState:
    def __init__(self, channel, queue_name, high=50000, low=20000):
        self.ch, self.q = channel, queue_name
        self.high, self.low = high, low
        self.level = 0

    def depth(self):
        m = self.ch.queue_declare(queue=self.q, passive=True)
        d = m.method.message_count
        queue_depth_g.set(d)
        return d

    def update(self):
        d = self.depth()
        if d >= self.high:
            self.level = 2
        elif d <= self.low:
            self.level = 0
        else:
            self.level = max(self.level, 1)
        shed_level_g.set(self.level)
        return self.level

class ChainTracker:
    def __init__(self):  # best-effort per bike_id
        self.length = collections.Counter()   # bike_id -> chain length so far
        self.first_end = {}                   # bike_id -> first a ended_at

    def on_event(self, evt):
        bid = evt.get("bike_id")
        if not bid: return
        if bid not in self.first_end:
            self.first_end[bid] = evt["ended_at"]
            self.length[bid] = 1
        else:
            self.length[bid] += 1

    def get_len(self, bid):   return self.length.get(bid, 0)
    def get_start(self, bid): return self.first_end.get(bid)

class RecentEnds:
    def __init__(self, ttl_sec=1800):
        self.ttl = ttl_sec
        self.deq = collections.deque()  # (ts, station_id)
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
            # remove old if no duplicates remain
            if all(x != old for _, x in self.deq):
                self.set.discard(old)

    def seen(self, sid): return str(sid).strip() in self.set

TARGET = {"6822.09", "5779.09", "5905.12"}

def utility_score(evt, chains: ChainTracker, recents: RecentEnds):
    bid = evt.get("bike_id")
    L   = chains.get_len(bid) if bid else 0
    s0  = chains.get_start(bid)
    near = 0.0
    if s0:
        elapsed = (evt["ended_at"] - s0).total_seconds()
        near = max(0.0, min(1.0, elapsed / 3600.0))  # closer to 1h → nearer completion
    tail = 1.0 if evt.get("end_station_id") in TARGET else 0.0
    hand = 1.0 if recents.seen(evt.get("start_station_id")) else 0.0
    # weights (tune during evaluation):
    wL, wN, wT, wH = 1.0, 2.0, 3.0, 0.5
    return wL*L + wN*near + wT*tail + wH*hand

TAU1 = 0.5      # Level-1 keep threshold
TAU2 = 2.0      # Level-2 keep threshold
SAMPLE_MED = 0.5

def should_forward(level, score):
    if level == 0:
        return True
    if level == 1:
        return score >= TAU1
    # level >= 2
    if score >= TAU2:
        return True
    if score < TAU1:
        return False
    return (random.random() < SAMPLE_MED)
