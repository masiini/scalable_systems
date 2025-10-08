import collections
import pika
import logging, os, sys, time

from datetime import datetime, timedelta

import threading
from msgspec.json import Decoder
from msgspec import Struct
from msgspec.structs import asdict

from ride import Ride, parse_datetime_hook
from shedding import SheddingState, utility_score, TARGET, ChainTracker, RecentEnds, should_forward, events_drop, events_fwd

from CEP import CEP
from condition.Condition import Variable, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.KCCondition import KCIndexCondition, KCValueCondition
from misc.ConsumptionPolicy import ConsumptionPolicy, SelectionStrategies
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
from base.DataFormatter import EventTypeClassifier, DataFormatter
from stream.Stream import Stream
from stream.FileStream import FileOutputStream

from prometheus_client import start_http_server, Counter, Histogram

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        filename='cep.log',
        filemode='a',
        encoding='utf-8',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s","%Y-%m-%d %H:%M"))
    logger.addHandler(sh)

QUEUE_NAME = os.getenv('QUEUE_NAME')
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME')
ROUTE_KEY = os.getenv('ROUTE_KEY')
BROKER_HOST = os.getenv('BROKER_HOST')
BROKER_USER = os.getenv('BROKER_USER')
BROKER_PASS = os.getenv('BROKER_PASS')

latency_h = Histogram('pattern_recognition_latency', 'Latency for pattern recognition')
messages_processed = Counter('pattern_recognition_messages_processed', 'Total number of messages processed for pattern recognition')

class BikeLinker:
    def __init__(self, max_gap: timedelta = timedelta(minutes=45)):
        self.max_gap = max_gap
        self.lock = threading.Lock()
        # station_id -> deque[(ended_at, bike_key)]
        self.ends_by_station: collections.defaultdict[str, collections.deque[tuple[datetime, str]]] = collections.defaultdict(collections.deque)
        self._next = 1

    def _new_key(self) -> str:
        k = f"bk{self._next}"
        self._next += 1
        return k

    def assign_and_register(self, evt: dict) -> None:
        start_sid = str(evt["start_station_id"]).strip()
        end_sid   = str(evt["end_station_id"]).strip()
        t_start   = evt["started_at"]
        t_end     = evt["ended_at"]

        with self.lock:
            dq = self.ends_by_station[start_sid]
            cutoff = t_start - self.max_gap
            # drop too-old
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            # pick closest prior end <= start
            best_idx, best_tend, best_key = -1, None, None
            for i, (t_end_prev, key_prev) in enumerate(dq):
                if t_end_prev <= t_start and (best_tend is None or t_end_prev > best_tend):
                    best_idx, best_tend, best_key = i, t_end_prev, key_prev

            if best_key is None:
                best_key = self._new_key()
            else:
                if best_tend is not None:
                    dq.remove((best_tend, best_key))

            evt["bike_id"] = best_key
            # register this trip’s end at its end station
            self.ends_by_station[end_sid].append((t_end, best_key))

class CitibikeByRideEventTypeClassifier(EventTypeClassifier):
    def get_event_type(self, event_payload: dict):
        t = (event_payload.get("rideable_type") or "")
        if t == "classic_bike":  return "ClassicTrip"
        if t == "electric_bike": return "ElectricTrip"
        return "BikeTrip"

class CitibikeDataFormatter(DataFormatter):
    def __init__(self, event_type_classifier: EventTypeClassifier = CitibikeByRideEventTypeClassifier()):
        super().__init__(event_type_classifier)
        self.decoder = Decoder(type=Ride, dec_hook=parse_datetime_hook)
        self._linker = BikeLinker(max_gap=timedelta(minutes=45))

    def parse_event(self, raw_data: str):
        evt = asdict(self.decoder.decode(raw_data))
        for k in ("start_station_id", "end_station_id"):
            v = evt.get(k)
            evt[k] = "" if v is None else str(v).strip()
        # bike_id
        if not evt.get("bike_id"):
            self._linker.assign_and_register(evt)
        return evt

    def get_event_timestamp(self, event_payload: dict):
        return event_payload['ended_at']

class Cepper:
    def __init__(self, queue_name, conn_params, cep):
        self.queue_name = queue_name
        self.rabbitmq_conn_params = conn_params

        # Messaging connection & channel
        self.queue_connection = None
        self.channel = None
        
        # CEP
        logger.info(f'Initializing CEP')
        self.cep = cep
        self.input_stream = Stream()
        self.output_stream = FileOutputStream('/opt/app', 'patterns.txt', True)
        self.data_formatter = CitibikeDataFormatter()

        self.shed = None
        self.chains = ChainTracker()
        self.recents = RecentEnds(ttl_sec=1800)
        # reuse the same decoder for prefilter to avoid extra json lib:
        self.prefilter_decoder = self.data_formatter.decoder

        self.max_retries = 5
        self.retry_delay = 5

    def is_connected(self):
        return self.queue_connection is not None

    def connect(self):
        retries = 0
        while retries < self.max_retries:
            try:
                self.queue_connection = pika.BlockingConnection(self.rabbitmq_conn_params)
                self.channel = self.queue_connection.channel()
                self.channel.queue_declare(queue=self.queue_name, durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)
                self.channel.queue_bind(exchange=EXCHANGE_NAME, queue=self.queue_name, routing_key=ROUTE_KEY)
                self.shed = SheddingState(self.channel, self.queue_name, high=50000, low=20000)
                logger.info("Pattern node: Connected to RabbitMQ")
                return
            except pika.exceptions.AMQPConnectionError as e:
                logger.info(f"Pattern node: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                retries += 1
                time.sleep(self.retry_delay)
        logger.info("Pattern node: Failed to connect to RabbitMQ after several attempts.")


    def close_connections(self):
        if self.queue_connection:
            self.channel.stop_consuming()
            self.queue_connection.close()

    @latency_h.time()
    def process_messages(self, ch, method, properties, body):
        # quick decode to score shedding (same dec_hook; cheap)
        evt = asdict(self.prefilter_decoder.decode(body))
        for k in ("start_station_id","end_station_id"):
            v = evt.get(k); evt[k] = "" if v is None else str(v).strip()

        # update recents and chain tracker
        self.recents.add(evt.get("end_station_id"))
        self.chains.on_event(evt)

        level = self.shed.update()
        score = utility_score(evt, self.chains, self.recents)

        if should_forward(level, score):
            self.input_stream.add_item(body.decode('utf-8'))
            events_fwd.inc()
        else:
            events_drop.labels(reason=f"level{level}").inc()

        messages_processed.inc()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_cep_engine(self):
        def run_engine():
            try:
                self.cep.run(self.input_stream, self.output_stream, self.data_formatter)
            except Exception:
                logger.exception("[ENGINE] error")
        threading.Thread(target=run_engine, daemon=True, name="CEPEngine").start()

    def consume_messages(self):
        self.start_cep_engine()
        self.channel.basic_qos(prefetch_count=1000)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_messages, auto_ack=False)
        logger.info(f"Pattern node: Waiting for messages.")
        self.channel.start_consuming()

def full_pattern(event_type: str, stations: set[str]):
    """
    PATTERN SEQ (BikeTrip+ a[], BikeTrip b)
    WHERE  a[i+1].bike = a[i].bike
        AND b.end in {7,8,9}
        AND a[last].bike = b.bike
        AND a[i+1].start = a[i].end
    WITHIN 1h
    """
    return Pattern(
        SeqOperator(
            KleeneClosureOperator(PrimitiveEventStructure(event_type,"a"), min_size=2),
            PrimitiveEventStructure(event_type,"b")
        ),
        AndCondition(
            KCIndexCondition(names={"a"}, getattr_func=lambda x: x["bike_id"],
                             relation_op=lambda p,n: n==p, offset=1),
            KCIndexCondition(names={"a"}, getattr_func=lambda x: (x["start_station_id"], x["end_station_id"]),
                             relation_op=lambda p,n: n[0]==p[1], offset=1),
            KCIndexCondition(names={"a","b"}, getattr_func=lambda x: x["bike_id"],
                             relation_op=lambda la, lb: la==lb, first_index=-1, second_index=0, offset=None),
            SimpleCondition(Variable("b", lambda x: x["end_station_id"]),
                            relation_op=lambda end_id, S=stations: end_id in S),
        ),
        timedelta(hours=1)
    )

# for testing (loosen pattern)
def tail_probe(event_type: str, stations: set[str]):
    return Pattern(
        SeqOperator(PrimitiveEventStructure(event_type, "b")),
        SimpleCondition(Variable("b", lambda x: x["end_station_id"]),
                        relation_op=lambda eid, S=stations: eid in S),
        timedelta(hours=1)
    )

def build_patterns(stations=TARGET):
    return [tail_probe("ClassicTrip", stations), tail_probe("ElectricTrip", stations), tail_probe("BikeTrip", stations)]

def main():
    try:
        start_http_server(8080, addr="0.0.0.0")
        patterns = build_patterns()
        cep = CEP(patterns)
        conn_params = pika.ConnectionParameters(BROKER_HOST, 5672, '/', pika.PlainCredentials(BROKER_USER, BROKER_PASS))
        pattern_detector = Cepper(QUEUE_NAME, conn_params, cep)
        pattern_detector.connect()
        pattern_detector.consume_messages()


    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    main()
