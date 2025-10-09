import json
import pika
import logging, os, sys, time

from datetime import datetime, timedelta

import threading
from msgspec.json import Decoder
from msgspec import Struct
from msgspec.structs import asdict

from ride import Ride, parse_datetime_hook
from shedding import (
    SheddingState, TypeChainTracker, RecentEnds,
    utility_score_type, should_forward,
    events_drop, events_fwd, TARGET,
    latency_window
)

from CEP import CEP
from condition.Condition import Variable, SimpleCondition, BinaryCondition
from condition.CompositeCondition import AndCondition
from condition.KCCondition import KCIndexCondition, KCValueCondition
from misc.ConsumptionPolicy import ConsumptionPolicy, SelectionStrategies
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
from base.DataFormatter import EventTypeClassifier, DataFormatter
from stream.Stream import Stream
from stream.FileStream import FileOutputStream

from prometheus_client import REGISTRY, start_http_server, Counter, Histogram
from prometheus_client.exposition import write_to_textfile, generate_latest

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
logger = logging.getLogger(__name__)
logger.addHandler(sh)

def dump_metrics():
    try:
        write_to_textfile("/opt/app/metrics.txt", REGISTRY)
    except Exception:
        logger.exception("Failed to write metrics snapshot")

QUEUE_NAME = os.getenv('QUEUE_NAME')
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME')
ROUTE_KEY = os.getenv('ROUTE_KEY')
BROKER_HOST = os.getenv('BROKER_HOST')
BROKER_USER = os.getenv('BROKER_USER')
BROKER_PASS = os.getenv('BROKER_PASS')

BASE_P95 = 0.012340246001258492 # from a no-shed test run
P95_PERCENTAGE = 0.10 # 10%, 30%, 50%, 70%, 90%
TARGET_P95 = BASE_P95 * P95_PERCENTAGE

PREFETCH = 1000
KLEENE_MIN = 2
KLEENE_MAX = 5

latency_h = Histogram('pattern_recognition_latency', 'Latency for pattern recognition')
messages_processed = Counter('pattern_recognition_messages_processed', 'Total number of messages processed for pattern recognition')

SNAP_PATH = os.getenv("SNAP_PATH", "/opt/app/snap.json")
SNAP_PERIOD = float(os.getenv("SNAP_PERIOD", "2.0"))
SHED_TARGET_PCT = os.getenv("SHED_TARGET_PCT")

def snapshot(stop_event: threading.Event):
    while not stop_event.is_set():
        p95 = latency_window.p95()
        snap = {
            "time": time.time(),
            "p95": p95,
        }
        try:
            with open(SNAP_PATH, "w") as f:
                json.dump(snap, f)
        except Exception:
            pass
        stop_event.wait(SNAP_PERIOD)

class CitibikeByRideEventTypeClassifier(EventTypeClassifier):
    def get_event_type(self, event_payload: dict):
        return event_payload['rideable_type']

class CitibikeDataFormatter(DataFormatter):
    def __init__(self, event_type_classifier: EventTypeClassifier = CitibikeByRideEventTypeClassifier()):
        super().__init__(event_type_classifier)
        self.decoder = Decoder(type=Ride, dec_hook=parse_datetime_hook)

    def parse_event(self, raw_data: str):
        evt = asdict(self.decoder.decode(raw_data))
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
        self.last_msg_ts = None
        self.timeout_seconds = 10
        self.monitor_stop = threading.Event()
        self.monitor_thread = None

        # shedding state
        self.shed = None
        self.chains = TypeChainTracker(max_gap=timedelta(minutes=45))
        self.recents = RecentEnds(ttl_sec=1800)

        self.snap_thread = None
        self.snap_stop = None

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
                self.shed = SheddingState(self.rabbitmq_conn_params, self.queue_name, target_p95=TARGET_P95)
                self.snap_stop = threading.Event()
                self.snap_thread = threading.Thread(target=snapshot, args=(self.snap_stop,), daemon=True)
                self.snap_thread.start()
                logger.info("Pattern node: Connected to RabbitMQ")
                return
            except pika.exceptions.AMQPConnectionError as e:
                logger.info(f"Pattern node: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                retries += 1
                time.sleep(self.retry_delay)
        logger.info("Pattern node: Failed to connect to RabbitMQ after several attempts.")


    def close_connections(self):
        try:
            if self.monitor_stop: self.monitor_stop.set()
        except Exception:
            pass
        try:
            if self.shed: self.shed.stop()
        except Exception:
            pass
        if self.queue_connection:
            try: self.channel.stop_consuming()
            except Exception: pass
            try: self.queue_connection.close()
            except Exception: pass

    @latency_h.time()
    def process_messages(self, ch, method, properties, body):
        self.last_msg_ts = time.monotonic()
        t0 = time.perf_counter()
        try:
            evt = asdict(self.data_formatter.decoder.decode(body))

            self.recents.add(evt.get("end_station_id"))

            level = self.shed.update()  # uses queue depth + local p95
            score = utility_score_type(evt, self.chains, self.recents, TARGET)

            if should_forward(level, score):
                self.input_stream.add_item(body.decode('utf-8'))
                events_fwd.inc()
            else:
                events_drop.labels(reason=f"level{level}").inc()

            messages_processed.inc()
            ch.basic_ack(delivery_tag=method.delivery_tag)
        finally:
            latency_window.add(time.perf_counter() - t0)

    def start_cep_engine(self):
        def run_engine():
            try:
                self.cep.run(self.input_stream, self.output_stream, self.data_formatter)
            except Exception:
                logger.exception("[ENGINE] error")
        t = threading.Thread(target=run_engine, daemon=True, name="CEPEngine")
        t.start()
        self.engine = t

    def start_timeout_monitor(self):
        def monitor():
            while not self.monitor_stop.is_set():
                time.sleep(1)
                if self.last_msg_ts is not None:
                    elapsed = time.monotonic() - self.last_msg_ts
                    if elapsed > self.timeout_seconds:
                        logger.info(f"No messages received for {self.timeout_seconds}s, stopping...")
                        self.channel.stop_consuming()
                        dump_metrics()
                        break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True, name="TimeoutMonitor")
        self.monitor_thread.start()

    def consume_messages(self):
        self.start_cep_engine()
        self.start_timeout_monitor()
        self.channel.basic_qos(prefetch_count=PREFETCH)
        self.channel.basic_consume(queue=self.queue_name,
                                   on_message_callback=self.process_messages,
                                   auto_ack=False)
        logger.info(f"Pattern node: Waiting for messages. Prefetch={PREFETCH}")
        self.channel.start_consuming()
        logger.info("Stopped consuming messages due to timeout")

def full_pattern(event_type: str, stations: set[str]):
    """
    PATTERN SEQ (BikeTrip+ a[], BikeTrip b)
    WHERE  a[i+1].bike = a[i].bike
        AND b.end in {7,8,9}
        AND a[last].bike = b.bike
        AND a[i+1].start = a[i].end
    WITHIN 1h
    """
    consumption_policy = ConsumptionPolicy(
        primary_selection_strategy=SelectionStrategies.MATCH_ANY,
        secondary_selection_strategy=SelectionStrategies.MATCH_SINGLE,
        single=["classic_bike", "electric_bike"]
    )

    return Pattern(
        SeqOperator(
            KleeneClosureOperator(
                PrimitiveEventStructure(event_type,"a"), 
                min_size=KLEENE_MIN, 
                max_size=KLEENE_MAX
            ),
            PrimitiveEventStructure(event_type,"b")
        ),
        AndCondition(
            # Already covered by event_type
            # # (1) a[i+1].bike = a[i].bike
            # KCIndexCondition(names={"a"}, getattr_func=lambda x: x["rideable_type"],
            #                  relation_op=lambda p,n: n==p, offset=1),
            # 
            # # (3) a[last].bike = b.bike
            # KCIndexCondition(names={"a","b"}, getattr_func=lambda x: x["rideable_type"],
            #                  relation_op=lambda la, lb: la==lb, first_index=-1, second_index=0, offset=None),

            # (4) a[i+1].start = a[i].end
            KCIndexCondition(
                names={"a"},
                getattr_func=lambda x: (x["start_station_id"], x["end_station_id"]),
                relation_op=lambda prev, nxt: nxt[0] == prev[1],
                offset=1
            ),

            # (2) b.end in {7,8,9}
            SimpleCondition(
                Variable("b", lambda x: x["end_station_id"]),
                relation_op=lambda eid, S=stations: eid in S
            ),
        ),
        timedelta(hours=1),
        consumption_policy=consumption_policy
    )

# for testing
def test(event_type: str, stations: set[str]):
    return Pattern(
        SeqOperator(PrimitiveEventStructure(event_type, "b")),
        SimpleCondition(Variable("b", lambda x: x["end_station_id"]),
                        relation_op=lambda eid, S=stations: eid in S),
        timedelta(hours=1)
    )

def build_patterns(stations=TARGET):
    return [full_pattern("classic_bike", stations), full_pattern("electric_bike", stations)]

def main():
    try:
        start_http_server(8080, addr="0.0.0.0")

        patterns = build_patterns(TARGET)
        cep = CEP(patterns)

        conn_params = pika.ConnectionParameters(
            BROKER_HOST, 5672, '/',
            pika.PlainCredentials(BROKER_USER, BROKER_PASS),
            heartbeat=30, blocked_connection_timeout=300
        )
        node = Cepper(QUEUE_NAME, conn_params, cep)
        node.connect()
        node.consume_messages()

        logger.info("Keeping metrics server up... Ctrl+C to exit.")
        threading.Event().wait()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    main()
