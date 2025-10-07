from msgspec.json import decode
from msgspec import Struct

from datetime import datetime

class Ride(Struct):
    ride_id: str
    rideable_type: str
    started_at: datetime
    ended_at: datetime
    start_station_id: str
    end_station_id: str

def parse_datetime_hook(type_, obj):
    if type_ is datetime and isinstance(obj, str):
        try:
            return datetime.strptime(obj, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(obj, "%Y-%m-%d %H:%M:%S")
    return value