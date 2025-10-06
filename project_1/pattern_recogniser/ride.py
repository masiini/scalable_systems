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