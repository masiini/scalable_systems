from msgspec.structs import asdict
from msgspec import Struct
from msgspec.json import encode

class Ride(Struct):
    ride_id: str
    rideable_type: str
    started_at: str
    ended_at: str
    start_station_id: str
    end_station_id: str

    def to_msg(self):
        return encode(self)