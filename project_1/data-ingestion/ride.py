import json

class Ride:
    def __init__(self, ride_id, rideable_type, 
        started_at, ended_at, start_station_id, end_station_id):

        self.ride_id = ride_id 
        self.rideable_type = rideable_type
        self.started_at = started_at
        self.ended_at = ended_at
        self.start_station_id = start_station_id
        self.end_station_id = end_station_id

    def to_dict(self):
        return {
            "ride_id": self.ride_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "start_station_id": self.start_station_id,
            "end_station_id": self.end_station_id,
        }

    def to_json(self):
        return json.dumps(self.to_dict())