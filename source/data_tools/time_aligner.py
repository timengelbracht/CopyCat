class TimeAligner:
    def __init__(self, aria_pair, sensor_pair):
        # compute delta to align sensor to aria timestamps
        aria_dev, aria_utc = aria_pair
        sens_dev, sens_utc = sensor_pair
        aria_offset  = aria_utc - aria_dev
        sens_offset  = sens_utc - sens_dev
        self.delta   = sens_offset - aria_offset    

    def get_delta(self) -> int:
        return self.delta

    def to_aria_time(self, sensor_ts_ns: int) -> int:
        return sensor_ts_ns + self.delta
    

