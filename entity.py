import numpy as np


class Action:
    def __init__(self):
        self.flight_velocity = None
        self.flight_direction = None


class Sensor:
    def __init__(self, id):
        self.sensor_id = id
        self.pos = None
        self.aoi = None
        self.sen_fre = None
        self.color = 'green'
        self.size = 5
        self.package_generate_time_slot = None


class UAV:
    def __init__(self):
        self.pos = None
        self.aoi = None
        self.color = 'red'
        self.size = 10
        self.action = Action()


class BS:
    def __init__(self):
        self.pos = None
        self.aoi = None
        self.color = 'blue'
        self.size = 10