import math
import numpy as np
import configparser

config = configparser.RawConfigParser()
config.read('./paramaters.ini')
max_velocity = int(config.get('simulation parameters', 'max_velocity'))
time_slot = int(config.get('simulation parameters', 'time_slot'))


def get_next_pos(uav):
    cur_pos = uav.pos.copy()
    velocity = (uav.action[1]+1) * (max_velocity/2)
    direction = uav.action[0] * 2 * math.pi

    if direction <= math.pi / 2:
        cur_pos[0] -= velocity * time_slot * math.cos(direction)
        cur_pos[1] += velocity * time_slot * math.sin(direction)
    elif math.pi / 2 < direction <= math.pi:
        cur_pos[0] += velocity * time_slot * math.cos(math.pi - direction)
        cur_pos[1] += velocity * time_slot * math.sin(math.pi - direction)
    elif math.pi < direction <= math.pi * 3 / 2:
        cur_pos[0] += velocity * time_slot * math.cos(direction - math.pi)
        cur_pos[1] -= velocity * time_slot * math.sin(direction - math.pi)
    else:
        cur_pos[0] -= velocity * time_slot * math.cos(2 * math.pi - direction)
        cur_pos[1] -= velocity * time_slot * math.sin(2 * math.pi - direction)
    # if 0 <= cur_pos[0] <= 1000 and 0 <= cur_pos[1] <= 1000:
    uav.pos = cur_pos
