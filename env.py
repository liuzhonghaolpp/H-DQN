import numpy as np
import torch
from gym import spaces, core
from gym.envs.classic_control import rendering
import entity
from utils import uav_mobility
import configparser
import time

config = configparser.RawConfigParser()
config.read('./paramaters.ini')

# 参数
MAX_AoI = 100
DURATION = config.get('simulation parameters', 'duration')


class MyEnv(core.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = None
        self.num_sensors = 6
        self.sensors = []
        self.side_length = 1000  # 目标区域的边长
        self.time = 0
        self.viewer = None
        for i in range(self.num_sensors):
            sensor = entity.Sensor(i)
            self.sensors.append(sensor)
        self.uav = entity.UAV()
        self.BS = entity.BS()

    def reset(self):
        # 初始化sensor的位置(非随机初始化)
        self.sensors[0].pos = np.array([250, 550])
        self.sensors[1].pos = np.array([550, 150])
        self.sensors[2].pos = np.array([750, 250])
        self.sensors[3].pos = np.array([350, 850])
        self.sensors[4].pos = np.array([550, 750])
        self.sensors[5].pos = np.array([750, 950])

        # 初始化UAV的位置和所携带数据AoI
        self.uav.pos = np.array([500, 500])
        self.uav.aoi = np.array([MAX_AoI, MAX_AoI, MAX_AoI, MAX_AoI, MAX_AoI, MAX_AoI])

        # 初始化BS的AoI
        self.BS.pos = np.array([500, 500])
        self.BS.aoi = np.array([MAX_AoI, MAX_AoI, MAX_AoI, MAX_AoI, MAX_AoI, MAX_AoI])

        self.time = 0
        obs = self._get_observation()
        return obs

    def step(self, action):
        self.uav.action = action
        # 更新UAV的位置
        uav_mobility.get_next_pos(self.uav)
        # 查看UAV是否在sensor的通信范围内，且未采集该设备数据，然后更新uav_aoi
        # 查看UAV是否在BS通信范围内，且携带有未上传的数据，然后更新bs_aoi
        done = self._get_done()
        reward = self._get_reward()
        obs = self._get_observation()
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        screen_width = 500 # 按比例缩小一下，1:2的比例
        screen_height = 500
        # 如果没有viewer，创建viewer和uav、landmarks
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_height, screen_width)
        self.viewer.set_bounds(0, 500, 0, 500)
        self.viewer.geoms.clear()
        for sensor in self.sensors:
            geom = rendering.make_circle(sensor.size)
            geom.set_color(1, 0, 0)
            geom_form = rendering.Transform(translation=(sensor.pos[0]/2, sensor.pos[1]/2))
            geom.add_attr(geom_form)
            self.viewer.add_geom(geom)

        geom = rendering.make_circle(self.BS.size)
        geom.set_color(0, 0, 1)
        geom_form = rendering.Transform(translation=(self.BS.pos[0]/2, self.BS.pos[1]/2))
        geom.add_attr(geom_form)
        self.viewer.add_geom(geom)

        geom = rendering.make_circle(self.uav.size)
        geom.set_color(0, 1, 0)
        geom_form = rendering.Transform(translation=(self.uav.pos[0]/2, self.uav.pos[1]/2))
        geom.add_attr(geom_form)
        self.viewer.add_geom(geom)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_observation(self):
        obs_uav = np.concatenate((self.uav.pos, self.uav.aoi), axis=0)
        obs_bs = self.BS.aoi
        obs = np.concatenate((obs_uav, obs_bs), axis=0)
        return obs

    def _get_done(self):
        return None

    def _get_reward(self):
        return None


if __name__ == '__main__':
    env = MyEnv()
    obs = env.reset()
    env.render()

    while True:
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        env.render()
        time.sleep(2)