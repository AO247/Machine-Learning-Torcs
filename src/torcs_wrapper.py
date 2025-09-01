import gym
import numpy as np
from gym import spaces
from gym_torcs import TorcsEnv


class GymTorcsWrapper(gym.Env):
    def __init__(self, vision=False, throttle=True, gear_change=False):
        self.torcs_env = TorcsEnv(vision=vision, throttle=throttle, gear_change=gear_change)

        # Definiujemy przestrzeń obserwacji o 29 wymiarach
        high = np.array([np.inf] * 29)
        low = np.array([-np.inf] * 29)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = self.torcs_env.action_space

    def step(self, action):
        obs_tuple, reward, done, info = self.torcs_env.step(action)
        obs_array = self._convert_obs(obs_tuple)
        return obs_array, reward, done, {}

    def reset(self):
        obs_tuple = self.torcs_env.reset()
        obs_array = self._convert_obs(obs_tuple)
        return obs_array

    def end(self):
        self.torcs_env.end()

    def close(self):
        self.torcs_env.end()

    def _convert_obs(self, obs_tuple):
        # NOWA, BEZPIECZNA METODA: Tworzymy listę, a dopiero potem konwertujemy na array
        obs_list = [
            obs_tuple.speedX,
            obs_tuple.speedY,
            obs_tuple.speedZ,
            obs_tuple.angle,
            obs_tuple.trackPos,
            obs_tuple.rpm
        ]
        obs_list.extend(obs_tuple.track)
        obs_list.extend(obs_tuple.wheelSpinVel)

        return np.array(obs_list).astype(np.float32)

    def render(self, mode='human'):
        pass