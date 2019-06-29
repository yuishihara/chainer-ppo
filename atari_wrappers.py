import gym
import numpy as np

from collections import deque


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)

    def observation(self, observation):
        normalized = np.array(observation).astype(np.float32) / 255.0
        return normalized


class StackedStatesEnv(gym.Wrapper):
    def __init__(self, env, size=4):
        super(StackedStatesEnv, self).__init__(env)
        self._stack_size = size
        self._frames = deque(maxlen=size)

    def reset(self):
        obs = self.env.reset()
        obs = np.reshape(obs, newshape=(1, ) + obs.shape)
        for _ in range(self._stack_size):
            self._frames.append(obs)
        return self._get_observation()

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = np.reshape(obs, newshape=(1, ) + obs.shape)
        self._frames.append(obs)
        return self._get_observation(), r, done, info

    def _get_observation(self):
        assert len(self._frames) == self._stack_size
        obs = np.concatenate(self._frames, axis=0)
        return obs


def build_atari_env(id):
    env = gym.make(id)
    env = gym.wrappers.atari_preprocessing.AtariPreprocessing(
        env, terminal_on_life_loss=True)
    env = NormalizedEnv(env)
    env = StackedStatesEnv(env)
    return env


if __name__ == "__main__":
    id = 'BreakoutNoFrameskip-v0'
    original_env = gym.make(id)
    original_env.reset()
    obs, _, _, _ = original_env.step(0)
    assert np.all(0.0 <= obs) and np.all(obs <= 255.0)

    atari_env = gym.wrappers.atari_preprocessing.AtariPreprocessing(original_env)
    obs, _, _, _ = atari_env.step(0)
    assert obs.shape == (84, 84)
    assert np.all(0.0 <= obs) and np.all(obs <= 255.0)
    
    env = StackedStatesEnv(atari_env)
    env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(0)
    assert obs.shape == (4, 84, 84)
    assert np.all(0.0 <= obs) and np.all(obs <= 255.0)

    env = NormalizedEnv(original_env)
    obs, _, _, _ = env.step(0)
    assert obs.shape == (210, 160, 3)
    assert np.all(0.0 <= obs) and np.all(obs <= 1.0)

    env = build_atari_env(id)
    env.reset()
    obs, _, _, _ = env.step(0)
    assert obs.shape == (4, 84, 84)
    assert np.all(0.0 <= obs) and np.all(obs <= 1.0)

