import gym
import numpy as np

from collections import deque

import cv2


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)

    def observation(self, observation):
        normalized = np.array(observation).astype(np.float32) / 255.0
        return normalized


class ClippedRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClippedRewardEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


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

# Copied from chainerrl implementation
# See: https://github.com/chainer/chainerrl/blob/master/chainerrl/wrappers/atari_wrappers.py


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game end.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.needs_real_reset = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.needs_real_reset = done or info.get('needs_reset', False)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few
            # frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.needs_real_reset:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


def build_atari_env(id, clip_reward=True):
    env = gym.make(id)
    env = gym.wrappers.atari_preprocessing.AtariPreprocessing(
        env, terminal_on_life_loss=False)
    env = EpisodicLifeEnv(env)
    env = NormalizedEnv(env)
    env = StackedStatesEnv(env)
    if clip_reward:
        env = ClippedRewardEnv(env)
    return env


if __name__ == "__main__":
    id = 'BreakoutNoFrameskip-v4'
    original_env = gym.make(id)
    original_env.reset()
    obs, _, _, _ = original_env.step(0)
    assert np.all(0.0 <= obs) and np.all(obs <= 255.0)

    atari_env = gym.wrappers.atari_preprocessing.AtariPreprocessing(
        original_env)
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
    
    reward = 0
    for i in range(1000):
        o, r, done, _ = env.step(i % 4)
        cv2.imshow('obs1', o[0])
        cv2.imshow('obs2', o[1])
        cv2.imshow('obs3', o[2])
        cv2.imshow('obs4', o[3])
        cv2.waitKey(0)
        reward += r
        if env.lives == 0:
            break
        if done:
            env.reset()
    print('earned reward: ', reward)
