import chainer

import numpy as np


class PPOActor(object):
    def __init__(self, env, timesteps, gamma, lmb, device=-1, render=False):
        super(PPOActor, self).__init__()
        self._env = env
        self._device = device
        self._timesteps = timesteps
        self._gamma = gamma
        self._lambda = lmb
        self._render = render
        self._state = self._env.reset()

    def run_policy(self, policy, value_function):
        dataset = []
        with chainer.no_backprop_mode():
            for _ in range(self._timesteps):
                s_current = self._state

                state = chainer.Variable(np.reshape(
                    s_current, newshape=(1, ) + s_current.shape))
                # print('state shape: ', state.shape)
                if not self._device < 0:
                    state.to_gpu()

                action = policy(state)
                # print('action shape: ', action.shape)
                log_likelihood = policy.compute_log_likelihood(state, action)
                # print('likelihood shape: ', log_likelihood.shape)

                action.to_cpu()
                action = action.data
                action = np.squeeze(action)
                # print('after action shape: ', action.shape)

                log_likelihood.to_cpu()
                log_likelihood = log_likelihood.data
                log_likelihood = np.squeeze(log_likelihood)
                # print('after log likelihood shape: ', log_likelihood.shape)

                s_next, reward, end, _ = self._env.step(action)
                reward = np.float32(reward)

                if end:
                    self._state = self._env.reset()
                else:
                    self._state = s_next

                # print('dtypes: s_current: {}, action: {}, reward: {}, s_next: {}, ll: {}'.format(
                #    s_current.dtype, action.dtype, reward.dtype, s_next.dtype, log_likelihood.dtype))
                data = (s_current, action, reward, s_next, log_likelihood)
                dataset.append(data)
        v_target, advantage = self._compute_v_target_and_advantage(
            dataset, value_function)
        return dataset, v_target, advantage

    def run_evaluation(self, policy, test_env, trials):
        rewards = []
        print('evaluation start')
        with chainer.no_backprop_mode():
            for trial in range(trials):
                # print('evaluation trial: ', trial)
                s_current = test_env.reset()
                done = False
                reward = 0.0
                while not done:
                    state = chainer.Variable(np.reshape(
                        s_current, newshape=(1, ) + s_current.shape))

                    if not self._device < 0:
                        state.to_gpu()

                    action = policy(state)
                    action.to_cpu()
                    action = action.data
                    action = np.squeeze(action)

                    s_current, reward, done, _ = test_env.step(action)
                    reward += np.float32(reward)

                rewards.append(reward)
                # print('trial ', trial, ' total reward: ', reward)
        return rewards

    def release(self):
        self._env.close()

    def _compute_v_target_and_advantage(self, dataset, value_function):
        T = len(dataset)
        v_targets = []
        advantages = []
        advantage = 0
        v_current = None
        v_next = None
        for t in reversed(range(T)):
            s_current, _, r, s_next, _ = dataset[t]

            s_current = chainer.Variable(np.reshape(
                s_current, newshape=(1, ) + s_current.shape))

            if not self._device < 0:
                s_current.to_gpu()
            v_current = value_function(s_current)
            v_current.to_cpu()
            v_current = np.squeeze(v_current.data)
            if v_next is None:
                s_next = chainer.Variable(np.reshape(
                    s_next, newshape=(1, ) + s_next.shape))
                if not self._device < 0:
                    s_next.to_gpu()
                v_next = value_function(s_next)
                v_next.to_cpu()
                v_next = np.squeeze(v_next.data)

            v_target = np.float32(
                r + self._gamma * v_next + self._gamma * self._lambda * advantage)
            advantage = np.float32(v_target - v_current)
            v_next = v_current

            v_targets.insert(0, v_target)
            advantages.insert(0, advantage)
        return v_targets, advantages
