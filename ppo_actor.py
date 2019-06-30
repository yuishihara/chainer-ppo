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

    def run_policy(self, model):
        dataset = []
        with chainer.no_backprop_mode():
            for _ in range(self._timesteps):
                s_current = self._state

                state = chainer.Variable(np.reshape(
                    s_current, newshape=(1, ) + s_current.shape))
                if not self._device < 0:
                    state.to_gpu()

                action = model(state)
                # print('action shape: ', action.shape)
                log_likelihood = model.compute_log_likelihood(state, action)
                # print('likelihood shape: ', log_likelihood.shape)

                action.to_cpu()
                action = action.data
                action = np.squeeze(action, axis=0)
                # print('after action ', action, ' shape: ', action.shape)

                log_likelihood.to_cpu()
                log_likelihood = log_likelihood.data
                log_likelihood = np.squeeze(log_likelihood)
                # print('after log likelihood shape: ', log_likelihood.shape)

                s_next, reward, done, _ = self._env.step(action)
                reward = np.float32(reward)

                if done:
                    self._state = self._env.reset()
                else:
                    self._state = s_next

                # print('dtypes: s_current: {}, action: {}, reward: {}, s_next: {}, ll: {}'.format(
                #    s_current.dtype, action.dtype, reward.dtype, s_next.dtype, log_likelihood.dtype))
                data = (s_current, action, reward,
                        s_next, done, log_likelihood)
                dataset.append(data)
        v_target, advantage = self._compute_v_target_and_advantage(
            dataset, model)
        return dataset, v_target, advantage

    def _compute_v_target_and_advantage(self, dataset, model):
        T = len(dataset)
        v_targets = []
        advantages = []
        advantage = 0
        v_current = None
        v_next = None
        for t in reversed(range(T)):
            s_current, _, r, s_next, done, _ = dataset[t]

            s_current = chainer.Variable(np.reshape(
                s_current, newshape=(1, ) + s_current.shape))

            if not self._device < 0:
                s_current.to_gpu()
            v_current = model.value(s_current)
            v_current.to_cpu()
            v_current = np.squeeze(v_current.data)
            if v_next is None:
                s_next = chainer.Variable(np.reshape(
                    s_next, newshape=(1, ) + s_next.shape))
                if not self._device < 0:
                    s_next.to_gpu()
                v_next = model.value(s_next)
                v_next.to_cpu()
                v_next = np.squeeze(v_next.data)
            if done:
                v_target = np.float32(r)
                advantage = 0
            else:
                v_target = np.float32(
                    r + self._gamma * v_next + self._gamma * self._lambda * advantage)
            advantage = np.float32(v_target - v_current)
            v_next = v_current

            v_targets.insert(0, v_target)
            advantages.insert(0, advantage)
        return v_targets, advantages

    def run_evaluation(self, model, test_env, trials, render=False):
        rewards = []
        print('evaluation start')
        with chainer.no_backprop_mode():
            for trial in range(trials):
                s_current = test_env.reset()
                done = False
                reward = 0.0
                while True:
                    if render:
                        test_env.render()
                    state = chainer.Variable(np.reshape(
                        s_current, newshape=(1, ) + s_current.shape))

                    if not self._device < 0:
                        state.to_gpu()

                    action = model(state)
                    action.to_cpu()
                    action = action.data
                    action = np.squeeze(action)

                    s_current, r, done, _ = test_env.step(action)
                    # print('reward: ', r, ' done?: ', done, ' action: ', action)
                    reward += np.float32(r)

                    if 0 == test_env.lives:
                        break
                    if done:
                        s_current = test_env.reset()
                
                print('evaluation trial: ', trial, ' reward: ', reward)
                rewards.append(reward)
                # print('trial ', trial, ' total reward: ', reward)
        return rewards

    def release(self):
        self._env.close()
