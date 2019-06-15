import chainer


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
                if self._render:
                    self._env.render()
                s_current = self._state

                state = chainer.Variable(s_current)
                state.to_gpu()

                action = policy(state)
                likelihood = policy.likelihood(state, action)

                action.to_cpu()
                action = action.data
                likelihood.to_cpu()
                likelihood = likelihood.data

                s_next, reward, end, _ = self._env.step(action)

                if end:
                    self._state = self._env.reset()
                else:
                    self._state = s_next

                data = (s_current, action, reward, s_next, likelihood)
                dataset.append(data)
        v_target, advantage = self._compute_v_target_and_advantage(
            dataset, value_function)
        return dataset, v_target, advantage

    def _compute_v_target_and_advantage(self, dataset, value_function):
        T = len(dataset)
        v_targets = []
        advantages = []
        advantage = 0
        v_current = None
        v_next = None
        for t in reversed(range(T)):
            s_current, _, r, s_next, _ = dataset[t]

            s_current = chainer.Variable(s_current)
            s_current.to_gpu()
            v_current = value_function(s_current)
            v_current.to_cpu()
            v_current = v_current.data
            if v_next is None:
                s_next = chainer.Variable(s_next)
                s_next.to_gpu()
                v_next = value_function(s_next)
                v_next.to_cpu()
                v_next = v_next.data

            v_target = r + self._gamma * v_next + self._gamma * self._lambda * advantage
            advantage = v_target - v_current
            v_next = v_current

            v_targets.insert(0, v_target)
            advantages.insert(0, advantage)
        return v_targets, advantages
