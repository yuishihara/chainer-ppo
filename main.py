from OpenGL import GL

import argparse

import roboschool
import gym


def build_env(args):
    env = gym.make(args.env)
    env.reset()
    return env


def start_training(args):
    print('training started')
    env = build_env(args)
    for episode in range(20):
        print('current episode: ', episode)
        observation = env.reset()
        for _ in range(args.horizon):
            env.render()
            actions = env.action_space
            sample_action = actions.sample()
            observation, reward, end_of_episode, _ = env.step(sample_action)
            print('observation: ', observation)
            print('actions: ', actions, ' shape: ', actions.shape)
            print('sample_action: ', sample_action)
            print('reward: ', reward)
            print('end of episode?: ', end_of_episode)
    env.close()


def main():
    parser = argparse.ArgumentParser()

    # Environment parameters
    parser.add_argument('--env', type=str, default='CartPole-v0')

    # Training parameters
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2.5*1e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda', type=float, default=0.95)

    args = parser.parse_args()

    start_training(args)


if __name__ == "__main__":
    main()
