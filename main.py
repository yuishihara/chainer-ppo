from OpenGL import GL

import argparse

import numpy as np

import os

import roboschool
import gym

from ppo_actor import PPOActor

from models.ppo_policy import PPOPolicy
from models.ppo_value import PPOValue

from chainer import optimizers
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.datasets import tuple_dataset
import chainer.functions as F

from researchutils.arrays import unzip
from researchutils import files
import researchutils.chainer.serializers as serializers


def build_env(args):
    env = gym.make(args.env)
    env.reset()
    return env


def setup_adam_optimizer(model, lr):
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    return optimizer


def prepare_policy(args, num_actions):
    policy = PPOPolicy(num_actions)
    serializers.load_model(args.policy_model, policy)
    return policy


def prepare_value_function(args):
    value = PPOValue()
    serializers.load_model(args.value_model, value)
    return value


def prepare_iterator(args, *data):
    dataset = tuple_dataset.TupleDataset(*data)
    return iterators.SerialIterator(dataset, args.batch_size * args.actor_num)


def sample_data(actors, policy, value_function):
    datasets = []
    v_targets = []
    advantages = []
    for actor in actors:
        dataset, v_target, advantage = actor.run_policy(
            policy, value_function)
        datasets.extend(dataset)
        v_targets.extend(v_target)
        advantages.extend(advantage)

    s_current, a, s_next, r, likelihood = unzip(datasets)
    data = (s_current, a, s_next, r, likelihood, v_targets, advantages)
    any(len(item) == len(s_current) for item in data)
    return data


def optimize_surrogate_loss(iterator, policy, value_function, p_optimizer, v_optimizer, alpha, args):
    p_optimizer.target.cleargrads()
    v_optimizer.target.cleargrads()

    batch = iterator.next()
    s_current, action, _, _, log_likelihood, v_target, advantage = concat_examples(
        batch, device=args.gpu)

    log_pi_theta = policy.compute_log_likelihood(s_current, action)
    log_pi_theta_old = log_likelihood
    # print('log_pi_theta: ', log_pi_theta, ' shape: ', log_pi_theta.shape)
    # print('log_pi_theta_old: ', log_pi_theta_old, ' shape: ', log_pi_theta_old.shape)
    # division of probability is exponential of difference between log probability
    probability_ratio = F.exp(log_pi_theta - log_pi_theta_old)
    clipped_ratio = F.clip(
        probability_ratio, 1 - args.epsilon * alpha, 1 + args.epsilon * alpha)
    lower_bounds = F.minimum(
        probability_ratio * advantage, clipped_ratio * advantage)
    clip_loss = F.mean(lower_bounds)

    value = value_function(s_current)
    # print('value: ', value, ' shape: ', value.shape)
    # print('v_target: ', v_target, ' shape: ', v_target.shape)
    value_loss = F.mean_squared_error(value, v_target)

    entropy = log_likelihood * F.exp(log_likelihood)
    entropy_loss = F.sum(entropy)

    loss = -clip_loss + args.vf_coeff * value_loss - args.entropy_coeff * entropy_loss

    # Update parameter
    loss.backward()
    p_optimizer.update()
    v_optimizer.update()
    loss.unchain_backward()


def run_training_loop(actors, policy, value_function, test_env, outdir, args):
    p_optimizer = setup_adam_optimizer(policy, args.learning_rate)
    v_optimizer = setup_adam_optimizer(value_function, args.learning_rate)

    result_file = os.path.join(outdir, 'result.txt')
    if not files.file_exists(result_file):
        with open(result_file, "w") as f:
            f.write('iteration\tmean\tmedian\n')

    for iteration in range(args.iterations):
        print('current iteration: ', iteration)
        data = sample_data(actors, policy, value_function)
        iterator = prepare_iterator(args, *data)
        alpha = 1.0 / args.iterations * (args.iterations - iteration)
        print('alpha: ', alpha)
        for _ in range(args.epochs):
            # print('epoch num: ', epoch)
            iterator.reset()
            while not iterator.is_new_epoch:
                optimize_surrogate_loss(
                    iterator, policy, value_function, p_optimizer, v_optimizer, alpha, args)
        p_optimizer.hyperparam.alpha *= alpha
        v_optimizer.hyperparam.alpha *= alpha
        print('optimizer step size',
              ' p: ', p_optimizer.hyperparam.alpha,
              ' v: ', v_optimizer.hyperparam.alpha)

        if iteration % args.evaluation_interval == 0:
            actor = actors[0]
            rewards = actor.run_evaluation(
                policy, test_env, args.evaluation_trial)

            mean = np.mean(rewards)
            median = np.median(rewards)
            print('mean: {mean}, median: {median}'.format(
                mean=mean, median=median))

            print('saving model of iter: ', iteration, ' to: ')
            policy_filename = 'policy_iter-{}'.format(iteration)
            value_filename = 'value_iter-{}'.format(iteration)

            policy.to_cpu()
            value_function.to_cpu()
            serializers.save_model(os.path.join(
                outdir, policy_filename), policy)
            serializers.save_model(os.path.join(
                outdir, value_filename), value_function)
            policy.to_gpu()
            value_function.to_gpu()

            with open(result_file, "a") as f:
                f.write('{iteration}\t{mean}\t{median}\n'.format(
                    iteration=iteration, mean=mean, median=median))


def start_training(args):
    print('training started')
    test_env = build_env(args)
    action_num = test_env.action_space.shape[0]

    policy = prepare_policy(args, action_num)
    value_function = prepare_value_function(args)

    if not args.gpu < 0:
        policy.to_gpu()
        value_function.to_gpu()

    actors = []
    for _ in range(args.actor_num):
        env = build_env(args)
        actor = PPOActor(env, args.timesteps, args.gamma, args.lmb, args.gpu)
        actors.append(actor)

    outdir = files.prepare_output_dir(base_dir=args.outdir, args=args)

    run_training_loop(actors, policy, value_function, test_env, outdir, args)

    for actor in actors:
        actor.release()

    test_env.close()


def main():
    parser = argparse.ArgumentParser()

    # data saving options
    parser.add_argument('--outdir', type=str, default='results')

    # Environment parameters
    parser.add_argument('--env', type=str, default='RoboschoolHumanoid-v1')

    # Evaluation setting
    parser.add_argument('--evaluation-interval', type=int, default=1e4)
    parser.add_argument('--evaluation-trial', type=int, default=10)

    # Gpu setting
    parser.add_argument('--gpu', type=int, default=0)

    # Training parameters
    parser.add_argument('--iterations', type=int, default=5*1e7)
    parser.add_argument('--timesteps', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2.5*1e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmb', type=float, default=0.95)
    parser.add_argument('--actor-num', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--vf_coeff', type=float, default=1.0)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)

    # model paths
    parser.add_argument('--policy-model', type=str, default='')
    parser.add_argument('--value-model', type=str, default='')

    args = parser.parse_args()

    start_training(args)


if __name__ == "__main__":
    main()
