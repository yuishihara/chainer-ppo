from OpenGL import GL

import argparse

import numpy as np

import os

import roboschool
import gym

from ppo_actor import PPOActor

import numpy as np

from models.ppo_mujoco_model import PPOMujocoModel
from models.ppo_atari_model import PPOAtariModel

import atari_wrappers

from chainer import optimizers
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.datasets import tuple_dataset
import chainer.functions as F

from researchutils.arrays import unzip
from researchutils import files
import researchutils.chainer.serializers as serializers

from concurrent.futures import ThreadPoolExecutor


def build_env(args):
    if args.env_type == 'atari':
        env = atari_wrappers.build_atari_env(args.env)
    else:
        env = gym.make(args.env)
    env.reset()
    return env


def setup_adam_optimizer(model, lr):
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    return optimizer


def prepare_model(args, num_actions):
    env_type = args.env_type
    if env_type == 'mujoco':
        model = PPOMujocoModel(num_actions)
    elif env_type == 'atari':
        model = PPOAtariModel(num_actions)
    else:
        NotImplementedError("Unknown ent_type: ", env_type)
    serializers.load_model(args.model_params, model)
    return model


def prepare_iterator(args, *data):
    dataset = tuple_dataset.TupleDataset(*data)
    return iterators.SerialIterator(dataset, args.batch_size * args.actor_num)


def run_policy_of(actor, model):
    return actor.run_policy(model)


def sample_data(actors, model, executor):
    datasets = []
    v_targets = []
    advantages = []

    futures = []
    for actor in actors:
        future = executor.submit(
            run_policy_of, actor, model)
        futures.append(future)
    for future in futures:
        dataset, v_target, advantage = future.result()
        datasets.extend(dataset)
        v_targets.extend(v_target)
        advantages.extend(advantage)

    s_current, a, r, s_next, _, likelihood = unzip(datasets)
    data = (s_current, a, r, s_next,  likelihood, v_targets, advantages)
    any(len(item) == len(s_current) for item in data)
    return data


def optimize_surrogate_loss(iterator, model, optimizer, alpha, args):
    optimizer.target.cleargrads()

    batch = iterator.next()
    s_current, action, _, _, log_likelihood, v_target, advantage = concat_examples(
        batch, device=args.gpu)

    log_pi_theta = model.compute_log_likelihood(s_current, action)
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

    value = model.value(s_current)
    # print('value: ', value, ' shape: ', value.shape)
    # print('v_target: ', v_target, ' shape: ', v_target.shape)
    value_loss = F.mean_squared_error(value, v_target)

    entropy = model.compute_entropy(s_current)
    entropy_loss = F.sum(entropy)

    loss = -clip_loss + args.vf_coeff * value_loss - args.entropy_coeff * entropy_loss

    # Update parameter
    loss.backward()
    optimizer.update()
    loss.unchain_backward()


def run_training_loop(actors, model, test_env, outdir, args):
    optimizer = setup_adam_optimizer(model, args.learning_rate)

    result_file = os.path.join(outdir, 'result.txt')
    if not files.file_exists(result_file):
        with open(result_file, "w") as f:
            f.write('timestep\tmean\tmedian\n')

    alpha = 1.0
    with ThreadPoolExecutor(max_workers=8) as executor:
        previous_evaluation = 0
        for timestep in range(0, args.total_timesteps, args.timesteps * args.actor_num):
            alpha = (1.0 - timestep / args.total_timesteps)
            print('current timestep: ', timestep, '/', args.total_timesteps)
            data = sample_data(actors, model, executor)
            iterator = prepare_iterator(args, *data)
            for _ in range(args.epochs):
                # print('epoch num: ', epoch)
                iterator.reset()
                while not iterator.is_new_epoch:
                    optimize_surrogate_loss(
                        iterator, model, optimizer, alpha, args)
            optimizer.hyperparam.alpha = args.learning_rate * alpha
            print('optimizer step size', optimizer.hyperparam.alpha)

            if (timestep - previous_evaluation) // args.evaluation_interval == 1:
                previous_evaluation = timestep
                actor = actors[0]
                rewards = actor.run_evaluation(
                    model, test_env, args.evaluation_trial)

                mean = np.mean(rewards)
                median = np.median(rewards)
                print('mean: {mean}, median: {median}'.format(
                    mean=mean, median=median))

                print('saving model of iter: ', timestep, ' to: ')
                model_filename = 'model_iter-{}'.format(timestep)

                model.to_cpu()
                serializers.save_model(os.path.join(
                    outdir, model_filename), model)
                if not args.gpu < 0:
                    model.to_gpu()

                with open(result_file, "a") as f:
                    f.write('{timestep}\t{mean}\t{median}\n'.format(
                        timestep=timestep, mean=mean, median=median))


def start_training(args):
    print('training started')
    test_env = build_env(args)
    print('action space: ', test_env.action_space)
    if args.env_type == 'atari':
        action_num = test_env.action_space.n
    else:
        action_num = test_env.action_space.shape[0]

    model = prepare_model(args, action_num)

    if not args.gpu < 0:
        model.to_gpu()

    actors = []
    for _ in range(args.actor_num):
        env = build_env(args)
        actor = PPOActor(env, args.timesteps, args.gamma, args.lmb, args.gpu)
        actors.append(actor)

    outdir = files.prepare_output_dir(base_dir=args.outdir, args=args)

    run_training_loop(actors, model, test_env, outdir, args)

    for actor in actors:
        actor.release()

    test_env.close()


def start_test_run(args):
    print('test run started')
    test_env = build_env(args)
    action_num = test_env.action_space.shape[0]

    model = prepare_model(args, action_num)

    if not args.gpu < 0:
        model.to_gpu()

    actor = PPOActor(test_env, args.timesteps, args.gamma, args.lmb, args.gpu)
    rewards = actor.run_evaluation(model, test_env, 10, render=True)
    mean = np.mean(rewards)
    median = np.median(rewards)
    print('test run result = mean: ', mean, ' median: ', median)

    actor.release()
    test_env.close()


def main():
    parser = argparse.ArgumentParser()

    # training/test option
    parser.add_argument('--test-run', action='store_true')

    # data saving options
    parser.add_argument('--outdir', type=str, default='results')

    # Environment parameters
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v0')

    # Policy types
    parser.add_argument('--env-type', type=str,
                        choices=['atari', 'mujoco'], required=True)

    # Evaluation setting
    parser.add_argument('--evaluation-interval', type=int, default=100000)
    parser.add_argument('--evaluation-trial', type=int, default=10)

    # Gpu setting
    parser.add_argument('--gpu', type=int, default=0)

    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000)
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
    parser.add_argument('--model-params', type=str, default='')

    args = parser.parse_args()

    if args.test_run:
        start_test_run(args)
    else:
        start_training(args)


if __name__ == "__main__":
    main()
