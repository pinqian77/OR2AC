import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque

from replay_memory import ReplayMemory
from env.env_sampler import EnvSampler
from env.risky_pointmass_random import PointMassRandom


def readParser():
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--task_name', default = "offline")
    parser.add_argument('--seed', type = int, default = 0)

    # model parameters
    parser.add_argument('--env', default = "riskymass")
    parser.add_argument('--algo', default = "codac")
    parser.add_argument('--entropy_tuning', type = bool, default = False)
    parser.add_argument('--risk_type', default = "cvar")
    parser.add_argument('--risk_param', default = 0.1)
    parser.add_argument('--tau_type', default = "iqn")
    parser.add_argument('--lag', type=float, default=10.0)
    parser.add_argument('--min_z_weight', type=float, default=10.0)

    # environment parameters
    parser.add_argument('--risk_prob', type = float, default = 0.9)
    parser.add_argument('--risk_penalty', type = float, default = 50.0)
    parser.add_argument('--dist_penalty_type', default = "cvar")
    parser.add_argument('--penalty', type = float, default = 1.0)

    # training parameters
    parser.add_argument('--dataset_epoch', type=int, default=100)
    parser.add_argument('--replay_size', type = int, default = 2000000)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--max_episode_length', type = int, default = 100)
    parser.add_argument('--max_episode', type = int, default = 1000)
    parser.add_argument('--init_exploration_steps', type = int, default = 1000)
    parser.add_argument('--p_threshold', type = float, default = -25)
    parser.add_argument('--p_window', type = int, default = 10)

    parser.add_argument('--save_interval', type = int, default = 20)
    parser.add_argument('--eval_interval', type = int, default = 10)
    parser.add_argument('--eval_n_episodes', type = int, default = 100)

    return parser.parse_args()


def train(args, env_sampler, agent, memory):
    reward_history = deque(maxlen=args.p_window)
    current_dataset_index = 0
    memory = env_pools[current_dataset_index]

    # main loop
    total_step = 0
    for episode_step in tqdm(range(args.max_episode)):
        if (episode_step + 1) % args.save_interval == 0:
            agent_path = f'saved_policies/{args.env}/{args.dataset}/{args.run_name}-epoch{episode_step + 1}'
            agent.save_model(agent_path)

        for i in range(args.max_episode_length):
            if (episode_step + 1) >= args.max_episode_length:
                break

            # train policy
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, 1)
            total_step += 1

        if episode_step % args.eval_interval == 0:
            rewards = [evaluate(env_sampler, agent, args.max_episode_length) for _ in range(args.eval_n_episodes)]
            rewards = np.array(rewards)
            rewards_avg = np.mean(rewards, axis = 0)
            rewards_std = np.std(rewards, axis = 0)
            sorted_rewards = np.sort(rewards)
            cvar = sorted_rewards[:int(0.1 * sorted_rewards.shape[0])].mean()

            reward_history.append(rewards_avg)
            if len(reward_history) >= args.p_window:
                performance = np.mean(reward_history)
                if performance >= args.p_threshold and current_dataset_index < len(env_pools) - 1:
                    current_dataset_index += 1
                    memory = env_pools[current_dataset_index]
                    print(f"Switching to dataset {current_dataset_index + 1}")
                    reward_history = []
            print("")
            print(f'Epoch {episode_step} Eval_Reward {rewards_avg:.2f} Eval_Cvar {cvar:.2f} Eval_Std {rewards_std:.2f}')

def evaluate(env_sampler, agent, epoch_length):
    env_sampler.current_state = None
    env_sampler.path_length = 0
    sum_reward = 0
    for t in range(epoch_length):
        _, _, _, reward, done, _ = env_sampler.sample(agent, eval_t=True)
        sum_reward += reward
        if done:
            break
    return sum_reward


if __name__ == '__main__':
    args = readParser()

    # Initial environment
    env = PointMassRandom(risk_prob = args.risk_prob, risk_penalty = args.risk_penalty)

    os.makedirs(f'saved_policies/{args.env}', exist_ok=True)
    run_name = f"offline-{args.algo}-{args.dist_penalty_type}-{args.risk_type}{args.risk_param}-{args.seed}"
    args.run_name = run_name

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.algo == 'sac':
        from model.sac import SAC
        agent = SAC(num_inputs = env.observation_space.shape[0], 
                    action_space = env.action_space,
                    automatic_entropy_tuning = False)
    elif args.algo == 'codac':
        from model.codac import CODAC
        agent = CODAC(num_inputs = env.observation_space.shape[0], 
                      action_space = env.action_space,
                      tau_type = args.tau_type, 
                      min_z_weight = args.min_z_weight,
                      risk_type = args.risk_type, 
                      risk_param = args.risk_param,
                      dist_penalty_type = args.dist_penalty_type,
                      lagrange_thresh = args.lag)

    # Sampler Environment
    env_sampler = EnvSampler(env, max_path_length=args.max_episode_length)

    # Initial replay buffer for env
    dataset_path = f'./dataset/{args.env}/{args.task_name}'
    level_files = sorted(os.listdir(dataset_path))
    parent_directory = f'./dataset/{args.env}/{args.task_name}'
    subdirectories = sorted(os.listdir(dataset_path))

    levels = []
    for subdir in subdirectories:
        # Construct the subdirectory path
        subdir_path = os.path.join(parent_directory, subdir)
        files = os.listdir(subdir_path)
        npy_files = [file for file in files if file.endswith('.npy')]

        # Iterate through the .npy files and construct their paths
        level = []
        for npy_file in npy_files:
            file_path = os.path.join(subdir_path, npy_file)
            level.append(file_path)
        levels.append(level)

    env_pools = []
    for level in levels:
        for path in level:
            dataset = np.load(path, allow_pickle=True).item()
            n = dataset['observations'].shape[0]
            env_pool = ReplayMemory(n)
            for i in range(n):
                state, action, reward, next_state, done = dataset['observations'][i], dataset['actions'][i], dataset['rewards'][i], dataset['next_observations'][i], dataset['terminals'][i]
                env_pool.push(state, action, reward, next_state, done)
            env_pools.append(env_pool)
    print(f"{args.env} dataset loaded with {len(env_pools)} levels")

    # Train
    train(args, env_sampler, agent, env_pool)