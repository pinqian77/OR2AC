import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque

from replay_memory import ReplayMemory
from env.env_sampler import EnvSampler


def readParser():
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--task_name', default = "online")
    parser.add_argument('--seed', type = int, default = 0)

    # model parameters
    parser.add_argument('--env', default = "riskymassrandom")
    parser.add_argument('--algo', default = "sac")
    parser.add_argument('--entropy_tuning', type = bool, default = False)
    parser.add_argument('--risk_type', default = "neutral")
    parser.add_argument('--risk_param', default = 0.1)

    # environment parameters
    parser.add_argument('--risk_prob', type = float, default = 0.9)
    parser.add_argument('--risk_penalty', type = float, default = 50.0)
    parser.add_argument('--dist_penalty_type', default = "none")
    parser.add_argument('--penalty', type = float, default = 1.0)

    # training parameters
    parser.add_argument('--replay_size', type = int, default = 1000000)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--max_episode_length', type = int, default = 100)
    parser.add_argument('--max_episode', type = int, default = 100)
    parser.add_argument('--max_level', type = int, default = 5)
    parser.add_argument('--init_exploration_steps', type = int, default = 1000)
    parser.add_argument('--p_threshold', type = float, default = -25)
    parser.add_argument('--p_window', type = int, default = 10)

    parser.add_argument('--save_interval', type = int, default = 10)
    parser.add_argument('--eval_interval', type = int, default = 10)
    parser.add_argument('--eval_n_episodes', type = int, default = 100)
    
    return parser.parse_args()

def train(args, env_sampler, agent, memory):
    # random exploration before training
    for i in range(args.init_exploration_steps):
        state, action, next_state, reward, done, _ = env_sampler.sample(agent, random_explore = True)
        memory.push(state, action, reward, next_state, done)

    # main loop
    total_step = 0
    reward_history = deque(maxlen=args.p_window)    
    for episode_step in tqdm(range(args.max_episode)):
        curr_level = env.num_obstacles
        episode_reward = 0

        # save buffer for offline learning
        if (episode_step + 1) % args.save_interval == 0:
            os.makedirs(f'dataset/{args.env}/{args.task_name}/level{curr_level}', exist_ok = True)
            buffer_path = f'dataset/{args.env}/{args.task_name}/level{curr_level}/{args.run_name}-epoch{episode_step + 1}.npy'
            memory.save_buffer(buffer_path)
            
            agent_path = f'saved_policies/{args.env}/online/{args.task_name}/{args.run_name}-epoch{episode_step + 1}'
            agent.save_model(agent_path)

        # interact with environment and train policy
        env_sampler.current_state = None
        env_sampler.path_length = 0
        for i in range(args.max_episode_length):

            # check if episode is done
            if (episode_step + 1) >= args.max_episode_length:
                break

            # step in environment
            state, action, next_state, reward, done, info = env_sampler.sample(agent)
            memory.push(state, action, reward, next_state, done)
            episode_reward += reward

            # train policy
            if len(memory) > 1000:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, 1)
            total_step += 1
        reward_history.append(episode_reward)

        # evaluate policy
        if episode_step % args.eval_interval == 0:
            rewards = [evaluate(env_sampler, agent, args.max_episode_length) for _ in range(args.eval_n_episodes)]
            rewards = np.array(rewards)
            rewards_avg = np.mean(rewards, axis = 0)
            rewards_std = np.std(rewards, axis = 0)
            sorted_rewards = np.sort(rewards)
            cvar = sorted_rewards[:int(0.1 * sorted_rewards.shape[0])].mean()
            print("")
            print(f'Epoch {episode_step} Eval_Reward {rewards_avg:.2f} Eval_Cvar {cvar:.2f} Eval_Std {rewards_std:.2f}')
        
        # Check if the agent's performance is above the threshold and increase difficulty based on curriculum learning
        if len(reward_history) >= args.p_window:
            performance = np.mean(reward_history)
            print(f"Episode {episode_step}: number of obstacles: {env.num_obstacles}, average reward: {performance}")
            if performance > args.p_threshold:
                new_obstacle_number = min(env.num_obstacles + 1, args.max_level)
                env.update_obstacle_number(new_obstacle_number)
                print(f"Episode {episode_step}: Updated number of obstacles to {env.num_obstacles}")
                reward_history = []

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

    # set run name
    if args.algo == 'codac':
        run_name = f"online-{args.risk_prob}-{args.risk_penalty}-{args.algo}-{args.risk_type}{args.risk_param}-E{args.entropy_tuning}-{args.seed}"
    else:
        run_name = f"online-{args.risk_prob}-{args.risk_penalty}-{args.algo}-E{args.entropy_tuning}-{args.seed}"
    args.run_name = run_name

    # Initial environment   
    if args.env == "riskymass":
        from env.risky_pointmass import PointMass
        env = PointMass(risk_prob=args.risk_prob, risk_penalty=args.risk_penalty)
        args.max_episode_length = 100
        args.max_episode = 100
    elif args.env == "riskymassrandom":
        from env.risky_pointmass_random import PointMassRandom
        env = PointMassRandom(risk_prob=args.risk_prob, risk_penalty=args.risk_penalty, num_obstacles=1)
        args.max_episode_length = 100
        args.max_episode = 1000

    os.makedirs(f'saved_policies/{args.env}/online/{args.task_name}', exist_ok = True)
    os.makedirs(f'dataset/{args.env}/{args.task_name}', exist_ok = True)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Initialize agent model
    if args.algo == 'sac':
        from model.sac import SAC
        agent = SAC(num_inputs = env.observation_space.shape[0], 
                    action_space = env.action_space,
                    automatic_entropy_tuning = False)
    elif args.algo == 'codac':
        from model.codac import CODAC
        agent = CODAC(num_inputs = env.observation_space.shape[0], 
                      action_space = env.action_space,
                      risk_type = args.risk_type, 
                      risk_param = args.risk_param,
                      dist_penalty_type = args.dist_penalty_type)

    # Replay Buffer
    memory = ReplayMemory(capacity = args.replay_size)

    # Environment Sampler
    env_sampler = EnvSampler(env = env, max_path_length = args.max_episode_length)

    # Train
    train(args, env_sampler, agent, memory)