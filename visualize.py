
from env.risky_pointmass_random import PointMassRandom

MAX_NUM_EPISODES = 200
MAX_NUM_STEPS = 100

def play(env, agent):
    episode = 0
    while episode < MAX_NUM_EPISODES: 
        episode += 1
        episode_score = 0
        episode_steps = 0
        state = env.reset(eval=True)
        for _ in range(MAX_NUM_STEPS):
            env.render(show_local_grid=False)
            action = agent.select_action(state, eval=True)  # Sample action from policy
            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            episode_score += reward
            if done:
                break
            state = next_state

env = PointMassRandom(num_obstacles = 5, random_obstacle = True, random_goal = False)
env.seed(0)