import gym
import pygame
from gym import spaces
from gym.utils import seeding
import numpy as np

class PointMass(gym.Env):
    def __init__(self, risk_prob=0.9, risk_penalty=50):
        # Agent Settings
        self.v_max = 0.1                            # Maximum velocity of agent
        self.init_pos = np.array([1.0, 1.0])        # Initial position of agent

        # Environment Settings
        self.low_state = 0                          # Lower bound of state space
        self.high_state= 1                          # Upper bound of state space
        self.min_actions = np.array(
            [-self.v_max, -self.v_max], dtype=np.float32
        )
        self.max_actions = np.array(
            [self.v_max, self.v_max], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2+2, ),
            dtype=np.float32
        )

        self.goal = np.array([0.1, 0.1])        # Goal position
        self.d_goal = 0.05                      # Distance threshold for the goal

        self.risk_prob = risk_prob              # Probability of encountering risk situation
        self.risk_penalty = risk_penalty        # Penalty of risk
        self.obstacle_r = 0.3                   # Radius of obstacle
        self.obstacle_c = np.array([0.5, 0.5])  # Center of obstacle

        # Render Settings
        self.pygame_init = False

        self.screen_size = [600, 600]
        self.screen_scale = 600
        self.background_color = [223, 220, 230]

        self.obstacle_color = [36, 39, 46]
        self.goal_color = [0, 255, 0]
        self.agent_color = [0, 0, 0]
        self.agent_r = 5                        # Radius of agent
        self.agent_w = 3                        # Width of agent

        self.trace_points = []                  # List of (position, velocity) tuples for the current episode
        self.episode_traces = []                # List of trace_points lists for all episodes
        self.trace_color = [255, 0, 0]          # Color of trace

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

    def reset(self, eval=False):
        # Sample a new initial state
        sampled = False
        while not sampled:
            # uniform state space initial state distribution
            self.init_pos = self.np_random.uniform(0.1, 0.9, size=(2,))
            # sample initial state from a safe region
            if self.is_safe(self.init_pos):
                sampled = True
        if eval:
            # Save the previous episode's trace and clear the trace_points list for the new episode
            if self.trace_points:
                self.episode_traces.append(self.trace_points)
            self.trace_points = []
        self.state = np.array(list(self.init_pos) + list(self.goal))
        return np.array(self.state)

    # Help function to check if the state is safe.
    def is_safe(self, state):
        if len(state.shape) == 1:
            safe = True
            d_circle = (state[0]-self.obstacle_c[0])**2 + (state[1]-self.obstacle_c[1])**2
            if d_circle <= (self.obstacle_r ** 2):
                safe = False
            return safe
        elif len(state.shape) == 2:
            d_circle = (state[:, 0] - 0.5) ** 2 + (state[:, 1] - 0.5) ** 2
            safe = (d_circle > self.obstacle_r ** 2).astype(float)
            return safe

    def step(self, action):
        # clip the action to the action space
        action = np.clip(action, -self.v_max, self.v_max)
        assert self.action_space.contains(action)

        # Save the current position and velocity in the trace_points list
        self.trace_points.append((self.state[:2].copy(), action.copy()))

        # compute the distance to the goal
        d_goal = np.linalg.norm(self.state[-2:] - self.state[:2])
        
        # compute the reward
        reward = - d_goal - 0.1
        cost = 0
        if not self.is_safe(self.state):
            u = np.random.uniform(0, 1)
            if u > self.risk_prob:
                cost = 1
                reward -= self.risk_penalty
        
        # check if the episode is done
        done = 0
        if d_goal < self.d_goal:
            done = 1

        # update the state
        self.state[:2] = self.state[:2] + action
        self.state = np.clip(self.state, self.low_state, self.high_state)

        return np.array(self.state), reward, done, {'cost':cost}

    # Help Function to map velocity to color
    def velocity_to_color(self, velocity):
        v_norm = (np.linalg.norm(velocity) + self.v_max) / 2 * self.v_max
        return [int(255 * (1 - v_norm)), int(255 * v_norm), 0]

    def render(self):
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(self.background_color)

        p_car = self.state[:2]
        p = (self.screen_scale * p_car).astype(int).tolist()

        # Draw the obstacle
        c, r = (self.screen_scale*self.obstacle_c[:2]).astype(int), int(self.screen_scale*self.obstacle_r)
        pygame.draw.circle(self.screen, self.obstacle_color, c, r)

        # Draw the goal
        pygame.draw.circle(self.screen, self.goal_color, (self.screen_scale * self.goal).astype(int), int(self.screen_scale * self.d_goal))

        # Draw all the previous episode traces
        for episode_trace in self.episode_traces:
            if len(episode_trace) > 1:
                for i in range(len(episode_trace) - 1):
                    start_pos = self.screen_scale * episode_trace[i][0]
                    end_pos = self.screen_scale * episode_trace[i + 1][0]
                    pygame.draw.line(self.screen, self.trace_color, start_pos, end_pos, 2)

        # Draw the current episode trace
        if len(self.trace_points) > 1:
            for i in range(len(self.trace_points) - 1):
                start_pos = self.screen_scale * self.trace_points[i][0]
                end_pos = self.screen_scale * self.trace_points[i + 1][0]
                pygame.draw.line(self.screen, self.trace_color, start_pos, end_pos, 2)
        
        # Draw the agent
        pygame.draw.circle(self.screen, self.agent_color, p, self.agent_r, self.agent_w)

        # Update the display and limit to 20 frames per second
        pygame.display.flip()
        self.clock.tick(50)