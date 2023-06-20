import gym
import pygame
from gym import spaces
from gym.utils import seeding
import numpy as np

class PointMassRandom(gym.Env):
    def __init__(self, 
                 risk_prob = 0.9, 
                 risk_penalty = 50, 
                 num_obstacles = 1,
                 random_obstacle = True,
                 random_goal = True):
        # Agent Settings
        self.v_max = 0.02                           # Maximum velocity of agent
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

        self.random_goal = random_goal
        if self.random_goal:
            self.goal = np.array([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])        # Goal position
        else:
            self.goal = np.array([0.1, 0.1])
        self.d_goal = 0.03                      # Distance threshold for the goal
        
        self.local_grid_num = 3                  # Number of local grid cells
        self.local_grid_radius = 0.01            # Radius of local grid representation

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2 + 2 + self.local_grid_num * self.local_grid_num, ),
            dtype=np.float32
        )

        self.risk_prob = risk_prob              # Probability of encountering risk situation
        self.risk_penalty = risk_penalty        # Penalty of risk
        self.random_obstacle = random_obstacle
        if self.random_obstacle:
            self.num_obstacles = num_obstacles
            self.obstacle_rectangles = self.generate_obstacles()
        else:
            self.num_obstacles = 2
            self.obstacle_rectangles = [            # List of obstacle rectangles
                {"tl": np.array([0.15, 0.6]), "w": 0.35, "h": 0.2},
                {"tl": np.array([0.6, 0.15]), "w": 0.2, "h": 0.35},
            ]

        # Render Settings
        self.pygame_init = False

        self.screen_size = [600, 600]
        self.screen_scale = 600
        self.background_color = [223, 220, 230]

        self.obstacle_color = [36, 39, 46]
        self.goal_color = [0, 255, 0]
        self.agent_color = [0, 0, 0]
        self.agent_r = 6                        # Radius of agent
        self.agent_w = 3                        # Width of agent

        self.trace_points = []                  # List of (position, velocity) tuples for the current episode
        self.episode_traces = []                # List of trace_points lists for all episodes
        self.trace_color = [255, 0, 0]          # Color of trace

        self.local_normal_color = [199, 140, 93]
        self.local_obstacle_color = [40, 138, 203]
        self.local_goal_color = [106, 179, 113]

    def generate_obstacles(self):
        def is_inside_rectangle(rect, state):
            in_x_range = (rect["tl"][0] <= state[0]) & (state[0] <= rect["tl"][0] + rect["w"])
            in_y_range = (rect["tl"][1] <= state[1]) & (state[1] <= rect["tl"][1] + rect["h"])
            return in_x_range & in_y_range
        
        obstacle_rectangles = []

        for i in range(self.num_obstacles):
            valid_obstacle = False

            while not valid_obstacle:
                w = np.random.uniform(0.1, 0.3)
                h = np.random.uniform(0.1, 0.3)
                tl = np.random.uniform(0.1, 0.9 - max(w, h), size=(2,))
                rect = {"tl": tl, "w": w, "h": h}

                valid_start_goal = (not is_inside_rectangle(rect, self.init_pos) and not is_inside_rectangle(rect, self.goal))
                
                if valid_start_goal:
                    valid_obstacle = True
                    for existing_rect in obstacle_rectangles:
                        intersection_area = (
                            max(
                                0, 
                                min(existing_rect["tl"][0] + existing_rect["w"], rect["tl"][0] + rect["w"])
                                - max(existing_rect["tl"][0], rect["tl"][0])
                            ) 
                            * max(
                                0, 
                                min(existing_rect["tl"][1] + existing_rect["h"], rect["tl"][1] + rect["h"])
                                - max(existing_rect["tl"][1], rect["tl"][1])
                            )
                        )
                        
                        if intersection_area > 0:
                            valid_obstacle = False
                            break

            obstacle_rectangles.append(rect)
        
        return obstacle_rectangles
    
    def local_grid_representation(self, position):
        grid = np.zeros((self.local_grid_num, self.local_grid_num), dtype=np.float32)
        cell_size = 2 * self.local_grid_radius

        for y in range(self.local_grid_num):
            for x in range(self.local_grid_num):
                cell_top_left = position - self.local_grid_radius + np.array([x * cell_size, y * cell_size]) # 
                cell_bottom_right = cell_top_left + np.array([cell_size, cell_size])

                for rect in self.obstacle_rectangles:
                    rect_top_left = rect["tl"]
                    rect_bottom_right = rect_top_left + np.array([rect["w"], rect["h"]])

                    # Check if the cell is inside the current obstacle
                    if (cell_top_left[0] < rect_bottom_right[0] and cell_bottom_right[0] > rect_top_left[0] and
                        cell_top_left[1] < rect_bottom_right[1] and cell_bottom_right[1] > rect_top_left[1]):
                        grid[y, x] = 1
                        break

                # Check if the goal is inside the current cell
                if (self.goal[0] >= cell_top_left[0] and self.goal[0] <= cell_bottom_right[0] and
                    self.goal[1] >= cell_top_left[1] and self.goal[1] <= cell_bottom_right[1]):
                    grid[y, x] = 2
        return grid.flatten()

    def update_obstacle_number(self, new_obstacle_number):
        self.num_obstacles = new_obstacle_number
        self.obstacle_rectangles = self.generate_obstacles()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

    def reset(self, eval=False):
        if self.random_goal:
            self.goal = np.array([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])        # Goal position
        else:
            self.goal = np.array([0.1, 0.1])

        # Generate new obstacles if random_obstacle is True
        if self.random_obstacle:
            self.obstacle_rectangles = self.generate_obstacles()

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
        # self.state = np.array(list(self.init_pos) + list(self.goal))
        self.state = np.concatenate([self.init_pos, self.goal, self.local_grid_representation(self.init_pos)])
        return np.array(self.state)
        
    # Help function to check if the state is safe.    
    def is_safe(self, state):
        def is_inside_rectangle(rect, state):
            in_x_range = (rect["tl"][0] <= state[0]) & (state[0] <= rect["tl"][0] + rect["w"])
            in_y_range = (rect["tl"][1] <= state[1]) & (state[1] <= rect["tl"][1] + rect["h"])
            return in_x_range & in_y_range

        if len(state.shape) == 1:
            safe = True
            for rect in self.obstacle_rectangles:
                if is_inside_rectangle(rect, state):
                    safe = False
                    break
            return safe
        elif len(state.shape) == 2:
            safe = np.ones(state.shape[0], dtype=bool)
            for rect in self.obstacle_rectangles:
                inside_rect = is_inside_rectangle(rect, state)
                safe &= ~inside_rect
            return safe.astype(float)

    def step(self, action):
        # clip the action to the action space
        action = np.clip(action, -self.v_max, self.v_max)
        assert self.action_space.contains(action)

        # Save the current position and velocity in the trace_points list
        self.trace_points.append((self.state[:2].copy(), action.copy()))

        # compute the distance to the goal
        d_goal = np.linalg.norm(self.state[2:4] - self.state[:2])
        
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
        self.state[:4] = np.clip(self.state[:4], self.low_state, self.high_state)
        self.state = np.concatenate([self.state[:4], self.local_grid_representation(self.state[:2])])

        return np.array(self.state), reward, done, {'cost':cost}

    def render(self, show_local_grid = False, show_hist_trace = False):
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw the background
        self.screen.fill(self.background_color)

        # Draw the obstacle
        for rect in self.obstacle_rectangles:
            tl, w, h = (self.screen_scale * rect["tl"]).astype(int), int(self.screen_scale * rect["w"]), int(self.screen_scale * rect["h"])
            pygame.draw.rect(self.screen, self.obstacle_color, pygame.Rect(tl[0], tl[1], w, h))

        # Draw the goal
        pygame.draw.circle(self.screen, self.goal_color, (self.screen_scale * self.goal).astype(int), int(self.screen_scale * self.d_goal))

        # Draw all the previous episode traces
        if show_hist_trace:
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

        # Get the agent position on the screen
        agent_pos = self.state[:2]
        agent_pos_on_screen = (self.screen_scale * agent_pos).astype(int)
        
        # Draw the local grid representation
        if show_local_grid:
            local_grid = self.local_grid_representation(self.state[:2]).reshape(self.local_grid_num, self.local_grid_num)
            cell_size = 2 * self.local_grid_radius * self.screen_scale
            grid_top_left = agent_pos_on_screen - int(self.screen_scale * (self.local_grid_radius * self.local_grid_num))

            for y in range(self.local_grid_num):
                for x in range(self.local_grid_num):
                    rect_x = grid_top_left[0] + x * cell_size
                    rect_y = grid_top_left[1] + y * cell_size
                    rect = pygame.Rect(rect_x, rect_y, cell_size, cell_size)
                    if local_grid[y, x] == 1:
                        pygame.draw.rect(self.screen, self.local_obstacle_color, rect)
                    elif local_grid[y, x] == 2:
                        pygame.draw.rect(self.screen, self.local_goal_color, rect)
                    else:
                        pygame.draw.rect(self.screen, self.local_normal_color, rect)

        # Draw the agent
        pygame.draw.circle(self.screen, self.agent_color, agent_pos_on_screen, self.agent_r, self.agent_w)

        # Update the display and limit to 20 frames per second
        pygame.display.flip()
        self.clock.tick(24)