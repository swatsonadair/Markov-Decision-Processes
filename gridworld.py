import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2


class GridWorldMDP:

    # up, right, down, left
    _direction_deltas = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]
    _num_actions = len(_direction_deltas)

    def __init__(self,
                 start,
                 reward_grid,
                 terminal_mask,
                 obstacle_mask,
                 action_probabilities,
                 no_action_probability):

        self._start = start
        self._reward_grid = reward_grid
        self._terminal_mask = terminal_mask
        self._obstacle_mask = obstacle_mask
        self._T = self._create_transition_matrix(
            action_probabilities,
            no_action_probability,
            obstacle_mask
        )

    @property
    def shape(self):
        return self._reward_grid.shape

    @property
    def size(self):
        return self._reward_grid.size

    @property
    def reward_grid(self):
        return self._reward_grid

    def run_value_iterations(self, discount=1.0,
                             iterations=10,
                             title=None):
        utility_grids, policy_grids, time_stamps, num_steps_ar, total_reward_ar = self._init_utility_policy_storage(iterations)

        utility_grid = np.zeros_like(self._reward_grid)
        for i in range(iterations):
            print('Iteration: ', i)
            t0 = time()
            utility_grid = self._value_iteration(utility_grid=utility_grid)
            policy_grids[:, :, i] = self.best_policy(utility_grid)
            time_stamps[i] = time() - t0
            utility_grids[:, :, i] = utility_grid

            num_steps, total_reward = self.follow_policy(self._start, policy_grids[:, :, i])
            num_steps_ar[i] = num_steps
            total_reward_ar[i] = total_reward

            if title:
                self.plot_policy(utility_grid, policy_grids[:, :, i], 'frames/' + title + '-' + str(i))

        return policy_grids, utility_grids, time_stamps, num_steps_ar, total_reward_ar

    def run_policy_iterations(self, discount=1.0,
                              iterations=10,
                              title=None):
        utility_grids, policy_grids, time_stamps, num_steps_ar, total_reward_ar = self._init_utility_policy_storage(iterations)

        policy_grid = np.random.randint(0, self._num_actions,
                                        self.shape)
        utility_grid = self._reward_grid.copy()

        for i in range(iterations):
            print('Iteration: ', i)
            t0 = time()
            policy_grid, utility_grid = self._policy_iteration(
                policy_grid=policy_grid,
                utility_grid=utility_grid
            )
            time_stamps[i] = time() - t0
            policy_grids[:, :, i] = policy_grid
            utility_grids[:, :, i] = utility_grid

            num_steps, total_reward = self.follow_policy(self._start, policy_grids[:, :, i])
            num_steps_ar[i] = num_steps
            total_reward_ar[i] = total_reward

            if title:
                self.plot_policy(utility_grid, policy_grid, 'frames/' + title + '-' + str(i))

        return policy_grids, utility_grids, time_stamps, num_steps_ar, total_reward_ar

    def generate_experience(self, current_state_idx, action_idx):
        sr, sc = self.grid_indices_to_coordinates(current_state_idx)
        next_state_probs = self._T[sr, sc, action_idx, :, :].flatten()

        next_state_idx = np.random.choice(np.arange(next_state_probs.size),
                                          p=next_state_probs)

        return (next_state_idx,
                self._reward_grid.flatten()[next_state_idx],
                self._terminal_mask.flatten()[next_state_idx])

    def grid_indices_to_coordinates(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coordinates_to_indices(self, coordinates=None):
        if coordinates is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coordinates, self.shape)

    def best_policy(self, utility_grid):
        M, N = self.shape
        return np.argmax((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
                         .sum(axis=-1).sum(axis=-1), axis=2)

    def _init_utility_policy_storage(self, depth):
        M, N = self.shape
        utility_grids = np.zeros((M, N, depth))
        policy_grids = np.zeros_like(utility_grids)
        time_stamps = np.zeros(depth)
        num_steps_ar = np.zeros(depth)
        total_reward_ar = np.zeros(depth)
        return utility_grids, policy_grids, time_stamps, num_steps_ar, total_reward_ar

    def _create_transition_matrix(self,
                                  action_probabilities,
                                  no_action_probability,
                                  obstacle_mask):
        M, N = self.shape

        T = np.zeros((M, N, self._num_actions, M, N))

        r0, c0 = self.grid_indices_to_coordinates()

        T[r0, c0, :, r0, c0] += no_action_probability

        for action in range(self._num_actions):
            for offset, P in action_probabilities:
                direction = (action + offset) % self._num_actions

                dr, dc = self._direction_deltas[direction]
                r1 = np.clip(r0 + dr, 0, M - 1)
                c1 = np.clip(c0 + dc, 0, N - 1)

                temp_mask = obstacle_mask[r1, c1].flatten()
                r1[temp_mask] = r0[temp_mask]
                c1[temp_mask] = c0[temp_mask]

                T[r0, c0, action, r1, c1] += P

        terminal_locs = np.where(self._terminal_mask.flatten())[0]
        T[r0[terminal_locs], c0[terminal_locs], :, :, :] = 0

        obstacle_locs = np.where(self._obstacle_mask.flatten())[0]
        T[r0[obstacle_locs], c0[obstacle_locs], :, :, :] = 0
        return T

    def _value_iteration(self, utility_grid, discount=1.0):
        out = np.zeros_like(utility_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self._calculate_utility((i, j),
                                                    discount,
                                                    utility_grid)
        return out

    def _policy_iteration(self, utility_grid,
                          policy_grid, discount=1.0):
        r, c = self.grid_indices_to_coordinates()

        M, N = self.shape

        utility_grid = (
            self._reward_grid +
            discount * ((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
                        .sum(axis=-1).sum(axis=-1))[r, c, policy_grid.flatten()]
            .reshape(self.shape)
        )

        utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]
        utility_grid[self._obstacle_mask] = self._reward_grid[self._obstacle_mask]

        return self.best_policy(utility_grid), utility_grid

    def _calculate_utility(self, loc, discount, utility_grid):
        if self._terminal_mask[loc] or self._obstacle_mask[loc]:
            return self._reward_grid[loc]
        row, col = loc
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, :, :] * utility_grid,
                       axis=-1),
                axis=-1)
        ) + self._reward_grid[loc]

    def follow_policy(self, start, policy_grid):

        num_steps_ar = []
        total_reward_ar = []

        for i in range(1000):

            loc_idx = self.grid_coordinates_to_indices(start)
            next_state_idx, reward, terminal_mask = self.generate_experience(loc_idx, int(policy_grid[start]))
            loc = self.grid_indices_to_coordinates(next_state_idx)

            num_steps = 0
            total_reward = 0

            while not terminal_mask and num_steps < self.size:
                next_state_idx, reward, terminal_mask = self.generate_experience(next_state_idx, int(policy_grid[loc]))
                num_steps += 1
                total_reward += reward
                loc = self.grid_indices_to_coordinates(next_state_idx)

            num_steps_ar.append(num_steps)
            total_reward_ar.append(total_reward)

        num_steps = np.mean(num_steps_ar)
        total_reward = np.mean(total_reward_ar)

        return num_steps, total_reward

    def plot_policy(self, utility_grid, policy_grid=None, title=None):
        if policy_grid is None:
            policy_grid = self.best_policy(utility_grid)
        markers = "^>v<"
        marker_size = 100 // np.max(policy_grid.shape)
        marker_edge_width = marker_size // 20
        marker_fill_color = 'w'
        marker_edge_color = 'k'

        if self.size < 20:
            dpi = 100
            fontsize = 'small'
            x_offset = 0.15
            y_offset = 0.3
        else:
            dpi = 200
            fontsize = 'xx-small'
            x_offset = 0.3
            y_offset = 0.4

        no_action_mask = self._terminal_mask | self._obstacle_mask

        utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]

        utility_normalized = (utility_grid - utility_grid.min()) / \
                             (utility_grid.max() - utility_grid.min())

        utility_normalized = (255* (1 - utility_normalized)).astype(np.uint8)

        utility_rgb = cv2.applyColorMap(utility_normalized, cv2.COLORMAP_JET)
        for i in range(3):
            channel = utility_rgb[:, :, i]
            channel[self._obstacle_mask] = 0

        if utility_grid[self._start] == 0:
            for y in range(self.shape[0]):
                for x in range(self.shape[1]):
                    if not self._obstacle_mask[y, x] and not self._terminal_mask[y, x]:
                        utility_rgb[y, x] = [255, 255, 255]

        plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')

        if utility_grid[self._start] != 0:
            for i, marker in enumerate(markers):
                y, x = np.where((policy_grid == i) & np.logical_not(no_action_mask))
                plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                         color=marker_fill_color, markeredgecolor=marker_edge_color)
                for j in range(len(x)):
                    plt.annotate(round(utility_grid[y[j], x[j]], 2), xy=(x[j], y[j]), xytext=(x[j] - x_offset, y[j] + y_offset), fontsize=fontsize)
            
            y, x = np.where(self._terminal_mask)
            plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color, markeredgecolor=marker_edge_color)

        else:
            y, x = self._start
            plt.plot(x, y, 'o', ms=marker_size * 2, mew=marker_edge_width, color='grey', markeredgecolor=marker_edge_color)

        plt.margins(y=0)
        plt.margins(x=0)

        tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
        tick_step = np.max(policy_grid.shape)/8
        best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
        tick_step = tick_step_options[best_option]
        plt.xticks(np.arange(0, policy_grid.shape[1] - 0.5, tick_step))
        plt.yticks(np.arange(0, policy_grid.shape[0] - 0.5, tick_step))
        plt.xlim([-0.5, policy_grid.shape[0]-0.5])
        plt.xlim([-0.5, policy_grid.shape[1]-0.5])

        plt.title(title)
        filename = title.replace(' ', '-') + '-Map'
        plt.savefig('/' . join(['output', filename]), dpi=dpi, bbox_inches="tight")
        plt.close("all")
