from gridworld import GridWorldMDP
from qlearn import QLearner

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

COLORS = ['.b-', '.g-', '.m-', '.y-']


def plot_convergence(utility_grids, policy_grids, title):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, COLORS[0], label='Change in Utility Values (SSE)')
    ax1.set_ylabel('Change in Utility Values (SSE)', color='b')
    ax1.set_xlabel('Iteration')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
    if max(policy_changes) > 10:
        scale = 5
    else:
        scale = 1
    ax2.plot(policy_changes, COLORS[1], label='# Changes in Best Policy')
    ax2.set_yticks(range(0, max(policy_changes) + 2, scale))
    ax2.set_ylabel('# Changes in Best Policy', color='g')
    ax2.set_xlabel('Iteration')

    plt.title(title)
    filename = title.replace(' ', '-') + '-Convergence'
    plt.savefig('/' . join(['output', filename]), dpi=100, bbox_inches="tight")
    plt.close("all")


def plot_time(results, title):

    x_axis = range(1, results.shape[1] + 1)

    plt.plot(x_axis, results[0].astype('float'), COLORS[0], label='Value Iteration')
    plt.plot(x_axis, results[1].astype('float'), COLORS[1], label='Policy Iteration')
    plt.plot(x_axis, results[2].astype('float'), COLORS[2], label='Q-Learning')
    
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Time (ms)')
    plt.title(title)

    # Save graph
    filename = title.replace(' ', '-')
    plt.legend(loc="best", borderaxespad=0.)
    plt.savefig('/' . join(['output', filename]), dpi=100, bbox_inches="tight")
    plt.close("all")


def plot_num_steps(results, title):

    x_axis = range(1, results.shape[1] + 1)
    plt.plot(x_axis, results[0].astype('float'), COLORS[0], label='Value Iteration')
    plt.plot(x_axis, results[1].astype('float'), COLORS[1], label='Policy Iteration')
    plt.plot(x_axis, results[2].astype('float'), COLORS[2], label='Q-Learning')
    
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Expected # Steps')
    plt.title(title)

    # Save graph
    filename = title.replace(' ', '-')
    plt.legend(loc="best", borderaxespad=0.)
    plt.savefig('/' . join(['output', filename]), dpi=100, bbox_inches="tight")
    plt.close("all")


def plot_reward(results, title):

    x_axis = range(1, results.shape[1] + 1)
    plt.plot(x_axis, results[0].astype('float'), COLORS[0], label='Value Iteration')
    plt.plot(x_axis, results[1].astype('float'), COLORS[1], label='Policy Iteration')
    plt.plot(x_axis, results[2].astype('float'), COLORS[2], label='Q-Learning')
    
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Expected Total Reward')
    plt.title(title)

    # Save graph
    filename = title.replace(' ', '-')
    plt.legend(loc="best", borderaxespad=0.)
    plt.savefig('/' . join(['output', filename]), dpi=100, bbox_inches="tight")
    plt.close("all")


class MDP:

    def __init__(self,
                 shape,
                 goal,
                 traps,
                 obstacles,
                 start,
                 default_reward,
                 goal_reward,
                 trap_reward,
                 iterations):

        self.shape = shape
        self.goal = goal
        self.traps = traps
        self.obstacles = obstacles
        self.start = start
        self.default_reward = default_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.iterations = iterations


    def solve(self):

        reward_grid = np.zeros(self.shape) + self.default_reward
        reward_grid[self.goal] = self.goal_reward

        coords = zip(*self.traps)
        trap_mask = sparse.coo_matrix((np.ones(len(coords[0])), coords), shape=self.shape, dtype=bool).toarray()
        reward_grid[trap_mask] = self.trap_reward

        coords = zip(*self.obstacles)
        obstacle_mask = sparse.coo_matrix((np.ones(len(coords[0])), coords), shape=self.shape, dtype=bool).toarray()
        reward_grid[obstacle_mask] = 0
        
        terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
        terminal_mask[self.goal] = True
        terminal_mask[trap_mask] = True

        gw = GridWorldMDP(start=self.start,
                          reward_grid=reward_grid,
                          obstacle_mask=obstacle_mask,
                          terminal_mask=terminal_mask,
                          action_probabilities=[
                              (-1, 0.1),
                              (0, 0.8),
                              (1, 0.1),
                          ],
                          no_action_probability=0.0)

        utility_grid = np.zeros(self.shape)
        gw.plot_policy(utility_grid, None, str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld')

        mdp_solvers = {'Value Iteration': gw.run_value_iterations,
                       'Policy Iteration': gw.run_policy_iterations}

        time_results = []
        steps_results = []
        reward_results = []

        for solver_name, solver_fn in mdp_solvers.items():
            print('Solving {}:' . format(solver_name))

            title = str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld - ' + solver_name
            policy_grids, utility_grids, time_stamps, num_steps, total_reward = solver_fn(iterations=self.iterations[0], discount=0.5, title=title)

            a = np.empty(self.iterations[1] - self.iterations[0])

            a.fill(time_stamps[-1])
            time_stamps = np.concatenate((time_stamps, a))
            time_results.append(time_stamps)

            a.fill(num_steps[-1])
            num_steps = np.concatenate((num_steps, a))
            steps_results.append(num_steps)

            a.fill(total_reward[-1])
            total_reward = np.concatenate((total_reward, a))
            reward_results.append(total_reward)

            #print(policy_grids[:, :, -1])
            #print(utility_grids[:, :, -1])

            gw.plot_policy(utility_grids[:, :, -1], None, title)
            plot_convergence(utility_grids, policy_grids, title)

        """for lr in [0.7, 0.8, 0.9]:
          for ra in [0.2, 0.5, 0.8]:
            for e in [.79, .89, .99]:"""

        ql = QLearner(num_states=(self.shape[0] * self.shape[1]),
                      num_actions=4,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      learning_rate=0.8,
                      discount_rate=0.975,
                      random_action_prob=0.5,
                      random_action_decay_rate=0.89,
                      dyna_iterations=0)

        print('Solving QLearning:')
        start_state = gw.grid_coordinates_to_indices(self.start)

        #title = str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld - Q Learning - ' + str(lr).replace('.', '') + str(ra).replace('.', '') + str(e).replace('.', '')

        title = str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld - Q Learning'

        iterations = self.iterations[1]
        flat_policies, flat_utilities, time_stamps, num_steps, total_reward = ql.learn(start_state, gw,
                                                 iterations=iterations,
                                                 title=str(self.shape[0]) + 'x' + str(self.shape[1]) + '/QL/' + title)

        new_shape = (gw.shape[0], gw.shape[1], iterations)
        ql_utility_grids = flat_utilities.reshape(new_shape)
        ql_policy_grids = flat_policies.reshape(new_shape)

        time_results.append(time_stamps)
        steps_results.append(num_steps)
        reward_results.append(total_reward)

        #print(ql_policy_grids[:, :, -1])
        #print(ql_utility_grids[:, :, -1])

        gw.plot_policy(ql_utility_grids[:, :, -1], ql_policy_grids[:, :, -1], title)
        plot_convergence(ql_utility_grids[:, :, 0:-2], ql_policy_grids[:, :, 0:-2], title)

        plot_time(np.array(time_results), str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld - Time')
        plot_num_steps(np.array(steps_results), str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld - # Steps')
        plot_reward(np.array(reward_results), str(self.shape[0]) + 'x' + str(self.shape[1]) + ' Gridworld - Reward')


def simple_gridworld():
    shape = (4, 4)
    goal = (0, -1)
    traps = [(1, 3)]
    obstacles = [(1, 1), (2, 1), (3, 3)]
    start = (3, 0)
    default_reward = -0.1
    goal_reward = 1
    trap_reward = -1
    iterations = [25, 250]

    mdp = MDP(shape, goal, traps, obstacles, start, default_reward, goal_reward, trap_reward, iterations)
    mdp.solve()


def complex_gridworld():
    shape = (12, 12)
    goal = (0, -1)
    traps = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 4), (11, 8), (10, 4), (10, 8)]
    obstacles = [(2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (3, 2), (3, 6), (4, 2), (4, 4), (4, 6), (4, 8), (5, 2), (5, 4), (5, 6), (5, 8), (6, 2), (6, 4), (6, 6), (6, 8), (7, 2), (7, 4), (7, 6), (7, 8), (8, 2), (8, 4), (8, 6), (8, 8), (9, 2), (9, 4), (9, 6), (9, 8), (10, 2), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 6)]
    start = (11, 0)
    default_reward = -0.1
    goal_reward = 10
    trap_reward = -10
    iterations = [100, 400]

    mdp = MDP(shape, goal, traps, obstacles, start, default_reward, goal_reward, trap_reward, iterations)
    mdp.solve()


if __name__ == '__main__':
    
    simple_gridworld()
    complex_gridworld()

    
