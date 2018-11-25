import numpy as np
import random as rand
from time import time


class QLearner:
    '''A generic implementation of Q-Learning and Dyna-Q'''

    def __init__(self,
                 num_states,
                 num_actions,
                 obstacle_mask,
                 terminal_mask,
                 learning_rate,
                 init_q=0.1,
                 discount_rate=1.0,
                 random_action_prob=0.5,
                 random_action_decay_rate=0.99,
                 dyna_iterations=0):

        self._num_states = num_states
        self._num_actions = num_actions
        self._obstacle_mask = obstacle_mask.flatten()
        self._terminal_mask = terminal_mask.flatten()
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._random_action_prob = random_action_prob
        self._random_action_decay_rate = random_action_decay_rate
        self._dyna_iterations = dyna_iterations

        self._experiences = []

        # Initialize Q to small random values.
        self._Q = np.zeros((num_states, num_actions), dtype=np.float)
        self._Q += np.random.normal(0, init_q, self._Q.shape)

    def initialize(self, state):
        '''Set the initial state and return the learner's first action'''
        self._decide_next_action(state)
        self._stored_state = state
        return self._stored_action

    def learn(self, initial_state, gw, iterations=100, title=None):
        '''Iteratively experience new states and rewards'''
        all_policies = np.zeros((self._num_states, iterations))
        all_utilities = np.zeros_like(all_policies)
        time_stamps = np.zeros(iterations)
        num_steps_ar = np.zeros(iterations)
        total_reward_ar = np.zeros(iterations)
        
        for i in range(iterations):
            print('Iteration: ', i)
            t0 = time()
            done = False
            self.initialize(initial_state)
            for j in range(iterations):
                state, reward, done = gw.generate_experience(self._stored_state, self._stored_action)

                self.experience(state, reward)

                if done:
                    break
                
            policy, utility = self.get_policy_and_utility()
            all_policies[:, i] = policy
            all_utilities[:, i] = utility

            time_stamps[i] = time() - t0

            start_state = gw.grid_indices_to_coordinates(initial_state)
            num_steps, total_reward = gw.follow_policy(start_state, policy.reshape(gw.reward_grid.shape))
            num_steps_ar[i] = num_steps
            total_reward_ar[i] = total_reward

            """if title:
                dim = title.split('/')[0].split('x')
                utility = utility.reshape((int(dim[0]), int(dim[1])))
                policy = policy.reshape((int(dim[0]), int(dim[1])))
                gw.plot_policy(utility, policy, 'frames/' + title + '-' + str(i))"""

        return all_policies, all_utilities, time_stamps, num_steps_ar, total_reward_ar

    def experience(self, state, reward):
        '''The learner experiences state and receives a reward'''
        self._update_Q(self._stored_state, self._stored_action, state, reward)

        # determine an action and update the current state
        self._decide_next_action(state)
        self._stored_state = state

        self._random_action_prob *= self._random_action_decay_rate

        return self._stored_action

    def get_policy_and_utility(self):
        policy = np.argmax(self._Q, axis=1)
        utility = np.max(self._Q, axis=1)
        return policy, utility

    def _update_Q(self, s, a, s_prime, r):
        best_reward = self._Q[s_prime, self._find_best_action(s_prime)]
        if self._terminal_mask[s_prime]:
            best_reward = 0
        self._Q[s, a] *= (1 - self._learning_rate)
        self._Q[s, a] += (self._learning_rate
                          * (r + self._discount_rate * best_reward))

    def _decide_next_action(self, state):
        if rand.random() <= self._random_action_prob:
            self._stored_action = rand.randint(0, self._num_actions - 1)
        else:
            self._stored_action = self._find_best_action(state)

    def _find_best_action(self, state):
        return int(np.argmax(self._Q[state, :]))
