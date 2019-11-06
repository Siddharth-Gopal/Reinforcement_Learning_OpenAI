# import tensorflow-gpu as tf
from QNetworkKeras import QNetworkKerasMountatin
from ReplayMemory import ReplayMemory
import gym
import math
import random
import numpy as np
import keyboard
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
import time


# possible actions for the env are: ['NOOP', 'FIRE', "RIGHT", 'LEFT']
class DQAgent():

    def __init__(self):

        # TODO: Set all the parameters for configuring the agent over here.
        self.save_model_as = 'modeltest.h5'
        self.load_model_from = 'modeltest.h5'  # should not be empty if self.use_previous_model = True
        self.use_previous_model = True
        self.Qnet = QNetworkKerasMountatin()

        self.save_model_every = 50  # number of episodes to save the

        self.max_steps = 2000
        self.num_actions = 3
        self.render = True
        self.gamma = 0.97  # Gamma value for calculating td(0) target
        self.max_episodes = 1500
        self.history_len = 4  # taken from (84,84,4)
        self.update_target_every = 5  # number of episodes to update the target network in
        self.train_model_every = 1500  # number of trials to train the behaviour model in
        self.batch_size = 16  # batch size for training the behaviour model
        self.stats_every = 5  # number of episodes to take the stats in, to produce a graph at the end
        self.stats = {'episode': [], 'avg': [], 'max': [], 'min': [], 'avg_len': [], 'epsilon': []}
        self.key_control = {'left': 0, 'stay': 1, 'right': 2}

        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.LAMBDA = 0.0001

        # epsilon value will exponentially decay in EPS_DECAY number of episodes
        self.EPS_DECAY = 10000
        self.EPS_BASE = self.EPS_END / self.EPS_START


    def choose_action(self, state, trial_num):

        # epsilon thresh will exponentially decay till self.EPS_DECAY and then stay constant at self.EPS_END
        # if episode_num < self.EPS_DECAY:
        #     epsilon_thresh = self.EPS_START * math.pow(self.EPS_BASE, (episode_num / self.EPS_DECAY))
        # else:
        #     epsilon_thresh = self.EPS_END

        # Epsilon value will decay with every action taken exponentially from 1.0 to 0.01
        epsilon_thresh = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-self.LAMBDA * trial_num)

        epsilon = random.random()

        if epsilon < epsilon_thresh:
            # For Debugging
            random_choice = True
            return random.randint(0, self.num_actions - 1), epsilon_thresh, random_choice

        else:
            # For Debugging
            random_choice = False

            state = np.expand_dims(state, axis=0)
            action_values, q_max, max_q_ind = self.Qnet.pred_q_values(state, 'behaviour')
            return int(max_q_ind), epsilon_thresh, random_choice




    def train_model_mountain(self):
        if len(self.replay_mem.memory) < self.batch_size:
            print('Not enough samples to train')
        else:
            # Generate self.batch_size random numbers from length of replay_mem
            # Transitions to train on
            indices_to_train = np.random.randint(len(self.replay_mem.memory)-1, size=self.batch_size)
            current_states, actions, rewards, future_states, terminations = self.replay_mem.get_sample_mountaincar(indices_to_train)

            current_qs, _, _ = self.Qnet.pred_q_values(current_states, 'behaviour')
            future_qs, _, _ = self.Qnet.pred_q_values(future_states, 'target')
            max_future_qs = np.max(future_qs, axis=1)
            for i in range(self.batch_size):
                if terminations[i]:
                    td_target = rewards[i]
                else:
                    td_target = rewards[i] + self.gamma*max_future_qs[i]

                current_qs[i][actions[i]] = td_target

            self.Qnet.behaviour_model.fit(x=current_states, y=current_qs, verbose=0)

    def run(self):

        self.env = gym.make('MountainCar-v0')
        self.replay_mem = ReplayMemory(50000)
        trial_num = 0

        for episode in range(self.max_episodes):
            state = self.env.reset()
            total_reward = 0
            if episode % self.update_target_every:
                self.Qnet.target_model.set_weights(self.Qnet.behaviour_model.get_weights())

            for step in range(self.max_steps):

                action, epsilon_thresh, _ = self.choose_action(state, trial_num)
                trial_num += 1
                self.env.render()

                # print('action chosen: ', action)
                # print(type(action))
                new_state, reward, termination, info = self.env.step(action=action)
                if new_state[0] >= 0.0:
                    reward += 10
                elif new_state[0] >= 0.25:
                    reward += 20
                elif new_state[0] >= 0.5:
                    reward += 100

                total_reward+=reward

                # Adding state, action, reward, new_state, termination
                self.replay_mem.memory.append([state, action, reward, new_state, termination])

                # This will take self.batch_size random samples, find out the td target for each transition
                # and then use the entire batch to train the behaviour model
                self.train_model_mountain()

                state = new_state

                if termination:
                    self.env.close()
                    print('episode num: ', episode, ' Total Reward: ', total_reward, 'Epsilon: ', epsilon_thresh)
                    break












agent = DQAgent()
agent.run()
