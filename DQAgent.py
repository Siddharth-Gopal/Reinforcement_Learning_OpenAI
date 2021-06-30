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
        self.env = gym.make('MountainCar-v0')
        self.replay_mem = ReplayMemory(50000)
        self.DoubleQLearning = True

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

    def load_prev_weights(self):
        if self.use_previous_model:
            self.Qnet.behaviour_model.load_weights(self.load_model_from)
            self.Qnet.target_model.load_weights(self.load_model_from)

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

    def display_state(self, state):
        subplot_num = 0

        for frame in range(4):
            plt.subplot(1, 4, subplot_num + 1)
            plt.imshow(state[:, :, subplot_num])
            subplot_num += 1

        plt.show()

        # TODO: Cant see the ball in terminal states

    # previous_frames is the number of frames starting from your right you want to use to train your model
    # sampling_length are the num of transitions(starting from right) that must be used to create the samples
    # sampling_length is used in imitation learning
    def train_model(self, replay_mem, sampling_length, sample_size, epochs=1):

        # print('training model')
        # We want to avoid first few frames of our episode
        lowest_ind = len(replay_mem.frames) - sampling_length
        if lowest_ind < 4:
            lowest_ind = 4

        # highest ind must be second last frame because we will find td target of second last frame with the last frame
        highest_ind = len(replay_mem.frames) - 1
        if (highest_ind - lowest_ind) < 5:
            print('Training did not occur, something is wrong')
            return

        # create a list from lowest_ind, to highest_ind
        frames_to_train = np.random.randint(low=lowest_ind, high=highest_ind, size=sample_size)

        # We dont want to train all the frames together as that will take up a lot of memory.
        # Instead we break up our shuffled_frames into chunks and train the chunks individually
        sliced_shuffled_frames = np.array_split(frames_to_train, 1)

        for counter, slice in enumerate(sliced_shuffled_frames):
            # frame indices is a list of shuffled frame indices
            # for frame_indices in shuffled_frame_batches:
            current_states, current_info, future_states = replay_mem.get_sample_mountaincar(slice)

            # #Debugging
            # print('This should not be true: \n')
            # print(np.array_equal(current_states[5][:, :, 0], current_states[5][:, :, 2]))
            # self.display_state(current_states[5])

            # print(current_states.shape)
            current_qs = self.Qnet.behaviour_model.predict(current_states, batch_size=8)  # shape=batch_size, 4

            # For Debugging
            # print('current qs: ', current_qs.shape)

            future_qs = self.Qnet.target_model.predict(future_states, batch_size=8)  # shape=batch_size, 4
            max_future_qs = np.max(future_qs, axis=1)

            # Debugging
            terminal_frames = []

            # change the q value of the state action pair to the td target
            for i in range(len(slice)):

                # current info consists of [rewards, actions, terminations] for current_states
                action = current_info[i][1]
                if current_info[i][2]:
                    target = current_info[i][0]
                    # Debugging
                    if counter == len(sliced_shuffled_frames) - 1:
                        terminal_frames.append(i)
                else:
                    target = current_info[i][0] + self.gamma * max_future_qs[i]

                # print('target: ', target)
                # print('action: ', action)
                # print('i: ', i)
                current_qs[i][int(action)] = target

            # if counter == len(sliced_shuffled_frames) - 1:
            #
            #     print('Last 10 terminal targets are: \n')
            #     k = 1
            #     for num in terminal_frames[-5:]:
            #         self.display_state(current_states[num])
            #         print(current_info[num])

            self.Qnet.behaviour_model.fit(x=current_states, y=current_qs, batch_size=16, epochs=epochs, verbose=0)

    def modify_reward(self, reward, x_value):
        # modified reward is reward + interpolated x value from 0 to 10
        # return reward + 10*((x_value+1.2)/1.8)

        if x_value >= 0.0:
            reward += 10

        if x_value >= 0.55:
            reward += 100

        return reward

    def play(self):
        env = gym.make('MountainCar-v0')
        self.load_prev_weights()

        # TODO: make replay_mem an instance variable rather than using it as an argument in train_model

        # maxlen of ReplayMemory must be greater than self.train_model_every,
        # because there are already 4 frames even when num_trial = 1
        replay_mem = ReplayMemory(50000)
        trial_num = 0
        ep_rewards = []
        ep_lengths = []

        # range starts from 1 to avoid control entering stats if loop
        for episode in range(1, self.max_episodes):

            frame = env.reset()
            # new_frame, reward, termination, _ = env.step(action=0)
            # replay_mem.add(frame, reward, 0, termination)
            # frame = new_frame

            # For Debugging
            prev_actions = deque(maxlen=20)

            # #add initial frames to the replay memory with action 0
            # for _ in range(self.history_len-1):
            #     new_frame, reward, termination, _ = env.step(action=1)
            #
            #     # The reward, action and termination are corresponding to the previous frame and not the
            #     # new frame taken from env.step
            #     replay_mem.add(frame, 0, 0, False)
            #     frame = new_frame

            # # Last action taken has been set to 'FIRE'. This way the env starts with the ball already fired
            # # and we can remove 'FIRE' from the choose_action method since it now serves no purpose
            # # but may still be chosen under exploration
            # new_frame, reward, termination, _ = env.step(action=1)
            # replay_mem.add(frame, 0, 1, 0)
            # frame = new_frame

            episode_reward = 0

            print('episode num: ', episode)
            for step_num in range(self.max_steps):
                trial_num += 1

                env.render()

                # if keyboard.is_pressed('e'):
                #     time.sleep(1)
                #     self.render = not self.render
                # if self.render:
                #     env.render()

                # TODO: is the frame, new_frame structure redundant, cant you do with just one?
                # Add only the frame to replay_mem.frames
                replay_mem.frames.append(frame)
                # Now at this point the replay memory has 1 more frame than others

                # state = replay_mem.get_latest_state()  # gives the state using the last 4 frames
                action_values, q_max, max_q_ind = self.Qnet.pred_q_values(frame, 'behaviour')

                action, epsilon, random_choice = self.choose_action(action_values, trial_num)

                new_frame, reward, termination, info = env.step(action=action)

                # Using x position as reward for the agent
                reward = self.modify_reward(reward, new_frame[0])

                # For Debugging
                prev_actions.append(action)

                if termination:
                    # Add the other info corresponding to the frame added earlier
                    replay_mem.others.append([reward, int(action), termination])
                    episode_reward += reward

                    # Debugging: train the model with 50 sample states after every step
                    # # Train network, and save the weights
                    # if trial_num % self.train_model_every == 0 and len(replay_mem.frames) > 1000:
                    #     self.train_model(replay_mem, len(replay_mem.frames), 50)


                    # if episode has terminated we only give the remaining frames on which the agent has not trained on
                    # self.train_model(replay_mem, step_num%self.train_model_every)
                    # replay_mem.clear_replay_memory()
                    print('episode reward: ', episode_reward)
                    ep_lengths.append(step_num)
                    ep_rewards.append(episode_reward)
                    self.train_model(replay_mem, len(replay_mem.frames), 50)
                    env.close()
                    break

                else:
                    # Add the other info corresponding to the frame added earlier
                    replay_mem.others.append([reward, int(action), termination])
                    frame = new_frame
                    episode_reward += reward
                    self.train_model(replay_mem, len(replay_mem.frames), 50)

                # Used if you want to analyse the current frame
                if keyboard.is_pressed('a'):
                    print('previous 20 actions are: ', prev_actions)
                    print('type: ', type(action_values))
                    print('random choice is : ', random_choice)
                    print('argmax: ', np.argmax(action_values))
                    print('action values: ', action_values)
                    print('epsilon threshhold: ', epsilon)
                    print('action chosen: ', action)

                    [print(replay_mem.others[-20 + i]) for i in range(20)]
                    # plt.imshow(frame)
                    # plt.show()

                if keyboard.is_pressed('i'):
                    time.sleep(1)

                    # def key2int(key_string):
                    #     return()
                    imit_counter = 10
                    while True:
                        imit_counter += 1
                        action = keyboard.read_key()

                        if keyboard.is_pressed('esc'):
                            break
                        try:
                            action = self.key_control[action]
                        except:
                            print('You pressed the wrong key')
                            continue

                        # # TODO: if no button is pressed than the action should be no operation, or else the key pressed must be used as action to
                        # # control the game for imitation learning. This can be achieved by using this
                        # # also have a look at keyboard_test .py
                        #
                        # keyboard.add_hotkey('left', key2int, args='left', timeout=0.1)
                        # keyboard.add_hotkey('right', key2int, args='right', timeout=0.1)
                        # keyboard.add_hotkey('space', key2int, args='space', timeout=0.1)

                        env.render()
                        new_frame, reward, termination, info = env.step(action)
                        reward = self.modify_reward(reward, new_frame[0])
                        print('action: ', action)
                        print('reward: ', reward)

                        replay_mem.frames.append(frame)
                        replay_mem.others.append([reward, int(action), termination])
                        frame = new_frame

                        trial_num += 1
                        # Train network, and save the weights
                        if trial_num % self.train_model_every == 0 and len(replay_mem.frames) > 8:
                            self.train_model(replay_mem, imit_counter, 500, epochs = 5)

                        if termination:
                            env.close()

                            break
                    break

                # # Debugging: train the model with 50 sample states after every step
                # # Train network, and save the weights
                # if trial_num % self.train_model_every == 0 and len(replay_mem.frames) > 8:
                #     self.train_model(replay_mem, len(replay_mem.frames), 50)

                # Used if you want to plot the current stats so far
                if keyboard.is_pressed('p'):
                    fig, ax = plt.subplots(3, 1)

                    ax[0].plot(self.stats['episode'], self.stats['avg'], label="average rewards", color='b')
                    ax[0].plot(self.stats['episode'], self.stats['max'], label="max rewards", color='g')
                    ax[0].plot(self.stats['episode'], self.stats['min'], label="min rewards", color='r')
                    ax[0].legend()

                    ax[1].plot(self.stats['episode'], self.stats['avg_len'], label="episode length", color='m')
                    ax[1].legend()

                    ax[2].plot(self.stats['episode'], self.stats['epsilon'], label='epsilon thresh', color='c')
                    ax[2].legend()

                    # plt.legend()
                    plt.savefig('model1visual.jpg')
                    plt.show()

            else:
                print('Max 2000 steps done')
                ep_lengths.append(2000)
                ep_rewards.append(episode_reward)

            # TODO: this should come inside the train loop?. Save your target weights
            # Update target model
            if episode % self.update_target_every == 0:
                print('updated target network')
                self.Qnet.target_model.set_weights(self.Qnet.behaviour_model.get_weights())

            if episode % self.save_model_every:
                self.Qnet.behaviour_model.save(self.save_model_as)

            if episode % self.stats_every == 0:
                avg_reward = sum(ep_rewards[len(ep_rewards) - self.stats_every:]) / self.stats_every
                avg_len = sum(ep_lengths[len(ep_lengths) - self.stats_every:]) / self.stats_every
                self.stats['episode'].append(episode)
                self.stats['avg'].append(avg_reward)
                self.stats['max'].append(max(ep_rewards[len(ep_rewards) - self.stats_every:], default=0))
                self.stats['min'].append(min(ep_rewards[len(ep_rewards) - self.stats_every:], default=0))
                self.stats['avg_len'].append(avg_len)
                self.stats['epsilon'].append(epsilon)

        fig, ax = plt.subplots(3, 1)

        ax[0].plot(self.stats['episode'], self.stats['avg'], label="average rewards", color='b')
        ax[0].plot(self.stats['episode'], self.stats['max'], label="max rewards", color='g')
        ax[0].plot(self.stats['episode'], self.stats['min'], label="min rewards", color='r')
        ax[0].legend()

        ax[1].plot(self.stats['episode'], self.stats['avg_len'], label="episode length", color='m')
        ax[1].legend()

        ax[2].plot(self.stats['episode'], self.stats['epsilon'], label='epsilon thresh', color='c')
        ax[2].legend()

        plt.legend()
        plt.savefig('model1viasual.jpg')
        plt.show()

    def train_model_mountain(self):
        if len(self.replay_mem.memory) < self.batch_size:
            print('Not enough samples to train')
        else:
            # Generate self.batch_size random numbers from length of replay_mem
            # Transitions to train on
            indices_to_train = np.random.randint(len(self.replay_mem.memory)-1, size=self.batch_size)
            current_states, actions, rewards, future_states, terminations = self.replay_mem.get_sample_mountaincar(indices_to_train)


            if self.DoubleQLearning:
                current_qs = self.Qnet.behaviour_model.predict(current_states)
                # print(current_qs)

                # future_beh_best_action has shape (batch_size,)
                future_behaviour_qs = self.Qnet.behaviour_model.predict(future_states)
                future_beh_best_action = np.argmax(future_behaviour_qs, axis=1)

                # shape (batch_size, num_actions)
                future_target_qs = self.Qnet.target_model.predict(future_states)

                DDQN_targets = np.copy(current_qs)
                for i, action in enumerate(future_beh_best_action):
                    if terminations[i]:
                        target = rewards[i]
                    else:
                        target = rewards[i] + self.gamma * future_target_qs[i][action]
                    DDQN_targets[i][actions[i]] = target

                self.Qnet.behaviour_model.fit(current_states, DDQN_targets, verbose=0)
                # print(absolute_errors)

            else:

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


        trial_num = 0
        ep_rewards = []
        ep_lengths = []

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
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

                episode_reward+=reward

                # Adding state, action, reward, new_state, termination
                self.replay_mem.memory.append([state, action, reward, new_state, termination])

                # This will take self.batch_size random samples, find out the td target for each transition
                # and then use the entire batch to train the behaviour model
                self.train_model_mountain()

                state = new_state

                if termination:
                    self.env.close()
                    ep_rewards.append(episode_reward)
                    ep_lengths.append(step)
                    print('episode num: ', episode, ' Total Reward: ', episode_reward, 'Epsilon: ', epsilon_thresh)
                    break

                # Used if you want to plot the current stats so far
                if keyboard.is_pressed('p'):
                    fig, ax = plt.subplots(3, 1)

                    ax[0].plot(self.stats['episode'], self.stats['avg'], label="average rewards", color='b')
                    ax[0].plot(self.stats['episode'], self.stats['max'], label="max rewards", color='g')
                    ax[0].plot(self.stats['episode'], self.stats['min'], label="min rewards", color='r')
                    ax[0].legend()

                    ax[1].plot(self.stats['episode'], self.stats['avg_len'], label="episode length", color='m')
                    ax[1].legend()

                    ax[2].plot(self.stats['episode'], self.stats['epsilon'], label='epsilon thresh', color='c')
                    ax[2].legend()

                    # plt.legend()
                    plt.savefig('model1visual.jpg')
                    plt.close()

            if episode % self.stats_every == 0:
                avg_reward = sum(ep_rewards[-self.stats_every:]) / self.stats_every
                avg_len = sum(ep_lengths[-self.stats_every:]) / self.stats_every
                self.stats['episode'].append(episode)
                self.stats['avg'].append(avg_reward)
                self.stats['max'].append(max(ep_rewards[-self.stats_every:], default=0))
                self.stats['min'].append(min(ep_rewards[-self.stats_every:], default=0))
                self.stats['avg_len'].append(avg_len)
                self.stats['epsilon'].append(epsilon_thresh)

            if episode % self.save_model_every:
                self.Qnet.behaviour_model.save(self.save_model_as)












agent = DQAgent()
agent.run()
