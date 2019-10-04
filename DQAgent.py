# import tensorflow-gpu as tf
from QNetworkKeras import QNetworkKeras
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


# possible actions for the env are: ['NOOP', 'FIRE', "RIGHT", 'LEFT']
class DQAgent():

    def __init__(self):

        #TODO: Set all the parameters for configuring the agent over here.
        self.Qnet = QNetworkKeras()
        self.use_previous_model = False
        self.save_model_every = 50  # number of episodes to save the
        self.save_model_as = 'model1.h5'
        self.load_model_from = 'model1.h5' # should not be empty if self.use_previous_model = True
        self.max_steps = 2000
        self.num_actions = 4
        self.gamma = 0.7        # Gamma value for calculating td(0) target
        self.max_episodes = 2000
        self.history_len = 4  # taken from (84,84,4)
        self.update_target_every = 10  # number of episodes to update the target network in
        self.train_model_every = 10000  # number of trials to train the behaviour model in
        self.batch_size = 16 # batch size for training the behaviour model
        self.stats_every = 15 # number of episodes to take the stats in, to produce a graph at the end
        self.stats = {'episode': [], 'avg': [], 'max': [], 'min': [], 'avg_len': [], 'epsilon': []}

        self.EPS_START = 0.7
        self.EPS_END = 0.05


        # epsilon value will exponentially decay in EPS_DECAY number of episodes
        self.EPS_DECAY = 1200
        self.EPS_BASE = self.EPS_END / self.EPS_START

    def load_prev_weights(self):
        if self.use_previous_model:
            self.Qnet.behaviour_model.load_weights(self.load_model_from)
            self.Qnet.target_model.load_weights(self.load_model_from)

    def choose_action(self, action_values, episode_num):

        # epsilon thresh will exponentially decay till self.EPS_DECAY and then stay constant at self.EPS_END
        if episode_num < self.EPS_DECAY:
            epsilon_thresh = self.EPS_START * math.pow(self.EPS_BASE, (episode_num/self.EPS_DECAY))
        else:
            epsilon_thresh = self.EPS_END

        epsilon = random.random()

        if epsilon < epsilon_thresh:
            # For Debugging
            random_choice = True
            return random.randint(0, self.num_actions-1), epsilon_thresh, random_choice

        else:
            # For Debugging
            random_choice = False

            max_q_ind = np.argmax(action_values)
            return max_q_ind, epsilon_thresh, random_choice

    def slicing(self, l, n):
        "Yield successive n-sized chunks from l."
        for i in range(0, len(l), n):
            yield l[i:i + n]


    # previous_frames is the number of frames starting from your right you want to use to train your model
    def train_model(self, replay_mem, previous_frames):

        # We want to avoid first few frames of our episode
        lowest_ind = len(replay_mem.frames) - previous_frames
        if lowest_ind < 4:
            lowest_ind = 4

        # highest ind must be second last frame because we will find td target of second last frame with the last frame
        highest_ind = len(replay_mem.frames) - 1
        if (highest_ind - lowest_ind) < 5:
            return

        # TODO: create a list from lowest_ind, to highest_ind
        # scramble the list, first batch size from it, pass it to get_samples, and use that

        shuffled_frames_to_train = np.arange(lowest_ind, highest_ind)
        np.random.shuffle(shuffled_frames_to_train)

        # print('shuffled frames: ', shuffled_frames_to_train)
        # print('replay_mem size: ', len(replay_mem.frames))
        # shuffled_frame_batches = list(self.slicing(shuffled_frames_to_train, self.batch_size))

        # frame indices is a list of shuffled frame indices
        # for frame_indices in shuffled_frame_batches:
        current_states, current_info, future_states = replay_mem.get_sample(shuffled_frames_to_train)
        print(current_states.shape)
        current_qs = self.Qnet.behaviour_model.predict(current_states / 255, batch_size=8)  # shape=batch_size, 4
        future_qs = self.Qnet.target_model.predict(future_states / 255, batch_size=8)  # shape=batch_size, 4
        max_future_qs = np.max(future_qs, axis=1)

        # change the q value of the state action pair to the td target
        for i in range(len(shuffled_frames_to_train)):

            # current info consists of [rewards, actions, terminations] for current_states
            action = current_info[i][1]
            if current_info[i][2]:
                target = current_info[i][0]
            else:
                target = current_info[i][0] + self.gamma*max_future_qs[i]

            current_qs[i][action] = target

        self.Qnet.behaviour_model.fit(x=current_states, y=current_qs, batch_size=self.batch_size)


    def play(self):
        env = gym.make('Breakout-v0')
        self.load_prev_weights()


        # TODO: make replay_mem an instance variable rather than using it as an argument in train_model

        # maxlen of ReplayMemory must be greater than self.train_model_every
        replay_mem = ReplayMemory(1500)
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

            #add initial frames to the replay memory with action 0
            for _ in range(self.history_len-1):
                new_frame, reward, termination, _ = env.step(action=0)

                # The reward, action and termination are corresponding to the previous frame and not the
                # new frame taken from env.step
                replay_mem.add(frame, 0, 0, 0)
                frame = new_frame


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

                # TODO: is the frame, new_frame structure redundant, cant you do with just one?
                # Add only the frame to replay_mem.frames
                replay_mem.frames.append(replay_mem.preprocess_image(frame))
                # Now at this point the replay memory has 1 more frame than others

                state = replay_mem.get_latest_state()  # gives the state using the last 4 frames
                action_values, q_max, max_q_ind = self.Qnet.pred_q_values(state, 'behaviour')

                action, epsilon, random_choice = self.choose_action(action_values, episode_num=episode)

                new_frame, reward, termination, info = env.step(action=action)
                prev_actions.append(action)

                # Used if you want to analyse the current frame
                if keyboard.is_pressed('a'):
                    print('previous 20 actions are: ', prev_actions)
                    print('type: ', type(action_values))
                    print('random choice is : ', random_choice)
                    print('argmax: ', np.argmax(action_values) )
                    print('action values: ', action_values)
                    print('epsilon threshhold: ', epsilon)
                    print('action chosen: ', action)
                    plt.imshow(frame)
                    plt.show()

                # Add the other info corresponding to the frame added earlier
                replay_mem.others.append([reward, action, termination])
                frame = new_frame
                episode_reward += reward




                # Train network, and save the weights
                if trial_num % self.train_model_every == 0 and len(replay_mem.frames)>8:
                    self.train_model(replay_mem, self.train_model_every)
                    self.Qnet.behaviour_model.save(self.save_model_as)

                if termination:
                    # if episode has terminated we only give the remaining frames on which the agent has not trained on
                    self.train_model(replay_mem, step_num%self.train_model_every)
                    replay_mem.clear_replay_memory()
                    print('episode length: ', step_num)
                    ep_lengths.append(step_num)
                    ep_rewards.append(episode_reward)
                    break

                # Used if you want to plot the current stats so far
                if keyboard.is_pressed('p'):
                    fig, ax = plt.subplots(3, 1)

                    ax[0].plot(self.stats['episode'], self.stats['avg'], label="average rewards", color='b')
                    ax[0].plot(self.stats['episode'], self.stats['max'], label="max rewards", color='g')
                    ax[0].plot(self.stats['episode'], self.stats['min'], label="min rewards", color='r')

                    ax[1].plot(self.stats['episode'], self.stats['avg_len'], label="episode length", color='m')

                    ax[2].plot(self.stats['episode'], self.stats['epsilon'], label='epsilon thresh', color='c')

                    plt.legend()
                    plt.savefig('model1visual.jpg')
                    plt.show()

            else:
                print('Max 2000 steps done')
                ep_lengths.append(2000)
                ep_rewards.append(episode_reward)





            # TODO: this should come inside the train loop?. Save your target weights
            # Update target model
            if episode % self.update_target_every == 0:
                self.Qnet.target_model.set_weights(self.Qnet.behaviour_model.get_weights())

            if episode % self.stats_every == 0:
                avg_reward = sum(ep_rewards[len(ep_rewards)-self.stats_every:]) / self.stats_every
                avg_len = sum(ep_lengths[len(ep_lengths)-self.stats_every:]) / self.stats_every
                self.stats['episode'].append(episode)
                self.stats['avg'].append(avg_reward)
                self.stats['max'].append(max(ep_rewards[len(ep_rewards)-self.stats_every:]))
                self.stats['min'].append(min(ep_rewards[len(ep_rewards)-self.stats_every:]))
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
        plt.savefig('model1visual.jpg')
        plt.show()














agent = DQAgent()
agent.play()



