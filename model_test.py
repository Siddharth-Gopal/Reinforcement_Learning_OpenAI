# objective is to get the cart to the flag.
# for now, let's just move randomly:

import gym
import numpy as np
from keras.models import load_model
import keyboard
import matplotlib.pyplot as plt
import sys

env = gym.make('MountainCar-v0')
model_name = sys.argv[1]
episodes_to_run = int(sys.argv[2])
Qnet = load_model(model_name)

def choose_action(state):
    state = np.expand_dims(state, axis=0)
    action_values = Qnet.predict(state)
    max_q_ind = np.argmax(action_values, axis=1)
    return int(max_q_ind)

def run():
    trial_num = 0
    ep_rewards = []
    ep_lengths = []

    for episode in range(episodes_to_run):
        state = env.reset()
        episode_reward = 0

        for step in range(1000):

            action= choose_action(state)
            trial_num += 1


            # print('action chosen: ', action)
            # print(type(action))
            new_state, reward, termination, info = env.step(action=action)
            env.render()

            if new_state[0] >= 0.0:
                reward += 10
            elif new_state[0] >= 0.25:
                reward += 20
            elif new_state[0] >= 0.5:
                reward += 100

            episode_reward += reward

            state = new_state

            if termination:
                env.close()
                ep_rewards.append(episode_reward)
                ep_lengths.append(step)
                print('episode num: ', episode, ' Total Reward: ', episode_reward)
                break

            # Used if you want to plot the current stats so far
            if keyboard.is_pressed('p'):
                fig, ax = plt.subplots(2, 1)
                ax[0].plot(list(range(episode)), ep_rewards, label="episode rewards", color='b')
                ax[0].legend()
                ax[1].plot(list(range(episode)), ep_lengths, label="episode lengths", color='g')
                ax[1].legend()
                plt.savefig('test_model_rewards.jpg')
                plt.close()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(list(range(episode)), ep_rewards, label="episode rewards", color='b')
    ax[0].legend()
    ax[1].plot(list(range(episode)), ep_lengths, label="episode lengths", color='g')
    ax[1].legend()
    plt.savefig('test_model_rewards.jpg')
    plt.close()



run()