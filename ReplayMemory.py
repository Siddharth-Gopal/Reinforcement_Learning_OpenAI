import tensorflow as tf
import cv2
import numpy as np
from collections import deque

class ReplayMemory():


    def __init__(self, memory_capacity):

        # Used for MountainCar
        self.memory = deque([], maxlen=memory_capacity)



    def get_sample_mountaincar(self, indices):

        # replay memory has state, action, reward, new_state, termination
        current_states = []
        actions = []
        rewards = []
        future_states = []
        terminations = []


        for i in indices:
            current_states.append(self.memory[i][0])
            actions.append(self.memory[i][1])
            rewards.append(self.memory[i][2])
            future_states.append(self.memory[i][3])
            terminations.append(self.memory[i][4])

        current_states = np.asarray(current_states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        future_states = np.asarray(future_states)
        terminations = np.asarray(terminations)

        return current_states, actions, rewards, future_states, terminations
