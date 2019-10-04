import tensorflow as tf
import cv2
import numpy as np
from collections import deque

class ReplayMemory():


    def __init__(self, memory_capacity):

        #will consist of all the frames played by the agent. Since 4 frames make up a state,
        # elements at index(0,1,2,3) will make up the first state
        # the first frame provided by the environment is not added to this
        self.frames = deque([], maxlen=memory_capacity)

        #others will consists of [rewards, actions, terminations]
        # it will have the same legnth as that of self.frames
        self.others = deque([], maxlen=memory_capacity)


    # To avoid saving float64 the replay mem only stores uint values so normalization steps in not
    # performed here but instead will be done during training and prediction
    def add(self, new_frame, reward, action, termination):
        gray = self.preprocess_image(new_frame)
        self.frames.append(gray)
        self.others.append([reward, action, termination])


    # state shape is (84,84,4)
    def get_latest_state(self):
        imgs = [self.frames[-4+i] for i in range(4)]
        state = np.stack(imgs, axis=2)
        return state

    def clear_replay_memory(self):
        self.frames.clear()
        self.others.clear()


    def get_state_from_indices(self, frame_indices):
        states = []
        # print('frame_indices', frame_indices)
        for i in frame_indices:
            # last frame is with index i
            state = [self.frames[i-3+j] for j in range(4)]

            state = np.stack(state, axis=2)
            states.append(state)
            # print('state shape: ', state.shape)

        states = np.asarray(states)
        # print(states.shape)

        return states


    # epochs is how many times you want your data to be feed forwarded through your network
    # previous_frames is the number of frames starting from your right you want to use to train your model
    def get_sample(self, frame_indices):
        # print("Control here")

        current_states = self.get_state_from_indices(frame_indices)

        # list of [rewards, actions, terminations] for all frame_indices
        current_info = [self.others[i] for i in frame_indices]

        future_frame_indices = [i+1 for i in frame_indices]
        future_states = self.get_state_from_indices(future_frame_indices)

        return current_states, current_info, future_states








    @staticmethod
    def preprocess_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[30:195, :]
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)

        return (gray)

    # def image_to_state(self, gray):

