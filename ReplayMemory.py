import tensorflow as tf
import cv2
import numpy as np
from collections import deque
import random


#SumTree class taken from https://github.com/jaromiru/AI-blog/blob/348628b105058d876001ca758b6ba59fb1726614/SumTree.py#L3

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.populated_tree = False

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.populated_tree = True

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayMemory:

    def __init__(self, memory_capacity):
        #will consist of all the frames played by the agent. Since 4 frames make up a state,
        # elements at index(0,1,2,3) will make up the first state
        # the first frame provided by the environment is not added to this
        self.frames = deque([], maxlen=memory_capacity)
        #others will consists of [rewards, actions, terminations]
        # it will have the same legnth as that of self.frames
        self.others = deque([], maxlen=memory_capacity)
        # Used for MountainCar
        self.memory = deque([], maxlen=memory_capacity)

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

    @staticmethod
    def preprocess_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[30:195, :]
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return (gray)
    # def image_to_state(self, gray):

class ReplayMemoryPER:



    def __init__(self, capacity):
        self.tree = SumTree(capacity)

        self.abs_upper_error = 1.0
        self.PER_b = 0.5
        # how much the value must increment per step
        self.PER_b_increment = 0.001

        # Hyper parameter to tune replay memory between prioritization and random sampling
        self.PER_a = 0.4

        # minimum constant to add to errors to maintain nonzero value
        self.PER_e = 0.01

    def add(self, transition):
        # We initialize all new experiences with max priority so that they would at least be used to train once
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_upper_error

        self.tree.add(max_priority, transition)

    def update_tree(self, tree_indices, absolute_errors):
        absolute_errors += self.PER_e
        clipped_errors = np.minimum(absolute_errors, self.abs_upper_error)
        priorities = np.power(clipped_errors, self.PER_a)

        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, priority)
        # print('replaymem tree total: ',self.tree.total())


    def sample(self, n):

        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_increment])
        priority_slice = self.tree.total() / n
        if self.tree.populated_tree:
            min_priority = np.min(self.tree.tree[-self.tree.capacity:])
        else:
            min_priority = np.min(self.tree.tree[-self.tree.capacity:-self.tree.capacity + self.tree.write])
        if min_priority==0:
            min_priority = 0.001
        max_ISweight = np.power(n*min_priority, -self.PER_b)

        # print('Max IS weight: ', max_ISweight)
        batch_ISweights = np.empty(n, dtype=np.float32)
        batch_tree_indices = np.empty(n, dtype=np.int32)
        batch_data = []

        for i in range(n):
            a, b = i*priority_slice, (i+1)*priority_slice
            r = random.uniform(a, b)

            idx, priority, data = self.tree.get(r)
            # print('priority: ',  idx, priority, data)
            # ISweights are divided by max weight in order to avoid very large values
            batch_ISweights[i] = np.power((n*priority), -self.PER_b) / max_ISweight
            batch_tree_indices[i] = idx
            batch_data.append(data)

        return batch_tree_indices, batch_ISweights, batch_data







