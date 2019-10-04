import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.initializers import TruncatedNormal, Zeros
from keras.optimizers import Adam
import numpy as np


class QNetworkKeras():

    def __init__(self, input_shape=(84, 84, 4), num_actions=4, learn_rate=0.001):
        self.num_actions = num_actions
        self.learning_rate = learn_rate

        self.behaviour_model = self.build_net()
        self.target_model = self.build_net()

        self.target_model.set_weights(self.behaviour_model.get_weights())


    def build_net(self):

        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), activation='relu', name='conv1', input_shape=(84, 84, 4),
                         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05), bias_initializer=Zeros()))      ##Output of 20x20x16

        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', name='conv2',
                         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05), bias_initializer=Zeros()))      ##Output of 9x9x32

        model.add(Conv2D(32, (3,3), strides=(1,1), activation='relu', name='conv3',
                         kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05), bias_initializer=Zeros()))      ## Output of 7x7x32



        model.add(Flatten())

        model.add(Dense(512, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05),          ## Flatten with shape 512
                        bias_initializer=Zeros()))
        

        model.add(Dense(128, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05),          ## Flatten with shape 128
                        bias_initializer=Zeros()))


        model.add(Dense(self.num_actions, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05),        ## Flatten with shape num_actions
                        bias_initializer=Zeros()))

        model.add(Activation('softmax'))

        adam = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

        return model



    def pred_q_values(self, input_state, model_type):
        # The states are saved as uint8 (0-255) but we need to normalize them by dividing by 255
        # Instead of saving float64 values in replay memory, I have normalized them before the
        # prediction and training methods


        if model_type=='behaviour':

            action_values = self.behaviour_model.predict((np.expand_dims(input_state, axis=0))/255)
            q_max = max(action_values)
            max_q_ind = np.where(action_values==q_max)[0][0]

        elif model_type=='target':
            action_values = self.target_model.predict((np.expand_dims(input_state, axis=0))/255)
            q_max = max(action_values)
            max_q_ind = np.where(action_values==q_max)[0][0]


        else:
            print('incorrect model type')

        return action_values, q_max, max_q_ind


    def update_target_model(self):
        self.target_model.set_weights(self.behaviour_model.get_weights())




    # def train_model(self, replay_memory, input_state, targets, sess):
#     # TODO: Use the replay memore to train the Qnetwork
#
#     with tf.Session() as sess:
#         sess.run(self.model['train'], feed_dict={self.init_state: input_state, self.td_targets: targets})


def print_trainable(self):
    # variables_names = [v.name for v in tf.trainable_variables()]

    # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(var)

    for var in tf.trainable_variables():
        print(var)
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     values = sess.run(variables_names)
    #     print(values)




