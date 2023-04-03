from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, Input
from keras.optimizers import Adam
import tensorflow as tf
from agent_memory import Memory
import numpy as np
import random


class Agent:
    def __init__(self, num_episodes, possible_actions, starting_mem_len, max_mem_len, starting_epsilon, learn_rate,
                 gamma, batch_size, epsilon_min, epsilon_decay, target_c):
        self.num_episodes = num_episodes

        self.memory = Memory(max_mem_len)
        self.starting_mem_len = starting_mem_len

        self.batch_size = batch_size

        self.possible_actions = possible_actions

        self.epsilon = starting_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma

        self.learn_rate = learn_rate

        self.target_c = target_c

        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.learns = 0

    def _build_model(self):
        model = Sequential()
        model.add(Input((4, 84, 84)))
        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, data_format="channels_first", activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, data_format="channels_first", activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, data_format="channels_first", activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        # model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, data_format="channels_first", activation="relu",
        #                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        # model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, data_format="channels_first", activation="relu",
        #                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())

        model.add(Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        # model.add(Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(len(self.possible_actions), activation='linear'))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model

    def get_action(self, state):
        """Explore"""
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]

        """Do Best Acton"""
        a_index = np.argmax(self.model.predict(np.expand_dims(state, axis=0) / 255))
        return self.possible_actions[a_index]

    def learn(self):
        if len(self.memory.mem) < self.starting_mem_len:
            return

        states = []
        targets = []

        minibatch = random.sample(self.memory.mem, self.batch_size)

        for state, action, reward, new_state, done in minibatch:

            state = state / 255
            new_state = new_state / 255

            target = reward

            if not done:
                best_future_action = np.argmax(self.model.predict(new_state))
                target = reward + self.gamma * self.model_target.predict(new_state)[0][best_future_action]

            target_vector = self.model.predict(state)[0]

            target_vector[action] = target

            states.append(state[0])

            targets.append(target_vector)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1

        if self.learns % self.target_c == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')