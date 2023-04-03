from collections import deque
import numpy as np


class Memory:
    def __init__(self, max_len):
        self.max_len = max_len
        self.mem = deque(maxlen=max_len)

    def add_experience(self, state, action, reward, next_state, done):
        state_expanded = np.expand_dims(state, axis=0)
        next_state_expanded = np.expand_dims(next_state, axis=0)

        self.mem.append([state_expanded, action, reward, next_state_expanded, done])
