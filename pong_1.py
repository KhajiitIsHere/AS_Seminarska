import agent as the_agent
import environment
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np

num_episodes = 10000
eps = 1
eps_min = 0.05
eps_linear_decay = .9 / 100000
discount_factor = 0.99
learn_rate = .0001
batch_size = 32
max_memory = 8000
init_memory = 4000
target_c = 1000

name = 'PongNoFrameskip-v4'

if __name__ == '__main__':
    agent = the_agent.Agent(num_episodes=num_episodes, batch_size=batch_size, possible_actions=[0, 1, 2, 3, 4, 5],
                            learn_rate=learn_rate, target_c=target_c, epsilon_min=eps_min,
                            epsilon_decay=eps_linear_decay, starting_epsilon=eps, max_mem_len=max_memory,
                            starting_mem_len=init_memory, gamma=discount_factor)

    env = environment.make_env(name, agent)

    print('environment initialized')

    last_100_avg = [-21]
    scores = deque(maxlen=100)
    max_score = -21

    for i in range(num_episodes):
        timee = time.time()
        score = environment.play_episode(env, agent)
        scores.append(score)
        if score > max_score:
            max_score = score

        if i % 100 == 0 and i != 0:
            last_100_avg.append(sum(scores) / len(scores))
            plt.plot(np.arange(0, i + 1, 100), last_100_avg)
            plt.show()

    agent.epsilon = 0

    avg_score = 0

    for i in range(10):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.get_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            state = new_state

        avg_score += score

    print(avg_score)
