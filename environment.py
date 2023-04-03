import random

import gym


def initialize_new_game(env, agent):
    while len(agent.memory.mem) < agent.starting_mem_len:
        state = env.reset()
        done = False
        while not done:
            action = random.choice(agent.possible_actions)

            new_state, reward, done, info = env.step(action)

            agent.memory.add_experience(state, action, reward, new_state, done)

            state = new_state

    env.reset()


def make_env(name, agent):
    env = gym.make(name)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4,
                                          screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

    env = gym.wrappers.FrameStack(env, 4)

    initialize_new_game(env, agent)

    return env


def take_step(env, state, agent, score):

    action = agent.get_action(state)

    new_state, reward, done, info = env.step(action)

    agent.memory.add_experience(state, action, reward, new_state, done)

    if not done:
        agent.learn()

    return (score + reward), done, new_state


def play_episode(env, agent):
    state = env.reset()
    score = 0
    done = False
    while not done:
        score, done, state = take_step(env, state, agent, score)

    return score
