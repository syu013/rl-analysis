from mcts import pv_mcts_action
import numpy as np


def self_play(player_model, env, alpha):
    history = []

    value = 0
    player_action = pv_mcts_action(player_model, alpha)

    env.reset()
    state = env.board

    while True:
        action, policy = player_action(env)
        next_state, reward, done, info = env.step(action)
        history.append([state, policy, None, info['player']])
        env.render()

        state = next_state

        if done:
            value = reward
            break

    for i in range(len(history)):
        if history[i][3] == env.player:
            history[i][2] = value
        else:
            history[i][2] = -value

    return history


def random_play(player_model, env):
    player_action = pv_mcts_action(player_model, 0)

    env.reset()

    while True:
        if env.current_player == env.player:
            action, _ = player_action(env)
        else:
            legal_actions = [i for i, m in enumerate(env.action_mask()) if m]
            action = np.random.choice(legal_actions)
        next_state, reward, done, info = env.step(action)
        env.render()

        if done:
            break

    point = 0
    result = env.get_result(env.player)
    if result == 'draw':
        point = 0.5
    elif result == 'win':
        point = 1
    elif result == 'loss':
        point = 0

    return point


def play(player_model, opponent_model, env, alpha):
    player_action = pv_mcts_action(player_model, alpha)
    opponent_action = pv_mcts_action(opponent_model, alpha)

    env.reset()

    while True:
        if env.current_player == env.player:
            action, _ = player_action(env)
        else:
            action, _ = opponent_action(env)
        next_state, reward, done, info = env.step(action)
        env.render()

        if done:
            value = reward
            break

    point = 0
    result = env.get_result(env.player)
    if result == 'draw':
        point = 0.5
    elif result == 'win':
        point = 1
    elif result == 'loss':
        point = 0

    return point
