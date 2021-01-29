from model import DualNet
import gym
from play import play, self_play, random_play
from tqdm import tqdm
from collections import OrderedDict
import os
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path
import torch.optim as optim

SELF_COUNT = 300
GAME_COUNT = 20
EPOCH = 100
BATCH_SIZE = 128
ALPHA = 1.0


def write_data(history):
    now = datetime.now()
    os.makedirs(os.path.dirname(
        os.path.abspath(__file__)) + '\\data', exist_ok=True)
    path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)) + '\\data', '{0:%Y%m%d-%H%M%S}.history'.format(now))
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def load_data():
    history_path = sorted(Path(os.path.dirname(
        os.path.abspath(__file__)) + '\\data').glob('*.history'))
    history = []
    with history_path[-1].open(mode='rb') as f:
        history.extend(pickle.load(f))

    return history


def train(env):
    total_points = 0
    histories = []

    player_model = DualNet(2, 128, 36, 1)

    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '\\data\\best.ckpt'):
        player_model.load_model('best')
    else:
        player_model.save_model('best')
    player_model.eval()

    # self-play
    if SELF_COUNT > 0:
        with tqdm(range(SELF_COUNT), desc='self-play') as pbar:
            for i in pbar:
                history = self_play(player_model, env, ALPHA)
                histories.extend(history)
                print('\u001B[1A', end='')
        write_data(histories)
    
    # train
    histories = load_data()
    player_model.train()
    optimizer = optim.Adam(player_model.parameters())
    for i in tqdm(range(EPOCH), desc='epoch'):
        with tqdm(np.array_split(histories, int(len(histories))/BATCH_SIZE), desc='optimize') as pbar:
            for history in pbar:
                valueloss, policyloss = player_model.optimize(
                    history, optimizer)
                pbar.set_postfix(OrderedDict(
                    value=valueloss.item(), policy=policyloss.item()))
        print('\u001B[1A', end='')
    player_model.save_model('latest')

    # play
    player_model.eval()
    opponent_model = DualNet(2, 128, 36, 1)
    opponent_model.load_model('best')
    opponent_model.eval()
    with tqdm(range(GAME_COUNT), desc='play') as pbar:
        for i in pbar:
            total_points += play(player_model, opponent_model, env, ALPHA)
            print('\u001B[1A', end='')
            pbar.set_postfix(OrderedDict(point=total_points/(i+1)))

    print(f'対戦結果: {total_points / GAME_COUNT}')

    if total_points / GAME_COUNT > 0.55:
        player_model.save_model('best')


if __name__ == "__main__":
    env = gym.make('gym_reversi:reversi-v0')
    for _ in range(30):
        train(env)
