import sys
from train import train

sys.path.append('\\'.join(__file__.split('\\')[:-2]))
from gym_reversi.envs import ReversiEnv

env = ReversiEnv()
for _ in range(30):
    train(env)