import gym
import numpy as np
from gym import spaces
import random
import time

BOARD_SIZE = 6


class ReversiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, BOARD_SIZE, BOARD_SIZE))
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)

        self._current_player = random.randint(1, 2)  # 現在のプレイヤー
        self._player = random.randint(1, 2)  # 操作プレイヤー
        self._done = False

        self._init_board()

    @property
    def done(self):
        return self._done

    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, current_player):
        self._current_player = int(current_player)

    @property
    def player(self):
        return self._player

    @player.setter
    def player(self, player):
        self._player = int(player)

    @property
    def board(self):
        numpy_board = np.array(self._board)
        player_board = np.where(numpy_board == 1, 1, 0)
        opponent_board = np.where(numpy_board == 2, 1, 0)
        return np.copy(np.array([player_board[1:BOARD_SIZE+1, 1:BOARD_SIZE+1], opponent_board[1:BOARD_SIZE+1, 1:BOARD_SIZE+1]]))

    @board.setter
    def board(self, board):
        numpy_board = np.array(board)
        board_ = np.where(numpy_board[0] == 1, 1, 0) + \
            np.where(numpy_board[1] == 1, 2, 0)
        self._board = np.pad(board_, [1, 1], constant_values=(-1, -1))

    def step(self, action):
        p1 = int(action / BOARD_SIZE) + 1
        p2 = int(action % BOARD_SIZE) + 1

        if not self._is_put_stone(p1, p2, self.current_player):
            assert ValueError(f'You can not put a stone [{p1}, {p2}]')

        self._put_stone(p1, p2, self.current_player)

        opponent = 2 if self.current_player == 1 else 1
        if not self._exist_put_stones(opponent):
            if(not self._exist_put_stones(self.current_player)):
                self._done = True
                info = {'player': self.player}
                result = self.get_result(self.player)
                if result == 'draw':
                    return self.board, 0, self.done, info
                elif result == 'win':
                    return self.board, 1, self.done, info
                elif result == 'loss':
                    return self.board, -1, self.done, info
                else:
                    assert Exception('The game is not over')
            else:
                info = {'player': self.current_player}
                return self.board, 0, self.done, info
        else:
            info = {'player': self.current_player}
            self.current_player = 2 if self.current_player == 1 else 1
            return self.board, 0, self.done, info

    def reset(self):
        self._init_board()
        self._done = False
        self.current_player = random.randint(1, 2)
        self.player = random.randint(1, 2)
        return self.board

    def action_mask(self):
        actions = self.get_put_place(self.current_player)
        mask = [False for _ in range(BOARD_SIZE * BOARD_SIZE)]
        for action in actions:
            mask[action] = True
        return mask

    def get_put_place(self, player):
        place = []
        for p1 in range(1, BOARD_SIZE+1):
            for p2 in range(1, BOARD_SIZE+1):
                if self._is_put_stone(p1, p2, player):
                    place.append((p1-1) * BOARD_SIZE + (p2-1))
        return place

    def render(self, mode='human'):
        board_ = ''
        for i in range(1, BOARD_SIZE+1):
            for j in range(1, BOARD_SIZE+1):
                if self._board[i][j] == 0:
                    board_ += '．'
                elif self._board[i][j] == 1:
                    board_ += '●'
                elif self._board[i][j] == 2:
                    board_ += '○'
            board_ += '\n'
        if self.player == 1:
            print(board_ + 'player: 白')
        else:
            print(board_ + 'player: 黒')
        time.sleep(0.01)
        print(f'\u001B[{BOARD_SIZE+1}A', end='')
        time.sleep(0.01)

    def get_result(self, player):
        if not self.done:
            return 'middle'

        player_count = 0
        opponent_count = 0
        opponent = 2 if player == 1 else 1

        for p1 in range(1, BOARD_SIZE+1):
            for p2 in range(1, BOARD_SIZE+1):
                if self._board[p1][p2] == player:
                    player_count += 1
                elif self._board[p1][p2] == opponent:
                    opponent_count += 1

        if player_count == opponent_count:
            return 'draw'
        elif player_count > opponent_count:
            return 'win'
        else:
            return 'loss'

    def _init_board(self):
        board_ = [[-1 for _ in range(BOARD_SIZE+2)]
                  for _ in range(BOARD_SIZE+2)]
        for i in range(1, BOARD_SIZE+1):
            for j in range(1, BOARD_SIZE+1):
                board_[i][j] = 0

        center = int((BOARD_SIZE+2)/2 - 1)

        board_[center][center+1] = board_[center+1][center] = 1
        board_[center][center] = board_[center+1][center+1] = 2

        self._board = board_

    def _put_stone(self, p1, p2, player):
        for d1 in range(-1, 2):
            for d2 in range(-1, 2):
                if d1 == 0 and d2 == 0:
                    continue
                count = self._count_turn_stones(p1, p2, d1, d2, player)
                for c in range(1, count+1):
                    self._board[p1+c*d1][p2+c*d2] = player
        self._board[p1][p2] = player

    def _exist_put_stones(self, player):
        for p1 in range(1, BOARD_SIZE+1):
            for p2 in range(1, BOARD_SIZE+1):
                if self._is_put_stone(p1, p2, player):
                    return True
        return False

    def _is_put_stone(self, p1, p2, player):
        if p1 < 1 or p1 > BOARD_SIZE or p2 < 1 or p2 > BOARD_SIZE:
            return False
        if self._board[p1][p2] != 0:
            return False
        for d1 in range(-1, 2):
            for d2 in range(-1, 2):
                if d1 == 0 and d2 == 0:
                    continue
                if self._count_turn_stones(p1, p2, d1, d2, player) > 0:
                    return True
        return False

    def _count_turn_stones(self, p1, p2, d1, d2, player):
        i = 1
        opponent = 2 if player == 1 else 1

        while True:
            if self._board[p1+i*d1][p2+i*d2] == opponent:
                i += 1
            else:
                break

        if self._board[p1+i*d1][p2+i*d2] == player:
            return i - 1
        else:
            return 0


if __name__ == '__main__':
    env = ReversiEnv()
    env.render()
