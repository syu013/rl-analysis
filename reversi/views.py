

from django.views import generic
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

import sys
import numpy as np
import copy
from datetime import datetime
from .models import Trajectory

from reversi.alpha_zero import DualNet, pv_mcts_action
from reversi.gym_reversi.gym_reversi.envs import ReversiEnv

env = ReversiEnv()
model = DualNet(2, 128, 36, 1)
model.load_model('best')
model.eval()
action_func = pv_mcts_action(model, 0, 3)

# Create your views here.
class IndexView(generic.TemplateView):
    template_name = "index.html"

    def post(self, request, *args, **kwargs):
        if request.POST.get('analysis'):
            num_action = np.zeros(36)
            trajectory = Trajectory.objects.filter(
                user=request.user).order_by('created_at').reverse()
            for i, t in enumerate(trajectory):
                for action in result_split(t.trajectory):
                    num_action[int(action)] += 1 * (0.9**i)
            params = {'value': ','.join(
                list(map(str, (num_action/np.max(num_action)).tolist())))}
            return render(request, 'analysis.html', params)


class AnalysisView(generic.TemplateView):
    template_name = "analysis.html"


@csrf_exempt
def board_post(request):
    # Unityからプレイヤー情報，ボード情報を受け取る
    post = request.POST.get('board', None)

    # プレイヤー情報を取り出す
    post = post.split('.')
    current_player = int(post[0][0])
    player = int(post[0][1])
    post = post[1]

    # ボード情報を取り出す
    board = extract_board(post)

    # ボード情報から行動を決定する
    env.reset()
    env.current_player = current_player
    env.player = player
    numpy_board = np.array(board)
    player_board = np.where(numpy_board == 1, 1, 0)
    opponent_board = np.where(numpy_board == 2, 1, 0)
    env.board = np.copy(np.array([player_board, opponent_board]))
    action, _ = action_func(copy.deepcopy(env))

    return HttpResponse(str(action))


@csrf_exempt
def result_post(request):
    # Unityから軌道を受け取る
    post = request.POST.get('result', None)
    model = Trajectory(user=request.user,
                       created_at=datetime.now(), trajectory=post)
    model.save()
    result_split(post)
    return HttpResponse(post)


def result_split(result):
    trajectory = []
    for t in result.split('?'):
        board, action = t.split('!')
        trajectory.append(action)
    return trajectory


def extract_board(board):
    board_ = []
    for b in board.split(':'):
        board_.append(list(map(int, b.split(','))))
    return board_
