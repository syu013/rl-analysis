import torch
import gym
import numpy as np
import copy

PV_EVALUATE_COUNT = 30


def boltzman(xs, alpha):
    xs = [x ** (1/alpha) for x in xs]
    return [x / np.sum(xs) for x in xs]


# 訪問回数取得
def nodes_to_n(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores


# スコア取得
def pv_mcts_scores(model, env, alpha, evaluate_count):
    root_node = Node(MCTS(env), 0)

    for _ in range(evaluate_count):
        root_node.evaluate(model)
    scores = nodes_to_n(root_node.child_nodes)
    # 最適行動
    if alpha == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, alpha)
    return scores


def pv_mcts_action(model, alpha=0, evaluate_count=30):
    def pv_mcts_action(env):
        scores = pv_mcts_scores(model, env, alpha, evaluate_count)
        legal_actions = env.get_put_place(env.current_player)
        action = np.random.choice(legal_actions, p=scores)
        policy = []
        i = 0
        for m in env.action_mask():
            if m:
                policy.append(scores[i])
                i += 1
            else:
                policy.append(0)

        return action, policy
    return pv_mcts_action


class Node:
    def __init__(self, mcts, p):
        self.mcts = mcts
        self.p = p
        self.w = 0
        self.n = 0
        self.child_nodes = None

    def evaluate(self, model):
        # 終端
        if self.mcts.env.done:
            value = 0
            result = self.mcts.env.get_result(self.mcts.env.current_player)
            if result == 'win':
                value = 0
            elif result == 'loss':
                value = -1
            elif result == 'draw':
                value = 0
            else:
                assert Exception

            self.w += value
            self.n += 1
            return value
        # 子ノードが存在しない場合
        if not self.child_nodes:
            policies, value = self.mcts.predict(model)
            self.w += value
            self.n += 1

            self.child_nodes = []
            actions = self.mcts.env.get_put_place(self.mcts.env.current_player)
            for action, policy in zip(actions, policies):
                env = copy.deepcopy(self.mcts.env)
                env.step(action)
                mtcs = MCTS(env)
                self.child_nodes.append(Node(mtcs, policy))
            return value
        # 子ノードが存在する場合
        else:
            value = 0
            next_node = self.next_child_node()
            if next_node.mcts.env.current_player == self.mcts.env.current_player:
                value = next_node.evaluate(model)
            else:
                value = -next_node.evaluate(model)

            self.w += value
            self.n += 1
            return value

    def next_child_node(self):
        C_PUCT = 1.0
        t = np.sum(nodes_to_n(self.child_nodes))
        pucb_values = []
        for child_node in self.child_nodes:
            pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                               C_PUCT*child_node.p * np.sqrt(t) / (1 + child_node.n))

        return self.child_nodes[np.argmax(pucb_values)]


class MCTS:
    def __init__(self, env):
        self.env = env

    def predict(self, model):
        state = self.env.board
        state = torch.from_numpy(state.astype(np.float32)).clone().unsqueeze(0)
        with torch.no_grad():
            p, v = model(state)
        self.env.board = state.numpy()[0]
        mask = self.env.action_mask()
        policies = np.where(mask, p.numpy()[0], 0)
        policies /= np.sum(policies) if np.sum(policies) else 1

        return policies, v.item()


if __name__ == "__main__":
    from model import DualNet
    model = DualNet(2, 128, 36, 1, 36)
    env = gym.make('gym_reversi:reversi-v0')
    next_action = pv_mcts_action(model, 1.0)
    print(next_action(env))
