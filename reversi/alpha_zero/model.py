import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

RESIDUAL_NUM = 8


class RasidualBlock(nn.Module):
    def __init__(self, hidden):
        super(RasidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=[3//2, 3//2])
        self.conv2 = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=[3//2, 3//2])
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        sc = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x)) + sc
        x = self.relu2(x)
        return x


class DualNet(nn.Module):
    def __init__(self, input, hidden, p_size, v_size):
        super(DualNet, self).__init__()
        self.conv = nn.Conv2d(
            input, hidden, kernel_size=3, padding=[3//2, 3//2])
        self.bn = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU()
        self.ave_pool = GlobalAvgPool2d()
        self.p = nn.Linear(hidden, p_size)
        self.v = nn.Linear(hidden, v_size)
        self.blocks = [RasidualBlock(hidden) for _ in range(RESIDUAL_NUM)]

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.ave_pool(x)
        p = torch.softmax(self.p(x), dim=-1)
        v = torch.tanh(self.v(x))
        return p, v

    def optimize(self, history, optimizer):
        states, policies, values, _ = zip(*history)
        states = torch.from_numpy(np.array(states).astype(np.float32)).clone()
        policies = torch.from_numpy(
            np.array(policies).astype(np.float32)).clone()
        values = torch.from_numpy(np.array(values).astype(np.float32)).clone()

        expected_policies, expected_values = self(states)
        mseloss = ((expected_values.view(-1) - values) ** 2).sum(0) / values.size(0)
        entropyloss = - (torch.log(expected_policies) * policies).sum(1).sum(0) / policies.size(0)
        loss = mseloss + entropyloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return mseloss, entropyloss

    def save_model(self, name):
        os.makedirs(os.path.dirname(os.path.abspath(
            __file__)) + '/models', exist_ok=True)
        path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)) + '/models', name + '.ckpt')
        torch.save({'model': self.state_dict()}, path)

    def load_model(self, name):
        path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)) + '/models', name + '.ckpt')
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


if __name__ == "__main__":
    state = torch.Tensor([[
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    ]])
    model = DualNet(2, 128, 36, 1, 36)
