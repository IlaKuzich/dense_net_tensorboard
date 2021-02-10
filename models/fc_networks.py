import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_net(layers=1, activation=F.relu, apply_batch_norm=True):
    return {
        1: Network1l(activation),
        2: Network2l(activation, apply_batch_norm),
        3: Network3l(activation, apply_batch_norm),
        4: Network4l(activation, apply_batch_norm)
    }[layers]


class Network1l(nn.Module):
    def __init__(self, activation):
        super(Network1l, self).__init__()

        self.name = '1_layer_' + activation.__name__

        self.activation = activation
        self.fc1 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))

        return x

    def name(self):
        return self.name


class Network2l(nn.Module):
    def __init__(self, activation, apply_batch_norm=True):
        super(Network2l, self).__init__()

        self.name = '2_layer_' + activation.__name__ + ('_normalized' if apply_batch_norm else '')

        self.apply_batch_norm = apply_batch_norm
        self.activation = activation

        self.fc1 = nn.Linear(10, 6)
        self.b1 = nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        if self.apply_batch_norm:
            x = self.b1(x)
        x = F.sigmoid(self.fc2(x))

        return x

    def name(self):
        return self.name


class Network3l(nn.Module):
    def __init__(self, activation, apply_batch_norm=True):
        super(Network3l, self).__init__()

        self.name = '3_layer_' + activation.__name__ + ('_normalized' if apply_batch_norm else '')

        self.apply_batch_norm = apply_batch_norm
        self.activation = activation

        self.fc1 = nn.Linear(10, 8)
        self.b1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 4)
        self.b2 = nn.BatchNorm1d(4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        if self.apply_batch_norm:
            x = self.b1(x)
        x = self.activation(self.fc2(x))
        if self.apply_batch_norm:
            x = self.b2(x)
        x = F.sigmoid(self.fc3(x))

        return x

    def name(self):
        return self.name


class Network4l(nn.Module):
    def __init__(self, activation, apply_batch_norm=True):
        super(Network4l, self).__init__()

        self.name = '4_layer_' + activation.__name__ + ('_normalized' if apply_batch_norm else '')

        self.apply_batch_norm = apply_batch_norm
        self.activation = activation

        self.fc1 = nn.Linear(10, 10)
        self.b1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 4)
        self.b3 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        if self.apply_batch_norm:
            x = self.b1(x)
        x = self.activation(self.fc2(x))
        if self.apply_batch_norm:
            x = self.b2(x)
        x = self.activation(self.fc3(x))
        if self.apply_batch_norm:
            x = self.b3(x)
        x = F.sigmoid(self.fc4(x))

        return x

    def name(self):
        return self.name
