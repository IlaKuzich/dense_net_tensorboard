import torch.nn.functional as F

from models.fc_networks import construct_net
from analysis.nn_analysis import nn_learning_curve
from analysis.nn_analysis import nn_accuracy_and_loss

from itertools import product
from dataset import load_preprocessed_titanic_dataset

if __name__ == "__main__":
    load_preprocessed_titanic_dataset()

    layers = (1, 2, 3, 4)
    functions = (F.sigmoid, F.relu, F.tanh)
    normalize = (False, True)

    for params in list(product(layers, functions, normalize)):
        model = construct_net(params[0], params[1], params[2])
        nn_learning_curve(model, [25, 50, 100, 200, 300, 400, 500, 600, 700])
        nn_accuracy_and_loss(model)