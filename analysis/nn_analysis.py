from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset import TitanicDataset
from utils.nn_utils import train
from torch.optim import Adam
import torch
import torch.nn as nn

from dataset import load_preprocessed_titanic_dataset

LC_EPOCHS = 50
EPOCHS = 50
BATCH_SIZE = 16


def nn_learning_curve(model, dataset_sizes):
    writer = SummaryWriter()

    optm = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for dataset_size in dataset_sizes:
        tr_loss = 0
        tr_correct = 0
        test_loss = 0
        test_correct = 0
        for _ in range(3):
            X_train, X_test, y_train, y_test = load_preprocessed_titanic_dataset(limit_len=dataset_size)

            tr = TitanicDataset(X=X_train, y=y_train)
            test = TitanicDataset(X=X_test, y=y_test)

            data_train = DataLoader(dataset=tr, batch_size=BATCH_SIZE, shuffle=True)
            data_test = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(LC_EPOCHS):
                for bidx, batch in tqdm(enumerate(data_train)):
                    x_train, y_train = batch['inp'], batch['oup']
                    train(model, x_train, y_train, optm, criterion)

            correct, loss, _ = calculate_correct_and_loss(data_train, model, criterion)
            tr_correct += correct
            tr_loss += loss

            correct, loss, _ = calculate_correct_and_loss(data_test, model, criterion)
            test_correct += correct
            test_loss += loss

        writer.add_scalars('LC_loss/' + model.name, {
            'train': tr_loss / (tr.__len__() * 3),
            'test': test_loss / (test.__len__() * 3)
        }, dataset_size)

        writer.add_scalars('LC_accuracy/' + model.name, {
            'train': tr_correct / (tr.__len__() * 3),
            'test': test_correct / (test.__len__() * 3)
        }, dataset_size)


def calculate_correct_and_loss(dataset, model, criterion):
    correct = 0
    loss = 0
    tp = 0
    fp = 0
    fn = 0

    for bidx, batch in tqdm(enumerate(dataset)):
        x_test, y_test = batch['inp'], batch['oup']

        output = model(x_test)
        for idx, i in enumerate(output):
            i = torch.round(i)
            if i == y_test[idx]:
                loss += criterion(output, y_test)
                correct += 1
                if y_test[idx] == 1.0:
                    tp += 1
            else:
                if y_test[idx] == 1.0:
                    fn += 1
                else:
                    fp += 1
    f1 = 0
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return correct, loss, f1


def nn_accuracy_and_loss(model):
    writer = SummaryWriter(comment=model.name)

    optm = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train, X_test, y_train, y_test = load_preprocessed_titanic_dataset(test_size=0.5)

    tr = TitanicDataset(X=X_train, y=y_train)
    test = TitanicDataset(X=X_test, y=y_test)

    data_train = DataLoader(dataset=tr, batch_size=BATCH_SIZE, shuffle=True)
    data_test = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for bidx, batch in tqdm(enumerate(data_train)):
            x_train, y_train = batch['inp'], batch['oup']
            if bidx == 0:
                writer.add_graph(model, x_train)
            train(model, x_train, y_train, optm, criterion)

        correct, loss, f1 = calculate_correct_and_loss(data_train, model, criterion)
        writer.add_scalar("Accuracy/train", correct*100/len(tr), epoch+1)
        writer.add_scalar("Loss/train", loss, epoch+1)
        writer.add_scalar("F1/train", f1, epoch + 1)

        correct, loss, f1 = calculate_correct_and_loss(data_test, model, criterion)
        writer.add_scalar("Accuracy/test", correct*100/len(test), epoch+1)
        writer.add_scalar("Loss/test", loss, epoch+1)
        writer.add_scalar("F1/test", f1, epoch+1)