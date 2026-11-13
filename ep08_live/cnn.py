import torch
from torch import nn
from scramblekit import fashion_mnist, race, slide

X_train, y_train, X_test, y_test = fashion_mnist()

def new_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128), nn.ReLU(),
        nn.Linear(128,64), nn.ReLU(),
        nn.Linear(64,10),
    )

torch.manual_seed(42)
order = torch.randperm(784)

def scramble(X):
    return X.reshape(-1, 784)[:, order].reshape(-1,28,28)

normal, scrambled = race(new_model,
                         normal = (X_train, X_test),
                         scrambled = (scramble(X_train), scramble(X_test)),
                         labels = (y_train, y_test), epochs = 12)

slide(normal, X_test, y_test, pixels = 2)