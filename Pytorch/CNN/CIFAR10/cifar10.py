import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from datetime import datetime
from Model import CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def LoadCIFAR10(flag=True):
    dataset = torchvision.datasets.CIFAR10(
        root=".",
        train=flag,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        ),
        download=True,
    )
    return dataset


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    # training loops
    for i in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []

        # training batch
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_losses[i] = np.mean(train_loss)

        # testing batch
        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())

        test_losses[i] = np.mean(test_loss)

        dt = datetime.now() - t0
        print(
            f"Epoch {i+1}/{epochs}, Train Loss: {train_losses[i]:.4f}, Test Loss: {test_losses[i]:.4f}, Duration: {dt}"
        )

    return train_losses, test_losses


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


if __name__ == "__main__":
    train_dataset = LoadCIFAR10()
    test_dataset = LoadCIFAR10(flag=False)

    size = 20
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=size, shuffle=False
    )

    model = CIFAR10()
    print(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = batch_gd(
        model, criterion, optimizer, train_loader, test_loader, epochs=20
    )

    plot_loss(train_losses, test_losses)

    # real test
    model.eval()
    n_correct = 0.0
    n_total = 0.0
    for inputs, targets in train_loader:
        # Move to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    train_acc = n_correct / n_total

    n_correct = 0.0
    n_total = 0.0
    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    x_test = test_dataset.data
    y_test = np.array(test_dataset.targets)
    p_test = np.array([])
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        p_test = np.concatenate((p_test, predictions.cpu().numpy()))

    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))
