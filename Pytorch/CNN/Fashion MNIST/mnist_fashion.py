import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from Model import FashionMNIST
from sklearn.metrics import confusion_matrix
import itertools
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def LoadFMNIST(flag=True):
    dataset = torchvision.datasets.FashionMNIST(
        root=".",
        train=flag,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True,
    )
    return dataset


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(
            f"Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}"
        )

    return train_losses, test_losses


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


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_dataset = LoadFMNIST()
    test_dataset = LoadFMNIST(flag=False)

    size = 128
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=size, shuffle=False)

    model = FashionMNIST()
    print(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)

    train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs=15)
    plot_loss(train_losses, test_losses)

    model.eval()
    n_correct = 0.0
    n_total = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    train_acc = n_correct / n_total

    n_correct = 0.0
    n_total = 0.0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total
    print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    p_test = np.array([])
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        p_test = np.concatenate((p_test, predictions.cpu().numpy()))

    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))
