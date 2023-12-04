import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from datetime import datetime
from LSTM_model import LSTM_MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_MNIST():
    train_dataset = torchvision.datasets.MNIST(
        root=".", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=".", train=False, transform=transforms.ToTensor(), download=True
    )

    return train_dataset, test_dataset


def training(model, criterion, optimizer, train_loader, test_loader, epochs):
    # Train the model
    n_epochs = 10

    # Stuff to store
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    for it in range(n_epochs):
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # reshape the input
            inputs = inputs.view(-1, 28, 28)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, 28, 28)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        print(
            f"Epoch {it+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )
    return train_losses, test_losses


def plotloss(train_losses, test_losses):
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.show()


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def Validation(model):
    n_correct = 0.
    n_total = 0.
    for inputs, targets in train_loader:
        # move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # reshape the input
        inputs = inputs.view(-1, 28, 28)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)
        
        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    train_acc = n_correct / n_total


    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_loader:
        # move data to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        
        # reshape the input
        inputs = inputs.view(-1, 28, 28)

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


if __name__ == "__main__":
    train_dataset, test_dataset = load_MNIST()

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )


    model = LSTM_MNIST(28, 128, 2, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, test_losses = training(
        model, criterion, optimizer, train_loader, test_loader, epochs=10
    )
    plotloss(train_losses, test_losses)
    Validation(model)

    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    p_test = np.array([])
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(-1, 28, 28)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        p_test = np.concatenate((p_test, predictions.cpu().numpy()))

    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))
