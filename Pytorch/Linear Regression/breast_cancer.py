import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Breast_Cancer_Detection
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def Information(data):
    print(data.DESCR)
    print("--------------------------------------")
    print(data.feature_names)
    print(data.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=43, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def Plotting(value1, value2, head):
    plt.title(head)
    plt.plot(value1)
    plt.plot(value2)
    plt.show()

def SaveModel(model):
    torch.save(model.state_dict(),"./module/breast_cancer.pt")


if __name__ == "__main__":
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = Information(data)

    # standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create model + training
    model = Breast_Cancer_Detection()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
    y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))

    # Train the model
    n_epochs = 1000
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    test_acc = np.zeros(n_epochs)
    train_acc = np.zeros(n_epochs)

    for it in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        train_losses[it] = loss.item()

        loss.backward()
        optimizer.step()

        # Get test loss + test acc + train_acc
        with torch.no_grad():
            p_train = model(X_train)
            p_train = np.round(p_train.numpy())
            train_acc[it] = np.mean(y_train.numpy() == p_train)

            p_test = model(X_test)
            loss_test = criterion(p_test, y_test)
            test_losses[it] = loss_test.item()

            p_test = np.round(p_test.numpy())
            test_acc[it] = np.mean(y_test.numpy() == p_test)

        if it % 10 == 0:
            print("--------------------------------------")
            print(
                f"Epoch #{it} - train_loss:{train_losses[it]} | val_loss:{test_losses[it]} | train_acc:{train_acc[it]} | val_acc:{test_acc[it]}"
            )

    Plotting(train_losses, test_losses, head = 'Loss value')
    Plotting(train_acc, test_acc, head = 'Accuracy value')
    SaveModel(model)