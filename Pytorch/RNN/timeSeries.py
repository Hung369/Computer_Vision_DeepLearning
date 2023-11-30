import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from RNN_model import SimpleRNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_series(series):
    plt.plot(series)
    plt.show()


# Training
def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=1000):

    # Stuff to store
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Save losses
        train_losses[it] = loss.item()

        # Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()
        
        if (it + 1) % 5 == 0:
            print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    return train_losses, test_losses

def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.show()

if __name__ == "__main__":
    # make original data series
    N = 1000
    series = np.sin(0.1 * np.arange(N))
    plot_series(series)

    T = 10
    X = []
    Y = []
    for t in range(len(series) - T):
        x = series[t:t+T]
        X.append(x)
        y = series[t+T]
        Y.append(y)

    X = np.array(X).reshape(-1, T, 1)
    Y = np.array(Y).reshape(-1, 1)
    N = len(X)
    print("X.shape", X.shape, "Y.shape", Y.shape)

    # create model
    model = SimpleRNN(n_inputs=1, n_hidden=15, n_rnnlayers=1, n_outputs=1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train test split
    X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
    y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
    X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
    y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test)
    plot_loss(train_losses, test_losses)

    validation_target = Y[-N//2:]
    validation_predictions = []

    # index of first validation input
    i = 0

    while len(validation_predictions) < len(validation_target):
        input_ = X_test[i].reshape(1, T, 1)
        p = model(input_)[0,0].item() # 1x1 array -> scalar
        i += 1
    
        # update the predictions list
        validation_predictions.append(p)
    
    plt.plot(validation_target, label='forecast target')
    plt.plot(validation_predictions, label='forecast prediction')
    plt.show()
