import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from Model import MyModel

url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'

def ReadData(path):
    data = pd.read_csv(path, header=None).values
    print(data)
    return data

def Scatter(X, Y):
    plt.scatter(X, Y)
    plt.show()

def StandardNormalize(X, Y):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    return X.astype(np.float32), Y.astype(np.float32)

def splitting(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle= True)
    return X_train, X_test, y_train, y_test

def Plotting(value1, value2):
    plt.plot(value1)
    plt.plot(value2)
    plt.show()

if __name__ == '__main__':
    data = ReadData(url)
    X = data[:, 0].reshape(-1, 1)  # making NxD matrix
    Y = data[:, 1].reshape(-1, 1)
    Scatter(X, Y) # X.shape = Y.shape = (162, 1)

    # bring back to linear equation
    Y = np.log(Y)
    Scatter(X, Y)
    
    # standard scaling
    X, Y = StandardNormalize(X, Y)
    Scatter(X, Y)

    # train test split
    X_train, X_test, y_train, y_test = splitting(X, Y)


    # implement and train model
    model = MyModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= 0.01, momentum= 0.7)

    input = torch.from_numpy(X_train)
    target = torch.from_numpy(y_train)
    input_test = torch.from_numpy(X_test)
    target_test = torch.from_numpy(y_test)

    # train your model
    train_losses = []
    val_losses = []
    val_acc = []
    train_acc = []

    for i in range(101):
        optimizer.zero_grad()

        # calculate losses
        outputs = model(input)
        loss = criterion(outputs, target)
        train_losses.append(loss.item())
        
        # gradient descent
        loss.backward()
        optimizer.step()

        # validation time
        with torch.no_grad():
            p_test = model(input_test)
            loss_test = criterion(p_test, target_test)
            val_losses.append(loss_test.item())

            # # validation acc
            # p_test = np.round(p_test.numpy())
            # validation_acc = np.mean(target_test.numpy() == p_test)
            # val_acc.append(validation_acc)

            # # train acc
            # p_train = model(input)
            # p_train = np.round(p_train.numpy())
            # training_acc = np.mean(target_test.numpy() == p_train)
            # train_acc.append(training_acc)
    
    # plot losses
    Plotting(val_losses, train_losses)

    # Plot the graph
    predicted = model(torch.from_numpy(X)).detach().numpy()
    plt.plot(X, Y, '*', label='Original data')
    plt.plot(X, predicted, label='Fitted line')
    plt.legend()
    plt.show()