import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import datetime


def load_MNIST(value=True):
  dataset = torchvision.datasets.MNIST(root='.', train=value, transform=transform.ToTensor(), download=True)
  return dataset

def build_model():
  model = nn.Sequential(
      nn.Linear(784, 128),
      nn.ReLU(),
      nn.Linear(128,10)
  )
  return model

def plot_loss(train_losses, test_losses):
  plt.plot(train_losses, label='train loss')
  plt.plot(test_losses, label='test loss')
  plt.legend()
  plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


if __name__ == "__main__":

  # initialize all set
  train_dataset = load_MNIST()
  test_dataset = load_MNIST(value=False)
  model = build_model()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())
  size = 128

  train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = size, shuffle = True)
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = size, shuffle = True)

  # training time
  n_epochs = 10

  train_losses = np.zeros(n_epochs)
  test_losses = np.zeros(n_epochs)

  for it in range(n_epochs):
    train_loss = []
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)

      inputs = inputs.view(-1, 784)
      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
    train_loss = np.mean(train_loss)

    test_loss = []
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      inputs = inputs.view(-1, 784)

      outputs = model(inputs)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    # Save losses
    train_losses[it] = train_loss
    test_losses[it] = test_loss
    print(f'Epoch {it+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

  plot_loss(train_losses, test_losses)

  # test accuracy
  n_correct = 0.
  n_total = 0.
  for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    inputs = inputs.view(-1, 784)
    outputs = model(inputs)

    _, predictions = torch.max(outputs, 1)

    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

  train_acc = n_correct / n_total


  n_correct = 0.
  n_total = 0.
  for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    inputs = inputs.view(-1, 784)

    # Forward pass
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
    inputs = inputs.to(device)
    inputs = inputs.view(-1, 784)
    outputs = model(inputs)

    _, predictions = torch.max(outputs, 1)
    p_test = np.concatenate((p_test, predictions.cpu().numpy()))

  cm = confusion_matrix(y_test, p_test)
  plot_confusion_matrix(cm, list(range(10)))

  # misclassified detection

  misclassified_idx = np.where(p_test != y_test)[0]
  i = np.random.choice(misclassified_idx)
  plt.imshow(x_test[i], cmap='gray')
  plt.title("True label: %s Predicted: %s" % (y_test[i], int(p_test[i])))