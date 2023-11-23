import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
print("All features name in this data: ")
print(data.info())
datanp = np.array(data)
X = datanp[:, :12]
labels = datanp[:, -1]
print(X.shape)
print(labels.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)
number, feat = X_train.shape
print('Number of features: ', feat)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float).view(-1, 1)


# Define model
model = nn.Sequential(
    nn.Linear(feat, 24), 
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid()
)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
loss_val = []
train_acc = []
val_acc = []
for epoch in range(180):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss_val.append(loss.item())
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    with torch.no_grad():
        model.eval()
        train_acc.append((output.round() == y_train).float().mean())
        val_output = model(X_test)
        val_acc.append((val_output.round() == y_test).float().mean())

# Print scores
print(
    "\n Model train score: ", (model(X_train).round() == y_train).float().mean().item()
)
print("\n Model test score: ", (model(X_test).round() == y_test).float().mean().item())

# Plot accuracy
plt.plot(train_acc)
plt.plot(val_acc)
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

# Plot loss
plt.plot(loss_val)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()