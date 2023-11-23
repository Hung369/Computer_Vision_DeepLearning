import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model = nn.Sequential(nn.Linear(1,1))

    def forward(self,X):
        X = self.model(X)
        return X

class Breast_Cancer_Detection(nn.Module):
    def __init__(self):
        super(Breast_Cancer_Detection,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 1),
            nn.Sigmoid()
        )
    
    def forward(self,X):
        X = self.model(X)
        return X