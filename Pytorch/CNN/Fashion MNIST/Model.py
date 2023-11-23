import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMNIST(nn.Module):
    def __init__ (self):
        super(FashionMNIST, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=64*6*6, out_features=600),
            nn.Dropout(0.3),
            nn.Linear(in_features=600, out_features=120),
            nn.Linear(in_features=120, out_features=10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
