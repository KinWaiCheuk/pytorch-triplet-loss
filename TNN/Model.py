import torch.nn as nn
import torch.nn.functional as F

class TNN(nn.Module):
    def __init__(self, output_size=4):
        super(TNN, self).__init__()
        self.cnn1 = nn.Conv2d(1, 128, (7,7))
        self.cnn2 = nn.Conv2d(128, 256, (5,5))
        self.pooling = nn.MaxPool2d((2,2), (2,2))
        
        self.linear = nn.Linear(256*3*3, output_size)
        
    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pooling(x)
        x = F.relu(self.cnn2(x))
        x = self.pooling(x)
        x = self.linear(x.flatten(1))
        return x
    
    
