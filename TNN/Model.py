import torch
import torch.nn as nn
import torch.nn.functional as F

class TNN(nn.Module):
    def __init__(self, input_shape=(), output_size=4):
        super(TNN, self).__init__()
        self.cnn1 = nn.Conv2d(input_shape[0], 128, (7,7))
        self.cnn2 = nn.Conv2d(128, 256, (5,5))
        self.pooling = nn.MaxPool2d((2,2), (2,2))
        self.CNN_outshape = self._get_conv_output(input_shape)
        self.linear = nn.Linear(self.CNN_outshape, output_size)
             
    def _get_conv_output(self, shape):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        x = self._forward_features(dummy_x)
        CNN_outshape = x.flatten(1).size(1)
        return CNN_outshape
    
    def _forward_features(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pooling(x)
        x = F.relu(self.cnn2(x))
        x = self.pooling(x)
        return x     
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.linear(x.flatten(1))
        return x
    
    
class TNN_CIFAR10(nn.Module):
    def __init__(self, input_shape=(), output_size=4):
        super(TNN_CIFAR10, self).__init__()
        self.cnn1 = nn.Conv2d(input_shape[0], 64, (5,5))
        self.cnn2 = nn.Conv2d(64, 128, (5,5))
        self.cnn3 = nn.Conv2d(128, 256, (5,5))
        self.pooling = nn.MaxPool2d((2,2), (1,1))
        self.CNN_outshape = self._get_conv_output(input_shape)
        self.linear = nn.Linear(self.CNN_outshape, output_size)
             
    def _get_conv_output(self, shape):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        x = self._forward_features(dummy_x)
        CNN_outshape = x.flatten(1).size(1)
        return CNN_outshape
    
    def _forward_features(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pooling(x)
        x = F.relu(self.cnn2(x))
        x = self.pooling(x)
        x = F.relu(self.cnn3(x))
        x = self.pooling(x)        
        return x     
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.linear(x.flatten(1))
        return x
    
    
