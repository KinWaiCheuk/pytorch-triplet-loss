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

    
class SimpleTNN(nn.Module):
    def __init__(self, input_shape=(), output_size=4):
        super(SimpleTNN, self).__init__()
        self.CNN_outshape = self._get_output(input_shape)
        self.linear = nn.Linear(self.CNN_outshape, output_size)
             
    def _get_output(self, shape):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        output_shape = dummy_x.flatten(1).size(1)
        return output_shape
     
        
    def forward(self, x):
        x = self.linear(x.flatten(1))
        return x 
    
class TNN_CIFAR10(nn.Module):
    def __init__(self, input_shape=(), output_size=4):
        super(TNN_CIFAR10, self).__init__()
        self.cnn1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cnn2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cnn3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
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
        x = self.cnn1(x)
        x = F.relu(self.bn1(x))
        x = self.pooling(x)
        
        x = self.cnn2(x)
        x = F.relu(self.bn2(x))
        x = self.pooling(x)
        
        x = self.cnn3(x)
        x = F.relu(self.bn3(x))
        x = self.pooling(x)        
        return x     
        
    def forward(self, x):
        x = self._forward_features(x)
        x = self.linear(x.flatten(1))
        return x
    
    
    
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
    
    
class VGG(nn.Module):
    def __init__(self, input_shape, vgg_name, output_size):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.CNN_outshape = self._get_conv_output(input_shape)
        self.classifier = nn.Linear(self.CNN_outshape, output_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _get_conv_output(self, shape):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        x = self._forward_features(dummy_x)
        CNN_outshape = x.flatten(1).size(1)
        return CNN_outshape
    
    def _forward_features(self, x):
        x = self.features(x)        
        return x      

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)