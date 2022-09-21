from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# GradCamModel class with a ResNet pretrained model
class GradCamModel(nn.Module):
    def __init__(self):
        super(GradCamModel, self).__init__()
        
        # define the resnet50
        # self.resnet = resnet50(pretrained=True)
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      self.resnet.relu,
                                      self.resnet.maxpool, 
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc

        # gradient placeholder
        self.gradient = None
    
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        
        # extract the features
        x = self.features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x


