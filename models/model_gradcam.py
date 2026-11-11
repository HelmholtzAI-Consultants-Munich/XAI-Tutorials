############################################################
# Imports
############################################################

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

############################################################
# Model Class
############################################################


class GradCamModel(nn.Module):
    """
    A ResNet50-based neural network model adapted for Grad-CAM visualization.

    This class modifies a standard ResNet50 architecture to support gradient-based
    Class Activation Mapping (Grad-CAM) by registering a backward hook to capture
    gradients with respect to the activations of the last convolutional layer.

    :param None: The model is initialized with pretrained ResNet50 weights and restructured
                 for Grad-CAM support.
    """

    def __init__(self):
        """Constructor for GradCamModel class"""
        super(GradCamModel, self).__init__()

        # define the resnet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # isolate the feature blocks
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        )

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc

        # gradient placeholder
        self.gradient = None

    def activations_hook(self, grad):
        """
        Hook function to save gradients from the target layer during backpropagation.

        :param grad: The gradients of the target layer.
        :type grad: torch.Tensor
        """
        self.gradient = grad

    def get_gradient(self):
        """
        Returns the saved gradients captured by the registered hook.

        :return: Gradient of the last convolutional layer.
        :rtype: torch.Tensor
        """
        return self.gradient

    def get_activations(self, x):
        """
        Extracts and returns the activations from the last convolutional layer.

        :param x: The input image tensor.
        :type x: torch.Tensor
        :return: Activations from the last convolutional layer.
        :rtype: torch.Tensor
        """
        return self.features(x)

    def forward(self, x):
        """
        Performs the forward pass through the network and registers a hook to capture gradients.

        :param x: Input tensor representing the image batch.
        :type x: torch.Tensor
        :return: Output logits from the final classification layer.
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x.register_hook(self.activations_hook)
        x = self.avgpool(x)
        x = x.view((x.size(0), -1))  # x = x.view((1, -1))
        x = self.classifier(x)

        return x
