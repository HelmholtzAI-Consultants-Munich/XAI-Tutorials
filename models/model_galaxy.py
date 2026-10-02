##############################################
### Imports
##############################################

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

##############################################
### Galaxy morphology model
##############################################

#: Galaxy MNIST morphology classes, in label order.
GALAXY_CLASSES = ["smooth round", "smooth cigar", "edge-on disk", "unbarred spiral"]


class GalaxyNet(nn.Module):
    """
    ResNet-18 adapted for galaxy morphology classification.

    The convolutional backbone is a ResNet-18 pretrained on ImageNet, with the
    classification head replaced by a linear layer over the four Galaxy MNIST
    classes. Only the head is trained from scratch, the backbone is fine-tuned.

    The feature extractor is kept as a separate attribute so that the activations
    of the last convolutional block, and the gradients flowing into them, can be
    read out for methods such as Grad-CAM.

    :param num_classes: Number of output classes.
    :type num_classes: int
    """

    def __init__(self, num_classes=4):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.avgpool = net.avgpool
        self.classifier = nn.Linear(net.fc.in_features, num_classes)
        self.gradients = None

    def activations_hook(self, grad):
        """
        Hook that stores the gradients flowing into the last convolutional block.

        :param grad: Gradients with respect to the feature maps.
        :type grad: torch.Tensor
        """
        self.gradients = grad

    def get_gradient(self):
        """
        Returns the gradients stored by :meth:`activations_hook`.

        :return: Gradients with respect to the feature maps, or None before a backward pass.
        :rtype: torch.Tensor or None
        """
        return self.gradients

    def get_activations(self, x):
        """
        Returns the feature maps of the last convolutional block.

        :param x: Input batch.
        :type x: torch.Tensor

        :return: Feature maps.
        :rtype: torch.Tensor
        """
        return self.features(x)

    def forward(self, x):
        """
        Runs the model and returns the class logits.

        :param x: Input batch of shape (N, 3, 224, 224).
        :type x: torch.Tensor

        :return: Logits of shape (N, num_classes).
        :rtype: torch.Tensor
        """
        x = self.features(x)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        return self.classifier(torch.flatten(self.avgpool(x), 1))


def load_galaxy_model(path_to_weights, num_classes=4):
    """
    Builds the galaxy model and loads the fine-tuned weights.

    :param path_to_weights: Path to the state dict saved during fine-tuning.
    :type path_to_weights: str
    :param num_classes: Number of output classes.
    :type num_classes: int

    :return: The model in evaluation mode.
    :rtype: GalaxyNet
    """
    model = GalaxyNet(num_classes=num_classes)
    model.load_state_dict(torch.load(path_to_weights, map_location="cpu"))
    model.eval()
    return model
