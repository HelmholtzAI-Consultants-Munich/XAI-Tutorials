##############################################
### Imports
##############################################

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

##############################################
### ECG dataloader class
##############################################


class ECG(data.Dataset):
    """
    ECG dataset class for loading and processing ECG signals from a CSV file.

    This class reads ECG data from a CSV file where the first 187 columns represent the ECG signal
    and the 188th column represents the target class label. It provides methods for retrieving
    individual samples and the length of the dataset, making it compatible with PyTorch's DataLoader.

    :param data: Path to the CSV file containing ECG samples and labels.
    :type data: str
    """

    def __init__(self, data):
        """Constructor for ECG class"""
        self.data = pd.read_csv(data, sep=",", header=None)
        self.samples = self.data.iloc[:, :187]
        self.targets = self.data[187].to_numpy()

    def __getitem__(self, index):
        """
        Retrieves the sample and label at the specified index.

        :param index: Index of the sample to retrieve.
        :type index: int
        :return: Tuple containing the ECG signal as a tensor and the corresponding label.
        :rtype: Tuple[torch.Tensor, int]
        """
        x = self.samples.iloc[index, :]
        x = torch.from_numpy(x.values).float()
        x = torch.unsqueeze(x, 0)
        y = self.targets[index].astype(np.int64)
        return x, y

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.data)


##############################################
### ResNet model
##############################################


class ResNetBlockECG(nn.Module):
    """
    A residual block for 1D convolutional ResNet architecture.

    This block contains two convolutional layers with batch normalization and ReLU activation.
    A shortcut connection is added either as identity or with a 1x1 convolution when downsampling is required.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param out_channels: Number of output channels.
    :type out_channels: int
    :param stride: Stride for the first convolution layer; used to reduce the temporal resolution.
    :type stride: int
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """Constructor for ResNetBlockECG class"""
        super(ResNetBlockECG, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Applies two convolutional layers with batch normalization and ReLU activation,
        and adds the shortcut connection to form the residual output.

        :param x: Input tensor of shape (batch_size, in_channels, sequence_length).
        :type x: torch.Tensor
        :return: Output tensor after applying residual block.
        :rtype: torch.Tensor
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNetECG(nn.Module):
    """
    A ResNet model for 1D convolutional data.

    This model stacks several residual blocks for feature extraction followed by an average pooling and a fully connected layer for classification.
    Optionally supports extraction of intermediate activations and class activation maps (CAM).

    :param block: Residual block type to use in the network.
    :type block: nn.Module
    :param layers: A list defining the number of blocks in each layer.
    :type layers: list
    :param num_classes: Number of output classes for classification.
    :type num_classes: int
    """

    def __init__(self, block, layers, num_classes=10):
        """Constructor for ResNetECG class"""
        super(ResNetECG, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Linear(512, num_classes)
        self.gradient = None

    def make_layer(self, block, out_channels, blocks, stride=1):
        """
        Constructs a sequential layer consisting of multiple residual blocks.

        :param block: Residual block to use.
        :type block: nn.Module
        :param out_channels: Number of output channels for each block in this layer.
        :type out_channels: int
        :param blocks: Number of residual blocks to stack.
        :type blocks: int
        :param stride: Stride for the first block in this layer.
        :type stride: int
        :return: A sequential container of residual blocks.
        :rtype: nn.Sequential
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def activations_hook(self, grad):
        """
        Hook function to store gradients from the final convolutional layer during backpropagation.

        :param grad: Gradient tensor from the output layer.
        :type grad: torch.Tensor
        """
        self.gradient = grad

    def get_gradient(self):
        """
        Returns the stored gradients from the most recent backward pass.

        :return: Gradient tensor.
        :rtype: torch.Tensor or None
        """
        return self.gradient

    def get_activations(self, x):
        """
        Returns the output activations from the final feature map layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Feature map before global pooling.
        :rtype: torch.Tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, label=None, return_cam=False):
        """
        Forward pass through the ResNet model. Optionally returns class activation maps (CAM).

        :param x: Input tensor of shape [batch_size, 1, sequence_length].
        :type x: torch.Tensor
        :param label: Class label index used to compute CAM (required if return_cam is True).
        :type label: int or None
        :param return_cam: Whether to return CAM alongside logits.
        :type return_cam: bool
        :return: Logits, and optionally CAM.
        :rtype: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x.register_hook(self.activations_hook)

        pre_logits = self.avg_pool(x)
        pre_logits = torch.flatten(pre_logits, 1)
        logits = self.fc(pre_logits)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[label]
            cams = (cam_weights.view(*feature_map.shape[:2], 1) * feature_map).mean(1, keepdim=False)
            return logits, cams

        return logits


##############################################
### training and evaluation functions
##############################################


def training_loop(model, model_weights_path, train_loader, criterion, optimizer, num_epochs, writer, device):
    """
    Training loop for a PyTorch model. Trains the model for a specified number of epochs,
    logs training loss to TensorBoard, and saves model weights after each epoch.

    :param model: The PyTorch model to train.
    :type model: torch.nn.Module
    :param model_weights_path: File path where the final model weights should be saved.
    :type model_weights_path: str
    :param train_loader: DataLoader providing the training data.
    :type train_loader: torch.utils.data.DataLoader
    :param criterion: Loss function used during training.
    :type criterion: torch.nn.modules.loss._Loss
    :param optimizer: Optimizer used for updating model weights.
    :type optimizer: torch.optim.Optimizer
    :param num_epochs: Number of training epochs.
    :type num_epochs: int
    :param writer: TensorBoard SummaryWriter to log training metrics.
    :type writer: torch.utils.tensorboard.SummaryWriter
    :param device: Device on which training is performed (e.g. "cuda" or "cpu").
    :type device: torch.device
    """
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs[0], labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))

        directory = os.path.dirname(model_weights_path)
        base_filename = os.path.splitext(os.path.basename(model_weights_path))[0]
        save_path = os.path.join(directory, base_filename + f"_epoch{epoch}.pth")

        torch.save(model.state_dict(), save_path)

        writer.add_scalar("training loss", running_loss / len(train_loader), epoch)

    torch.save(model.state_dict(), model_weights_path)


def inference_loop(model, test_loader, criterion):
    """
    Inference and evaluation loop for a trained PyTorch model. Computes accuracy, precision,
    recall, F1 score for 5 classes, and the average loss on the test dataset.

    :param model: The trained PyTorch model for inference.
    :type model: torch.nn.Module
    :param test_loader: DataLoader providing the test data.
    :type test_loader: torch.utils.data.DataLoader
    :param criterion: Loss function used during evaluation.
    :type criterion: torch.nn.modules.loss._Loss

    :return: Tuple containing:
        - accuracy (float): Overall accuracy across all classes.
        - recall (list[float]): Per-class recall scores.
        - precision (list[float]): Per-class precision scores.
        - f1 (list[float]): Per-class F1 scores.
        - average_loss (float): Average loss across all test samples.
    :rtype: tuple
    """
    model.eval()

    device = next(model.parameters()).device
    true_positives = [0] * 5
    false_positives = [0] * 5
    false_negatives = [0] * 5
    true_negatives = [0] * 5
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs[0]

            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_samples += inputs.size(0)
            _, predictions = torch.max(outputs, -1)

            for i in range(5):

                true_positives[i] += ((predictions == i) & (targets == i)).sum().item()
                false_positives[i] += ((predictions == i) & (targets != i)).sum().item()
                false_negatives[i] += ((predictions != i) & (targets == i)).sum().item()
                true_negatives[i] += ((predictions != i) & (targets != i)).sum().item()

    accuracy = (sum(true_positives) + sum(true_negatives)) / (
        sum(true_positives) + sum(false_positives) + sum(false_negatives) + sum(true_negatives)
    )
    recall = [
        (
            true_positives[i] / (true_positives[i] + false_negatives[i])
            if true_positives[i] + false_negatives[i] > 0
            else 0
        )
        for i in range(5)
    ]
    precision = [
        (
            true_positives[i] / (true_positives[i] + false_positives[i])
            if true_positives[i] + false_positives[i] > 0
            else 0
        )
        for i in range(5)
    ]
    f1 = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0
        for i in range(5)
    ]
    average_loss = total_loss / total_samples

    return accuracy, recall, precision, f1, average_loss


##############################################
### main
##############################################

if __name__ == "__main__":

    model_weights_path = "model_final_weights_ecg.pth"
    train_data_path = "data/Dataset_ECG/mitbih_train.csv"
    test_data_path = "data/Dataset_ECG/mitbih_test.csv"
    num_epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNetECG(ResNetBlockECG, [2, 2, 2, 2], num_classes=5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ecg = ECG(train_data_path)
    train_loader = DataLoader(ecg, batch_size=64, shuffle=True, num_workers=4)

    ecg = ECG(test_data_path)
    test_loader = DataLoader(ecg, batch_size=64, shuffle=True, num_workers=4)

    writer = SummaryWriter()

    training_loop(model, model_weights_path, train_loader, criterion, optimizer, num_epochs, writer, device)

    model.load_state_dict(torch.load(model_weights_path))

    accuracy, recall, precision, f1, average_loss = inference_loop(model, test_loader, criterion)

    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1: ", f1)
    print("Average loss: ", average_loss)
