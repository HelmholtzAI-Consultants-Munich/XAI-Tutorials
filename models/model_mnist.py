############################################################
##### Imports
############################################################

import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


############################################################
# Model Class with a 2D CNN
############################################################


class CnnMnist(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for MNIST digit classification.

    This model consists of convolutional layers followed by fully connected layers.
    It is designed for grayscale (single-channel) 28x28 pixel images such as MNIST.

    :param None: No external parameters needed during initialization.
    :type None: None
    """

    def __init__(self):
        """Constructor for CnnMnist class"""
        super(CnnMnist, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50), nn.ReLU(), nn.Dropout(), nn.Linear(50, 10), nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the CNN model.

        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, 10) representing class probabilities.
        :rtype: torch.Tensor
        """
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x


##############################################
### training and evaluation functions
##############################################


def get_trained_model(nb_of_epochs=5, seed=2, device="cpu"):
    """
    Trains a simple CNN on the MNIST dataset and returns the trained model and test loader.

    This function sets up the data loaders, initializes a CNN model,
    trains it for a specified number of epochs, and evaluates it after each epoch.

    :param nb_of_epochs: Number of epochs to train the model. Default is 5.
    :type nb_of_epochs: int
    :param seed: Random seed for reproducibility. Default is 2.
    :type seed: int
    :param device: Device on which to run the model ('cpu' or 'cuda'). Default is 'cpu'.
    :type device: str

    :return: A tuple containing the trained model and the DataLoader for the test dataset.
    :rtype: Tuple[nn.Module, DataLoader]
    """

    def _train(model, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    def _test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output.log(), target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
        )

    # Set seed and data loaders
    torch.manual_seed(seed)
    transform = transforms.ToTensor()
    batch_size = 128

    train_loader = DataLoader(
        datasets.MNIST("mnist_data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.MNIST("mnist_data", train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False,  # More appropriate for evaluation
    )

    # Initialize model and optimizer
    model = CnnMnist().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Training loop
    for epoch in range(1, nb_of_epochs + 1):
        _train(model, train_loader, optimizer, epoch)
        _test(model, test_loader)

    return model, test_loader
