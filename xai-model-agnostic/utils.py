############################################################
##### Imports
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.nn import functional as F
import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim

############################################################
##### Utility Fuctions
############################################################


def plot_permutation_feature_importance(result, data, title):
    perm_sorted_idx = result.importances_mean.argsort()
    perm_indices = np.arange(0, len(result.importances_mean)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    ax1.barh(
        perm_indices,
        result.importances_mean[perm_sorted_idx],
        height=0.7,
        color="#3470a3",  # color = 'cornflowerblue'
    )
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result.importances_mean)))
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=data.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()


def plot_permutation_feature_importance_with_variance(result, data, title):
    perm_sorted_idx = result.importances_mean.argsort()
    perm_indices = np.arange(0, len(result.importances_mean)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    ax1.barh(
        perm_indices,
        result.importances_mean[perm_sorted_idx],
        height=0.7,
        color="cornflowerblue",
    )
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result.importances_mean)))
    ax1.axvline(x=0, color=".5")
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=data.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()


def plot_impurity_feature_importance(importance, names, title):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(5, 4))
    # Plot Searborn bar chart
    sns.barplot(
        x=fi_df["feature_importance"], y=fi_df["feature_names"], color="#3470a3"
    )
    # Add chart labels
    plt.title(title)
    plt.xlabel("feature importance")
    plt.ylabel("feature names")


def plot_explanation(explanation):
    explanation_df = pd.DataFrame(
        {k: v for k, v in explanation.items() if k != "importances"}
    ).sort_values(by="importances_mean", ascending=True)

    f, ax = plt.subplots(1, 1, figsize=(9, 7))
    explanation_df.plot(kind="barh", ax=ax)
    plt.title("Permutation importances")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)

    if "feature" in explanation_df:
        _ = ax.set_yticklabels(explanation_df["feature"])


def plot_correlation_matrix(data):
    f, ax = plt.subplots(figsize=(5, 5))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    sns.heatmap(
        round(corr, 2),
        mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=True,
    )


#### Data loader and function for training on MNST###

def train_loader_fun(num_epochs = 5):
    batch_size = 128
    #device = torch.device('cpu')
    return torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)
    # return train_loader
    
def test_loader_fun(num_epochs = 5):
    batch_size = 128
    #device = torch.device('cpu')   
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)
    return test_loader
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))