############################################################
##### Imports
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2 as cv

import torch
from torchvision import transforms

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


def transform_img(img, mean, std, tensor_flag=True):
    transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])
    arr_img = np.array(img)
    # apply the transforms
    trans_img = transform(arr_img)
    # unsqueeze to add a batch dimension
    trans_img = trans_img.unsqueeze(0)
    if tensor_flag is False:
        # returns np.array with original axes
        trans_img = np.array(trans_img)
        trans_img = trans_img.swapaxes(-1,1).swapaxes(1, 2)

    return trans_img


def normalize_and_adjust_axes(image, mean, std):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axes so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


def read_img(path_to_img):
    img = cv.imread(path_to_img) # Insert the path to image.
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def calculate_localization_map(gcmodel, img, out, c):

    # Step 1 - Gradient output y wrt. to activation map
    # get the gradient of the output with respect to the parameters of the model
    out[:,c].backward(retain_graph=True)
    # pull the gradients out of the model
    gradients = gcmodel.get_gradient()

    # Step 2 - Global average pooling
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) #to be computed by students

    # Step 3 - Weighted combination of influence and feature maps
    # get the activations of the last convolutional layer
    activations = gcmodel.get_activations(img).detach()
    # weight the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    localization_map = torch.sum(activations, dim=1).squeeze()
    # convert the map to be a numpy array
    localization_map = localization_map.numpy()
    # relu on top of the localization map
    localization_map = np.maximum(localization_map, 0) #to be computed by students

    return localization_map


def convert_to_heatmap(localization_map, img):
    # normalize the localization_map
    localization_map /= np.max(localization_map)
    # resize to image size
    heatmap = cv.resize(localization_map, (img.shape[1], img.shape[0]))
    # normalize to [0, 255] range and convert to unsigned int
    heatmap = np.uint8(255 * heatmap)
    return heatmap