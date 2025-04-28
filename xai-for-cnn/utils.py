############################################################
##### Imports
############################################################

import os
import cv2 as cv
import numpy as np

from PIL import Image

import torch
from torch.nn import functional as F
from torchvision import transforms


############################################################
##### Utility Fuctions
############################################################


def transform_img(img, mean, std, tensor_flag=True, img_size=(224, 224)):
    """
    Applies transformations to an input image including resizing, normalization, and optional tensor conversion.

    :param img: Input image to be transformed.
    :type img: np.ndarray or PIL.Image
    :param mean: Mean values for normalization for each channel.
    :type mean: list or tuple
    :param std: Standard deviation values for normalization for each channel.
    :type std: list or tuple
    :param tensor_flag: Whether to return the output as a PyTorch tensor. If False, returns a NumPy array. Default is True.
    :type tensor_flag: bool
    :param img_size: Size to which the image should be resized (height, width). Default is (224, 224).
    :type img_size: tuple

    :return: Transformed image as a tensor or NumPy array, depending on the `tensor_flag`.
    :rtype: torch.Tensor or np.ndarray
    """
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    arr_img = np.array(img)
    # apply the transforms
    trans_img = transform(arr_img)
    # unsqueeze to add a batch dimension
    trans_img = trans_img.unsqueeze(0)
    if tensor_flag is False:
        # returns np.array with original axes
        trans_img = np.array(trans_img)
        trans_img = trans_img.swapaxes(-1, 1).swapaxes(1, 2)

    return trans_img


def normalize_and_adjust_axes(image, mean, std):
    """
    Normalizes the input image using the specified mean and standard deviation and adjusts its axes to PyTorch format.

    :param image: Input image array.
    :type image: np.ndarray
    :param mean: Mean values for normalization for each channel.
    :type mean: list or tuple
    :param std: Standard deviation values for normalization for each channel.
    :type std: list or tuple

    :return: Normalized image tensor with adjusted axes.
    :rtype: torch.Tensor
    """
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axes so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


def read_image_cv(path_to_img):
    """
    Reads an image from the specified path using OpenCV and converts it from BGR to RGB format.

    :param path_to_img: Path to the image file.
    :type path_to_img: str

    :return: Loaded image in RGB format.
    :rtype: np.ndarray
    """
    img = cv.imread(path_to_img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def read_image_PIL(path_to_img):
    """
    Reads an image from the specified path using PIL and converts it to RGB format.

    :param path_to_img: Path to the image file.
    :type path_to_img: str

    :return: Loaded image as a PIL Image object in RGB mode.
    :rtype: PIL.Image.Image
    """
    with open(os.path.abspath(path_to_img), "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def get_pil_transform():
    """
    Returns a PIL image transformation pipeline that resizes and center crops the image.

    :return: Transformations to apply on a PIL image.
    :rtype: torchvision.transforms.Compose
    """
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])
    return transform


def get_preprocess_transform():
    """
    Returns a preprocessing transformation pipeline that converts an image to a tensor and normalizes it.

    :return: Preprocessing transformations for an image.
    :rtype: torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([transforms.ToTensor(), normalize])

    return transf


def batch_predict(images, model):
    """
    Predicts class probabilities for a batch of images using the provided model.

    :param images: List of input images to be predicted.
    :type images: list of PIL.Image or np.ndarray
    :param model: Trained model used for prediction.
    :type model: torch.nn.Module

    :return: Predicted class probabilities for the input batch.
    :rtype: np.ndarray
    """
    # Set the model in evaluation mode
    model.eval()
    # Prepare a batch of preprocessed images
    preprocess_transform = get_preprocess_transform()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    # Move the model and batch to the selected device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    # Make predictions using the model
    logits = model(batch)
    # Convert logits to class probabilities using softmax
    probs = F.softmax(logits, dim=1)
    # Return the predicted probabilities as a NumPy array
    return probs.detach().cpu().numpy()


def calculate_localization_map(model, input, c, dim):
    """
    Calculates a localization map (e.g., Grad-CAM) for a given class index based on model activations and gradients.

    :param model: Trained model from which activations and gradients are extracted.
    :type model: torch.nn.Module
    :param input: Input tensor to the model.
    :type input: torch.Tensor
    :param c: Target class index for which the localization map is computed.
    :type c: int
    :param dim: Dimensions along which to compute the mean of gradients.
    :type dim: tuple or list

    :return: Computed localization map as a NumPy array.
    :rtype: np.ndarray
    """
    # forward pass
    logits = model(input)
    feat_maps = model.get_activations(input).detach()

    # compute gradients
    logits[:, c].backward(retain_graph=True)
    gradients = model.get_gradient()

    # compute localization map
    pooled_gradients = torch.mean(gradients, dim=dim)

    for i in range(feat_maps.size(1)):
        if len(dim) == 2:
            feat_maps[:, i, :] *= pooled_gradients[i]
        elif len(dim) == 3:
            feat_maps[:, i, :, :] *= pooled_gradients[i]

    localization_map = torch.sum(feat_maps, dim=1).squeeze()
    localization_map = localization_map.numpy()
    localization_map = np.maximum(localization_map, 0)

    return localization_map


def convert_localization_map_to_heatmap(localization_map, target_size):
    """
    Converts a localization map into a heatmap resized to the given target size.

    :param localization_map: The localization (importance) map, usually output from Grad-CAM.
    :type localization_map: numpy.ndarray
    :param target_size: Target size as (width, height) tuple or based on an image or signal shape.
    :type target_size: tuple(int, int)
    :return: Heatmap normalized to [0, 255] as uint8.
    :rtype: numpy.ndarray
    """
    # Normalize the localization map
    localization_map = localization_map / np.max(localization_map)

    # Ensure 2D format if needed
    if localization_map.ndim == 1:
        localization_map = np.expand_dims(localization_map, axis=0)
        heatmap = cv.resize(localization_map, target_size)
    else:
        heatmap = cv.resize(localization_map.squeeze(), target_size)

    # Convert to 0-255 heatmap
    heatmap = np.uint8(255 * heatmap)
    return heatmap
