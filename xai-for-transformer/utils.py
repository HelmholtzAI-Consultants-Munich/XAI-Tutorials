############################################################
##### Imports
############################################################


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torchvision import transforms

from sklearn.metrics import balanced_accuracy_score

import shap

############################################################
##### Utility Fuctions Attention Maps for Text
############################################################


# Function that plot the attention map
def showAttention(input_sentence, output_words, attentions):
    """
    Plot the attention weights between an input sentence and generated output words.

    :param input_sentence: List of words from the input sentence.
    :type input_sentence: list
    :param output_words: List of predicted output words.
    :type output_words: list
    :param attentions: Attention weights matrix.
    :type attentions: torch.Tensor
    """
    fig, ax = plt.subplots(figsize=(20, 5))
    cax = ax.matshow(attentions.detach().numpy(), cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([" "] + input_sentence, rotation=90)
    ax.set_yticklabels([" "] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


############################################################
##### Utility Fuctions Attention Maps for Images
############################################################


def transform_img(img, mean, std, tensor_flag=True, img_size=(224, 224)):
    """
    Apply transformations to an image including resizing, normalization, and optional tensor conversion.

    :param img: Input image as a NumPy array.
    :type img: numpy.ndarray
    :param mean: Mean values for normalization (per channel).
    :type mean: list
    :param std: Standard deviation values for normalization (per channel).
    :type std: list
    :param tensor_flag: Whether to return the output as a tensor or NumPy array.
    :type tensor_flag: bool
    :param img_size: Target image size (height, width).
    :type img_size: tuple
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


def read_image_cv(path_to_img):
    """
    Read an image from a path using OpenCV and convert it from BGR to RGB format.

    :param path_to_img: Path to the input image file.
    :type path_to_img: str
    """
    img = cv.imread(path_to_img)  # Insert the path to image.
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def plot_attention_maps(img, attentions):
    """
    Plot the original image alongside mean attention maps and per-head attention maps.

    :param img: The original input image.
    :type img: numpy.ndarray
    :param attentions: Attention maps from different heads.
    :type attentions: numpy.ndarray
    """
    n_heads = attentions.shape[0]

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attentions, 0)]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(fig)
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads // 3, 3, i + 1)
        plt.imshow(attentions[i])
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()


############################
### SHAP helper functions
############################


def evaluate_model(classifier_pipeline, data, label2id):
    # Run batched inference on the provided texts. For each text, the pipeline
    # returns the prediction scores for all emotion classes. Batching improves
    # inference speed, while truncation ensures that texts exceeding the model's
    # maximum input length are safely shortened.
    predictions = classifier_pipeline(
        data["text"].tolist(),
        batch_size=128,
        truncation=True,
    )

    # Select the highest-scoring predicted emotion for each text, convert the
    # emotion labels to their numerical class IDs, and compare them with the
    # ground-truth labels to compute the balanced classification accuracy.
    predicted_ids = [label2id[max(p, key=lambda x: x["score"])["label"]] for p in predictions]
    accuracy = balanced_accuracy_score(data["emotion"], predicted_ids)
    return accuracy


def plot_shap_values(shap_values_1, shap_values_2, output_name, max_display=20):

    # Aggregate SHAP values
    train = shap_values_1[:, :, output_name].mean(0)
    test = shap_values_2[:, :, output_name].mean(0)

    # Top 20 tokens based on absolute SHAP values (train)
    idx = np.argsort(np.abs(train.values))[-max_display:]

    # Sort by absolute SHAP value
    idx = idx[np.argsort(np.abs(train.values[idx]))]

    # Look up the corresponding SHAP values in the test set
    test_dict = dict(zip(test.feature_names, test.values))
    test_values = np.array([test_dict.get(token, 0) for token in train.feature_names[idx]])

    # Colors: red = positive, blue = negative
    train_colors = [
        shap.plots.colors.red_rgb if v >= 0 else shap.plots.colors.blue_rgb for v in train.values[idx]
    ]
    test_colors = [shap.plots.colors.red_rgb if v >= 0 else shap.plots.colors.blue_rgb for v in test_values]

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    axes[0].barh(train.feature_names[idx], train.values[idx], color=train_colors)
    axes[0].set_title("Train")
    axes[0].set_xlabel("Mean SHAP value")
    axes[0].axvline(0, color="black", linewidth=1)

    axes[1].barh(train.feature_names[idx], test_values, color=test_colors)
    axes[1].set_title("Test")
    axes[1].set_xlabel("Mean SHAP value")
    axes[1].axvline(0, color="black", linewidth=1)

    # Use the same symmetric x-axis for both plots
    xmax = max(np.max(np.abs(train.values[idx])), np.max(np.abs(test_values)))

    axes[0].set_xlim(-xmax, xmax)
    axes[1].set_xlim(-xmax, xmax)

    plt.tight_layout()
    plt.show()
