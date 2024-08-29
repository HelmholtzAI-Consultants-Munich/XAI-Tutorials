############################################################
##### Imports
############################################################


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torchvision import transforms


############################################################
##### Utility Fuctions Attention Maps for Text
############################################################


# Function that plot the attention map
def showAttention(input_sentence, output_words, attentions):
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


def read_img(path_to_img):
    img = cv.imread(path_to_img)  # Insert the path to image.
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def plot_attention_maps(img, attentions):
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
