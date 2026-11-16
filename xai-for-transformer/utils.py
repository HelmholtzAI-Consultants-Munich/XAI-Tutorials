############################################################
##### Imports
############################################################


import sys
import types
from functools import partial

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch import nn
from transformers.models.distilbert import modeling_distilbert
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
##### Utility Functions Layer-wise Relevance Propagation
############################################################


def skip_lxt_bundled_models():
    """
    Allow LXT to be imported together with transformers 5.

    Importing lxt.efficient (lxt 2.1) also imports LXT's bundled copies of the BERT, Llama, GPT-2, ... model files.
    Those copies use helpers that transformers 5 removed, so the import fails. The DistilBERT patch passes its own
    patch map and never uses the bundled models, so an empty stand-in is registered for them before LXT is imported.
    """
    if "lxt.efficient.models" in sys.modules:
        return

    def get_default_map(module):
        raise ValueError("LXT's bundled model patches are disabled; pass a patch_map.")

    stand_in = types.ModuleType("lxt.efficient.models")
    stand_in.get_default_map = get_default_map
    sys.modules["lxt.efficient.models"] = stand_in


def attnlrp_attention_forward(self, hidden_states, attention_mask=None, **kwargs):
    """
    DistilBERT self-attention with the AttnLRP rules. Replaces DistilBertSelfAttention.forward.

    The forward pass computes the same eager attention as transformers 5.13.0. In the backward pass, relevance is
    split equally between the two factors of each matrix multiplication (uniform rule).

    :param hidden_states: Token representations entering the attention layer.
    :type hidden_states: torch.Tensor
    :param attention_mask: Additive (eager) or boolean (SDPA) attention mask, or None if no token is padded.
    :type attention_mask: torch.Tensor
    """
    # LXT is only installed for the LRP tutorials, so it is imported here and not at the top of this file
    from lxt.efficient.rules import divide_gradient

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)

    # Project the tokens to queries, keys and values and split them into heads
    query_states = self.q_lin(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_lin(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_lin(hidden_states).view(hidden_shape).transpose(1, 2)

    # Attention scores, with the uniform rule for the query-key product
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    scores = divide_gradient(scores, 2)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            # SDPA-style mask: True marks positions that may be attended to
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        else:
            # Eager-style additive mask: 0 for kept positions, a large negative value otherwise
            scores = scores + attention_mask

    weights = nn.functional.softmax(scores, dim=-1)
    weights = self.dropout(weights)

    # Weighted sum of the values, with the uniform rule for the weights-values product
    context = torch.matmul(weights, value_states)
    context = divide_gradient(context, 2)
    context = context.transpose(1, 2).reshape(*input_shape, -1)
    return self.out_lin(context), weights


def attnlrp_ff_chunk(self, inputs):
    """
    DistilBERT feed-forward block with the AttnLRP identity rule for the activation. Replaces FFN.ff_chunk.

    :param inputs: Token representations entering the feed-forward block.
    :type inputs: torch.Tensor
    """
    # LXT is only installed for the LRP tutorials, so it is imported here and not at the top of this file
    from lxt.efficient.rules import identity_rule_implicit

    hidden = self.lin1(inputs)
    hidden = identity_rule_implicit(self.activation, hidden)
    return self.dropout(self.lin2(hidden))


def patch_distilbert_for_attnlrp(verbose=False):
    """
    Patch DistilBERT's attention, feed-forward, LayerNorm and dropout layers for LXT AttnLRP.

    The patch only changes the backward pass, so the model's predictions stay the same.

    :param verbose: Whether to print the name of every patched class.
    :type verbose: bool
    """
    # LXT is only installed for the LRP tutorials, so it is imported here and not at the top of this file
    skip_lxt_bundled_models()
    from lxt.efficient import monkey_patch
    from lxt.efficient.patches import dropout_forward, layer_norm_forward, patch_method

    patch_map = {
        nn.LayerNorm: partial(patch_method, layer_norm_forward),
        nn.Dropout: partial(patch_method, dropout_forward),
        modeling_distilbert.DistilBertSelfAttention: partial(patch_method, attnlrp_attention_forward),
        modeling_distilbert.FFN: partial(patch_method, attnlrp_ff_chunk, method_name="ff_chunk"),
    }
    monkey_patch(modeling_distilbert, patch_map=patch_map, verbose=verbose)


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
