Sanity Checks for Saliency Maps
=========================================

Saliency methods highlight the input features that are most relevant for a model's prediction and are widely used to explain image classifiers.
Their quality is often assessed by visual inspection only: a saliency map that looks reasonable to a human is considered a good explanation.

Adebayo et al. (2018) showed that visual inspection can be misleading. Some saliency methods produce maps that look convincing
but are largely independent of the model and of the data it was trained on, behaving similarly to an edge detector applied to the input image.
To detect this behaviour, the authors propose two randomization tests:

1. **Model parameter randomization test:** the explanation should change when the model parameters are randomized.
2. **Data randomization test:** the explanation should change when the model is trained on data with randomly permuted labels.

A method that fails these tests is insensitive to properties that an explanation is expected to depend on,
and therefore cannot be helpful for tasks such as debugging the model.

**Notation**

Let :math:`\mathbf{x} \in \mathbb{R}^d` be the input image and :math:`S_c(\mathbf{x})` the score (logit) of class :math:`c`.
A saliency method returns an explanation map :math:`E(\mathbf{x}) \in \mathbb{R}^d` that assigns a relevance value to each input feature.


Saliency Methods
-----------------

The paper evaluates the following saliency methods:

- **Gradient:** the gradient of the class score with respect to the input, :math:`E_{grad}(\mathbf{x}) = \frac{\partial S_c}{\partial \mathbf{x}}`.
- **Gradient** :math:`\odot` **Input:** the element-wise product of the gradient and the input, :math:`E(\mathbf{x}) = \mathbf{x} \odot \frac{\partial S_c}{\partial \mathbf{x}}`.
- **Integrated Gradients:** the gradients are accumulated along the path from a baseline :math:`\bar{\mathbf{x}}` to the input:

  .. math::

     E_{IG}(\mathbf{x})_i = (x_i - \bar{x}_i) \int_{0}^{1} \frac{\partial S_c(\bar{\mathbf{x}} + \alpha (\mathbf{x} - \bar{\mathbf{x}}))}{\partial x_i} \, d\alpha

- **Guided Backpropagation:** a modified gradient where negative gradients are set to zero when backpropagating through ReLU activations.
- **Grad-CAM:** a class activation map computed from the feature maps of the last convolutional layer (see :doc:`../_model_specific_xai/Grad-CAM`).
- **Guided Grad-CAM:** the element-wise product of Grad-CAM and Guided Backpropagation.
- **SmoothGrad:** an extension that can be applied to any of the methods above, averaging the explanations of :math:`N` noisy copies of the input:

  .. math::

     \hat{E}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} E(\mathbf{x} + g_i), \quad g_i \sim \mathcal{N}(0, \sigma^2)


Model Parameter Randomization Test
-----------------------------------

The explanation maps of a trained model are compared with those of the same architecture with randomly initialized weights.
If a method depends on what the model has learned, the maps should differ substantially.
The authors randomize the weights in two ways:

- **Cascading randomization:** the weights are randomized successively from the top layer (the logits) down to the bottom layer,
  so that an increasing part of the network is destroyed.
- **Independent randomization:** the weights of one layer at a time are randomized, while all other layers keep their trained weights.

Gradient and Grad-CAM change considerably as the layers are randomized and therefore pass the test.
Guided Backpropagation and Guided Grad-CAM, instead, produce almost the same maps even after the top layers have been randomized,
showing that they are largely insensitive to the parameters of the higher layers.


Data Randomization Test
------------------------

The explanation maps of a model trained on the original data are compared with those of a model trained on a copy of the data with randomly permuted labels.
The second model can only reach a high training accuracy by memorizing the training examples,
since the relationship between inputs and labels has been destroyed.
If a method depends on the relationship between inputs and labels, the explanations of the two models should differ.

Gradient and Grad-CAM produce clearly different maps for the model trained on random labels,
while methods such as Integrated Gradients and Gradient :math:`\odot` Input still highlight the structure of the input image.


Similarity Metrics
--------------------

The original explanation :math:`E` and the randomized explanation :math:`\tilde{E}` are compared both visually and quantitatively.

**Visual Comparison**

The explanation maps are visualized in two ways:

- **Absolute value:** the absolute values of the normalized map are shown, so only the magnitude of the relevance is visible.
- **Diverging:** the map is shown as is, with different colors for positive and negative relevance.

Visual inspection alone can be misleading, since maps of models with very different parameters can look similar.
For this reason, the similarity is also measured with the following metrics.

**Spearman Rank Correlation**

The Spearman rank correlation compares the rankings of the features in the two maps:

.. math::

   \rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}

where :math:`n` is the number of features and :math:`d_i` is the difference between the ranks of feature :math:`i` in :math:`E` and :math:`\tilde{E}`.
:math:`\rho` ranges from -1 (opposite rankings) to 1 (identical rankings).
The correlation is computed both on the absolute values of the maps, which considers only the magnitude of the relevance,
and on the signed values, which also takes into account whether a feature contributes positively or negatively.

The sign makes a difference: for Gradient :math:`\odot` Input and Integrated Gradients, the maps of the randomized model look similar to the original ones
and the rank correlation with absolute values remains high, but the randomization changes the sign of the attributions
and the rank correlation without absolute values drops close to zero.
Looking only at the magnitude of the relevance can therefore hide the effect of the randomization.

**Structural Similarity Index (SSIM)**

The SSIM compares two images in terms of luminance, contrast and structure:

.. math::

   \text{SSIM}(E, \tilde{E}) = \frac{(2 \mu_E \mu_{\tilde{E}} + C_1)(2 \sigma_{E\tilde{E}} + C_2)}{(\mu_E^2 + \mu_{\tilde{E}}^2 + C_1)(\sigma_E^2 + \sigma_{\tilde{E}}^2 + C_2)}

where :math:`\mu_E` and :math:`\sigma_E^2` are the mean and variance of :math:`E`, :math:`\sigma_{E\tilde{E}}` is the covariance between the two maps,
and :math:`C_1`, :math:`C_2` are small constants that stabilize the division.
The SSIM is 1 for identical maps and decreases as the maps become less similar.
It is computed on the signed maps.

High values of these metrics after randomization indicate that the explanation does not depend on the model parameters or on the training labels.


References
----------

- **Sanity Checks for Saliency Maps:** Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. `Sanity Checks for Saliency Maps. <https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html>`_ NeurIPS. 2018
