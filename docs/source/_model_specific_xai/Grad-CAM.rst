Introduction to Grad-CAM
=========================================

Grad-CAM (Gradient-weighted Class Activation Mapping) is a **model-specific** method, which provides local explanations for Deep Neural Networks.

For a short introduction to Grad-CAM, click below:

.. vimeo:: 745319946?h=fcd327fc80

    Short video lecture on the principles of Grad-CAM.


In summary, Grad-CAM is an explainability technique that visually highlights the regions in an image that are most important for a deep neural network's classification decision.
Grad-CAM works by computing the gradients of the model's output with respect to the feature maps in the final convolutional layer,
effectively revealing which parts of the image the model 'looks at' when making a prediction.

Mathematical Details
----------------------

Let us assume :math:`y^c` is the score for class :math:`c` i.e., the output for class :math:`c` before the softmax.

**Step 1: Computing Gradient**

Compute the gradient of :math:`y^c` with respect to the feature map activation :math:`A^k` of a convolution layer i.e., :math:`\frac {\delta y^c}{\delta A^k}`

**Step 2: Calculate Global Average Pooling (GAP) of the feature map.**
     
Global average pool the gradients over the width dimension (indexed by $i$) and the height dimension (indexed by $j$) to obtain weights ${\alpha_k^c}$

.. math::
    {\alpha_k^c} = \frac {1}{Z} \sum_{i} \sum_{j} \frac {\delta y^c}{\delta A^k_{ij}}

**Step 3: Calculate Final Grad-CAM Localization Map**
     
Perform a weighted combination of the feature map activations :math:`A^k` where the weights are the :math:`{\alpha_k^c}` we just calculated and keep only positive contributions applying a ReLU function.

.. math::
    L^c_{Grad-CAM} = ReLU (\sum_k {\alpha_k^c} A^k)


References
----------

- **Grad-CAM:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. `Grad-cam: Visual explanations from deep networks via gradient-based localization. <https://doi.org/10.1109/ICCV.2017.74>`_ ICCV. 2017
- **Tutorial on Grad-CAM:** `Grad-CAM for ResNet152 network <https://medium.com/@stepanulyanin/grad-cam-for-resnet152-network-784a1d65f3>`_
- **XAI Book:** Molnar, C. `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ Lulu.com. 2022.

