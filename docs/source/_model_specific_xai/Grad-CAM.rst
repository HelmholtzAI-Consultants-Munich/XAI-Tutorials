Introduction to Grad-CAM
=========================================

Grad-CAM is a **model-specific** method, which provides local explanations for Deep Neural Networks

For a short introduction to Grad-CAM, click below:

.. vimeo:: 745319946?h=fcd327fc80

    Short video lecture on the principles of Grad-CAM.


In summary, Grad-CAM (Gradient-weighted Class Activation Mapping) is an explainability technique
that visually highlights the regions in an image that are most important for a deep neural network's classification decision.
Grad-CAM works by computing the gradients of the model's output with respect to the feature maps in the final convolutional layer,
effectively revealing which parts of the image the model 'looks at' when making a prediction.

For additional information you can have a look at the paper from  "Grad-CAM:
Visual Explanations from Deep Networks via Gradient-based Localization" by Selvaraju et al., 2016.
