Introduction to Grad-CAM
=========================================

Grad-CAM (Gradient-weighted Class Activation Mapping) is a **model-specific** method, which provides local explanations for Deep Neural Networks.

For a short introduction to Grad-CAM, click below:

.. vimeo:: 745319946?h=fcd327fc80

    Short video lecture on the principles of Grad-CAM.


In summary, Grad-CAM is an explainability technique that visually highlights the regions in an image that are most important for a deep neural network's classification decision.
Grad-CAM works by computing the gradients of the model's output with respect to the feature maps in the final convolutional layer,
effectively revealing which parts of the image the model 'looks at' when making a prediction.


References
----------

Selvaraju R. R., Cogswell M., Das  A., Vedantam  R. , Parik D. and Batra D. , `"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization" <https://ieeexplore.ieee.org/document/8237336/references#references>`_,  2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.
