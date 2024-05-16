Introduction to Permutation Feature Importance
===============================================

Permutation Feature Importance is a **model-agnostic** method, which means that it is not restricted to a certain model type, 
and it is a **global** method, which means that it only provides explanations for a full dataset, but not for individual samples.
Permutation Feature Importance was introduced in 2001 by `Breiman et al. <https://link.springer.com/article/10.1023/a:1010933404324>`_ for Random Forest models and extended to a model-agnostic version in 2018 by `Fischer et al. <https://www.jmlr.org/papers/v20/18-760.html>`_

For a short video introduction to Permutation Feature Importance, click below:

.. vimeo:: 745319412?h=1e5bd15ff7

    Short video lecture on the principles of Permutation Feature Importance.


To summarize, Permutation Feature Importance for a dataset of your choice works in the following way:

First, an already trained model predicts outputs and computes a performance measure for our dataset. This will serve as baseline performance.

Then, we carry out the following steps for each feature (potentially repeated multiple times):

1) We permute the features value across the dataset, essentially messing up any connection between the label and this feature.  
2) We predict outputs and compute the performance measure based on our dataset with the permuted feature.  
3) We compare the performance measure with the baseline performance.  
4) We use the difference between baseline performance and permuted performance as an indicator of the importance of the feature.  

If both performances are similar, the permuted feature did not lead to a decrease in model performance, indicating that the model did not rely heavily on the feature, hence assigning low importance. 
On the other hand, if the performance of the data with the permuted feature is much worse than the baseline performance, this shows that the model highly depended on the feature to produce good scores.

References
-----------

- Permutation Feature Importance for Random Forests: Breiman, L. `Random forests. <https://link.springer.com/article/10.1023/a:1010933404324>`_ Machine learning. 2001.
- Model Agnostic Permutation Feature Importance: Fisher, A., Rudin, C., & Dominici, F. `All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. <https://www.jmlr.org/papers/v20/18-760.html>`_ Journal of Machine Learning Research. 2018.
- XAI Book: Molnar, Christoph. `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ Lulu.com. 2022.
