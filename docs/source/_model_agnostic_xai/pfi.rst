Introduction to Permutation Feature Importance
===============================================

Permutation Feature Importance is a **model-agnostic** method, which means that it is not restricted to a certain model type, 
and it is a **global** method, which means that it only provides explanations for a full dataset, but not for individual samples.

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

Molnar, Christoph. `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ Lulu.com. 2022.
