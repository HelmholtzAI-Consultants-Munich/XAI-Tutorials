Introduction to Forest-Guided Clustering
=========================================

Forest-Guided Clustering is a **model-specific** method, which means that it is restricted to Random Forest models, 
and it is a **global** method, which means that it only provides explanations for a full dataset, but not for individual samples.

For a short introduction to Forest-Guided Clustering, click below:

.. vimeo:: 746443233?h=07ddf2290b

    Short video lecture on the principles of Forest-Guided Clustering.


In summary, Forest-Guided Clustering (FGC) is an explainability method for Random Forest models. Standard explainability methods (e.g. feature importance) assume independence of model features and hence, 
are not suited in the presence of correlated features. The Forest-Guided Clustering algorithm does not assume independence of model features, 
because it computes the feature importance based on subgroups of instances that follow similar decision rules within the Random Forest model. 
Hence, this method is well suited for cases with high correlation among model features.

For additional information you can have a look at the documentation for the FGC: https://forest-guided-clustering.readthedocs.io/en/latest/
