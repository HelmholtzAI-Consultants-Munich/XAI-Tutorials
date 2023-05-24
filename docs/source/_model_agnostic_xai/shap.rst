Introduction to SHapley Additive exPlanations (SHAP)
=====================================================

SHapley Additive exPlanationsis a **model-agnostic** method, which means that it is not restricted to a certain model type, 
and it is a **local** method which means that it only provides explanations for individual samples. 
However, the individual explanations can be used to also get **global** interpretations. 

For a short video introduction to SHAP, click below:

.. vimeo:: 745352008?h=3168320cef

    Short video lecture on the principles of SHAP.

To summarize, SHAP is a method that enables a fast computation of Shapley values and can be used to explain the prediction of an instance x 
by computing the contribution (Shapley value) of each feature to the prediction. We get contrastive explanations that compare the prediction with the average prediction. 
The fast computation makes it possible to compute the many Shapley values needed for the global model interpretations. 
With SHAP, global interpretations are consistent with the local explanations, since the Shapley values are the “atomic unit” of the global interpretations. 
If you use LIME for local explanations and permutation feature importance for global explanations, you lack a common foundation. 
SHAP provides KernelSHAP, an alternative, kernel-based estimation approach for Shapley values inspired by local surrogate models, as well as TreeSHAP, an efficient estimation approach for tree-based models. 

References
-----------

 To learn more about Shapley values, the SHAP package, and how these are used to help us interpret our machine learning models, please refer to these resources:

- `A Unified Approach to Interpreting Model Predictions <https://arxiv.org/abs/1705.07874>`_ -- Scott Lundberg, Su-In Lee
- `Consistent feature attribution for tree ensembles <https://arxiv.org/abs/1706.06060>`_ -- Scott M. Lundberg, Su-In Lee
- `Consistent Individualized Feature Attribution for Tree Ensembles <https://arxiv.org/abs/1802.03888>`_ -- Scott M. Lundberg, Gabriel G. Erion, Su-In Lee
- `A game theoretic approach to explain the output of any machine learning model. <https://github.com/slundberg/shap>`_
- `Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable.  5.9 Shapley Values <https://christophm.github.io/interpretable-ml-book/shapley.html>`_ -- Christoph Molnar, 2019-12-17
- `Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable.  5.10 SHAP (SHapley Additive exPlanations <https://christophm.github.io/interpretable-ml-book/shap.html>`_ -- Christoph Molnar, 2019-12-17
- `Interpretable Machine Learning with XGBoost <https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27?gi=187ef710fdda>`_ -- Scott Lundberg, Apr 17, 2018
- `Explain Your Model with the SHAP Values <https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d>`_ -- Dataman, Sep 14, 2019
