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

Molnar, Christoph. `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ Lulu.com. 2022.
