Introduction to SHapley Additive exPlanations (SHAP)
=====================================================

SHapley Additive exPlanationsis a **model-agnostic** method, which means that it is not restricted to a certain model type, 
and it is a **local** method which means that it only provides explanations for individual samples. 
However, the individual explanations can be used to also get **global** interpretations. SHAP was introduced in 2017 by `Lundberg et al. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_

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

- **Original SHAP paper:** Lundberg, S. M., & Lee, S. I. `A unified approach to interpreting model predictions. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_ NeurIPS. 2017
- **Intro to TreeExplainer:** Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. `From local explanations to global understanding with explainable AI for trees. <https://doi.org/10.1038/s42256-019-0138-9>`_ Nature machine intelligence. 2020.
- **Intro to TreeExplainer accelerated with GPUs:** Mitchell, R., Frank, E., & Holmes, G. `GPUTreeShap: massively parallel exact calculation of SHAP scores for tree ensembles. <https://doi.org/10.48550/arXiv.2010.13972>`_ arxiv. 2022
- **Intro to Integrated Gradients:** Sundararajan, M., Taly, A., & Yan, Q. `Axiomatic attribution for deep networks. <https://doi.org/10.48550/arXiv.1703.01365>`_ PMLR. 2017.
- **Visualizing the Impact of Feature Attribution Baselines:** `blog post <https://distill.pub/2020/attribution-baselines/>`_
- **XAI Book with focus on SHAP:** Molnar, C. `Interpreting Machine Learning Models With SHAP. <https://leanpub.com/shap>`_ 2022
- **XAI Book:** Molnar, C. `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ Lulu.com. 2022.
