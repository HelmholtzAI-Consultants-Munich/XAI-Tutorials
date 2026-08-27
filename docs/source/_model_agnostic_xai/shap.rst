Introduction to SHapley Additive exPlanations (SHAP)
=====================================================

SHAP explains an individual model prediction by assigning a contribution to each input feature. These contributions explain how the model output moves from a baseline to the prediction for the instance being explained.

This chapter introduces the theoretical foundation of SHAP, explains how different SHAP algorithms compute these contributions, and discusses how the attribution problem changes for tabular, image, and text data.

.. toctree::
   :maxdepth: 2

   shap_theory
   shap_explainers
   shap_tabular
   shap_image
   shap_text


References
============

The following resources provide further information on Shapley values and
SHAP:

* Shapley, L. S. (1953).
  `A Value for n-Person Games
  <https://doi.org/10.1515/9781400881970-018>`_.
  In *Contributions to the Theory of Games II*, pp. 307--317.

* Lundberg, S. M. and Lee, S.-I. (2017).
  `A Unified Approach to Interpreting Model Predictions
  <https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_.
  *Advances in Neural Information Processing Systems 30*.

* Lundberg, S. M., Erion, G., Chen, H., et al. (2020).
  `From local explanations to global understanding with explainable AI for
  trees
  <https://doi.org/10.1038/s42256-019-0138-9>`_.
  *Nature Machine Intelligence*, 2, 56--67.

* Molnar, C.
  `Interpretable Machine Learning: SHAP
  <https://christophm.github.io/interpretable-ml-book/shap.html>`_.

Molnar, C.
  `Interpreting Machine Learning Models With SHAP
  <https://leanpub.com/shap>`_.
