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
