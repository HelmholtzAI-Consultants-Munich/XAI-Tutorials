.. XAI Tutorials documentation master file, created by
   sphinx-quickstart on Tue May 23 11:37:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorials for eXplainable Artificial Intelligence (XAI) methods
================================================================

Complex supervised Machine Learning (ML) and Deep Learning (DL) models are usually trained to achieve high accuracy, 
but are often considered to be "Black Boxes" because of a lack of understanding of the underlying reasons for decisions made by the model. 
To deploy such models to the real world, i.e. to move from research to application, it is indispensable to not only make accurate predictions, 
but also to understand the logic behind those predictions, to ensure that the models produce valuable results and are safe to use [1]. 
The application of interpretability methods allows us to identify potential biases, ensure fairness and prevent genre, social, and race discrimination. 
Moreover, in many fields, especially in the biomedical domain, explainability can be the key to acceptance and trust.

The XAI-Tutorials repository provides a collection of self-explanatory tutorials for different model-agnostic and model-specific XAI methods. 
Each tutorial comes in a Jupyter Notebook with practical exercises. 
In addition, we provide useful background information on those methods in this documentation.


.. toctree::
   :maxdepth: 1
   :caption: EXPLAINABLE AI

   _xai/terminology.rst
   _xai/importance.rst
   _xai/taxonomy_methods.rst


.. toctree::
   :maxdepth: 1
   :caption: MODEL-AGNOSTIC XAI METHODS

   _model_agnostic_xai/pfi.rst
   _model_agnostic_xai/lime.rst
   _model_agnostic_xai/shap.rst


.. toctree::
   :maxdepth: 1
   :caption: MODEL-SPECIFIC XAI METHODS

   _model_specific_xai/fgc.rst
   _model_specific_xai/Grad-CAM.rst


.. toctree::
   :maxdepth: 1
   :caption: MACHINE LEARNING BASICS

   _ml_basics/random_forest.rst
   _ml_basics/CNN_feature_visualization.rst



Contributions
==============
Comments and input are very welcome! Please, if you have a suggestion or you think something should be changed, open an issue or submit a pull request.


References
===========

[1] R. Roscher, B. Bohn, M. F. Duarte and J. Garcke, "Explainable Machine Learning for Scientific Insights and Discoveries," IEEE Access, Vol. 8, pp. 42200-42216 (2020)
