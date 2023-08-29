Introduction to Local Interpretable Model-Agnostic Explanations (LIME)
========================================================================

Local Interpretable Model-Agnostic Explanations is a **model-agnostic** method, which means that it is not restricted to a certain model type, 
and it is a **local** method which means that it only provides explanations for individual samples. 

For a short video introduction to LIME, click below:

.. vimeo:: 745319036?h=a86f126018

    Short video lecture on the principles of LIME.

To summarize, LIME, an abbreviation for "local interpretable model agnostic explanations" is an approach that tries to deliver explanations for individual samples. 
It works by constructing an interpretable surrogate model (like a linear regression) to approximate predictions of a more complex model in the neighborhood of a given sample.

References
-----------

Molnar, Christoph. `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ Lulu.com. 2022.
