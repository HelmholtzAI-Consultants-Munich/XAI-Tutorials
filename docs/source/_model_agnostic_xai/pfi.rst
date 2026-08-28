Introduction to Permutation Feature Importance
===============================================

What does Permutation Feature Importance explain?
-------------------------------------------------

Permutation Feature Importance (**PFI**) is a **global, model-agnostic explanation method**. It measures how much a trained model relies on each feature for its predictive performance.

PFI asks a simple question:

**How much does the model's predictive performance decrease when the information in one feature is disrupted?**

To answer this question, PFI randomly shuffles the values of one feature across the observations. The same feature values remain in the dataset, but they are assigned to different observations. This preserves the feature's overall distribution while breaking its associations with the target and the other features. If the model performs considerably worse after the feature has been shuffled, the model relied on that feature. If its performance changes very little, the feature provided little additional information to the model in the evaluated dataset. PFI is **model-agnostic** because it only requires predictions from an already trained model and a measure of predictive performance. The model is not retrained during the procedure. Standard PFI is a **global** method because it summarizes feature importance across a dataset rather than explaining an individual prediction.

.. important::

   PFI measures how much a particular model relies on a feature. It does not measure the feature's causal effect or its intrinsic importance in the real world.

The following video provides a short introduction to PFI and its main
concepts:

.. vimeo:: 745319412?h=1e5bd15ff7

    Short video lecture on the principles of Permutation Feature Importance.


How does PFI work?
------------------

PFI compares the model's prediction error on the original data with its error after one feature has been shuffled.

The procedure consists of the following steps:

1. Use the trained model to compute the prediction error on the original dataset. This is the **baseline error**.
2. Select one feature and randomly shuffle its values across the observations. The target and all other features remain unchanged.
3. Use the same trained model to make predictions for the permuted dataset. **Do not retrain the model.**
4. Compute the prediction error on the permuted dataset.
5. Subtract the original error from the permuted error. The increase in error is the permutation importance of the feature.
6. Repeat the permutation several times and average the resulting importance values to reduce random variability.
7. Repeat the procedure separately for every feature.

Why shuffle a feature?
^^^^^^^^^^^^^^^^^^^^^^

Suppose a dataset contains a feature called ``Feature B``. Before permutation, every value of ``Feature B`` belongs to a particular observation. After permutation, the column still contains exactly the same values, but the values have been reassigned to different observations. The shuffled feature therefore no longer provides its original information about the target. Its relationships with the other features are also broken. If the trained model depended on this information, its prediction error will increase when it is evaluated on the permuted data.

Shuffling instead of removing the feature has an important practical benefit: the input structure expected by the model does not change. The original model can therefore be evaluated again without training a second model.


Mathematical definition
^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`f` be an already trained model, :math:`X_{\mathrm{orig}}` the original feature matrix, :math:`y` the corresponding target values, and :math:`L` a prediction-error function for which larger values indicate worse predictions.

The original prediction error is

.. math::

   e_{\mathrm{orig}}
   =
   L\left(y, f\left(X_{\mathrm{orig}}\right)\right).

For feature :math:`j`, let :math:`X_{\mathrm{perm}(j,k)}` denote the dataset in which feature :math:`j` has been shuffled in repetition :math:`k`. The corresponding prediction error is

.. math::

   e_{\mathrm{perm}(j,k)}
   =
   L\left(y, f\left(X_{\mathrm{perm}(j,k)}\right)\right).

With :math:`K` permutation repetitions, the permutation feature importance is

.. math::

   \mathrm{PFI}_j
   =
   \frac{1}{K}
   \sum_{k=1}^{K}
   \left(
      e_{\mathrm{perm}(j,k)} - e_{\mathrm{orig}}
   \right).

For a single permutation, this simplifies to

.. math::

   \mathrm{PFI}_j
   =
   e_{\mathrm{perm}} - e_{\mathrm{orig}}.

This definition uses prediction error, where a larger value is worse. When a performance score is used instead, with larger values indicating better performance, the subtraction is reversed: permuted performance is subtracted from the original performance.


Interpreting PFI values
-----------------------

The size of the PFI value describes how strongly the model's predictive performance depends on a feature:

* A **large positive PFI value** means that the prediction error increased considerably after shuffling. The model relied strongly on the feature.
* A **PFI value close to zero** means that shuffling had little effect on the prediction error. The model obtained little additional predictive information from the feature in the evaluated dataset.
* A **negative PFI value** means that the model performed better after the feature was shuffled. This can occur because of random variation or because the model learned patterns involving the feature that do not generalize well.

A small PFI value does not necessarily mean that a feature contains no useful information. For example, another correlated feature may provide similar information and allow the model to compensate when the feature is shuffled.


Advantages and limitations
--------------------------

Advantages
^^^^^^^^^^

- **Straightforward interpretation**: The larger the increase in prediction error after shuffling, the more the model relied on the feature.
- **Model-agnostic**: PFI requires only model predictions and can therefore be applied to any trained predictive model.
- **Includes interaction effects**: The change in predictive performance can reflect how the model uses the feature both individually and through interactions with other features. PFI captures their combined influence on performance, but does not separate or describe the individual interaction effects.
- **Directly performance-based**: Feature importance is connected directly to changes in the chosen evaluation metric.


Limitations
^^^^^^^^^^^

- **Random variability**: Importance estimates can differ between shuffles. Several repetitions should therefore be averaged to obtain more stable estimates.
- **Sensitive to correlated features**: Correlated features can substitute for one another, hiding or distorting their individual importance. Shuffling can also create unrealistic combinations of feature values because it breaks dependencies between features.
- **Global explanation**: Standard PFI summarizes model reliance across a dataset. It does not explain individual predictions or show whether a feature increases or decreases a particular prediction.
- **Computational cost**: PFI requires repeated model evaluations for every feature and permutation. The computational cost therefore increases with the number of features, observations, and repetitions.


References
----------

* Breiman, L. (2001).
  `Random Forests
  <https://doi.org/10.1023/A:1010933404324>`_.
  *Machine Learning*, 45, 5--32.

* Fisher, A., Rudin, C., and Dominici, F. (2019).
  `All Models are Wrong, but Many are Useful: Learning a Variable's
  Importance by Studying an Entire Class of Prediction Models Simultaneously
  <https://www.jmlr.org/papers/v20/18-760.html>`_.
  *Journal of Machine Learning Research*, 20(177), 1--81.

* Molnar, C.
  `Interpretable Machine Learning: Permutation Feature Importance
  <https://christophm.github.io/interpretable-ml-book/feature-importance.html>`_.

