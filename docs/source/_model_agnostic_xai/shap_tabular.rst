SHAP for Tabular Data
======================

We now apply the general SHAP framework to **tabular data**. The Shapley
formula itself does not change. Instead, we need to define what the players
are, what it means for a feature to be "missing", and how the corresponding
coalition values :math:`v_x(S)` are evaluated.

What Are the Players?
---------------------

For tabular data, each **input feature (column)** is a player in the Shapley
game. A coalition :math:`S` is a subset of these input features.

Suppose, for example, that our model uses the features

.. code-block:: text

   Age    BMI    Income    Smoke

Then the set of players is

.. math::

   N = \{f_1, f_2, \ldots, f_M\},

where each :math:`f_i` represents one input feature.

Consider an individual instance :math:`x`:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Age
     - BMI
     - Income
     - Smoke
   * - 52
     - 27.4
     - 60k
     - Yes

For this instance, SHAP assigns a contribution :math:`\phi_i(x)` to each
input feature.

As introduced above, the SHAP value of feature :math:`i` is obtained by
combining its marginal contributions across different feature coalitions:

.. math::

   \phi_i(x) =
   \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!(|N|-|S|-1)!}{|N|!}
   \left[
      v_x(S \cup \{i\}) - v_x(S)
   \right].

For a particular coalition :math:`S`, the marginal contribution is

.. math::

   \Delta_{i,x}(S)
   =
   v_x(S \cup \{i\}) - v_x(S).

We therefore determine a feature's marginal contribution by comparing the
coalition value **with and without that feature**.

This raises an important practical question: how can we evaluate
:math:`v_x(S)` when the model still requires values for all input features?


What Does a "Missing" Feature Mean?
------------------------------------

Suppose we want to evaluate the coalition

.. math::

   S = \{\text{Age}, \text{BMI}\}

for the instance above.

The values of ``Age`` and ``BMI`` are available from the instance being
explained, while ``Income`` and ``Smoke`` are outside the coalition:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Age
     - BMI
     - Income
     - Smoke
   * - 52
     - 27.4
     - ?
     - ?

The features outside the coalition cannot simply be removed because the
model still expects a complete input. Their unknown values therefore need to
be represented in some way.

In the **background-based formulation** considered here, missing feature
values are replaced using values from samples drawn from a background
distribution.

For example, suppose the background dataset contains:

.. list-table::
   :header-rows: 1
   :widths: 10 22 22 23 23

   * -
     - Age
     - BMI
     - Income
     - Smoke
   * - :math:`x^{(1)}`
     - 35
     - 24.1
     - 45k
     - No
   * - :math:`x^{(2)}`
     - 67
     - 29.3
     - 80k
     - Yes
   * - :math:`x^{(3)}`
     - 43
     - 26.2
     - 55k
     - No

For coalition :math:`S=\{\text{Age},\text{BMI}\}`, the values of the features
inside the coalition remain fixed at the values of the instance being
explained:

.. code-block:: text

   Age = 52
   BMI = 27.4

The missing values for ``Income`` and ``Smoke`` can then be taken from the
background samples, producing complete model inputs such as

.. code-block:: text

   (52, 27.4, 45k, No)
   (52, 27.4, 80k, Yes)
   (52, 27.4, 55k, No)

The model is evaluated for each of these completed inputs and the resulting
predictions are averaged.

More generally, for :math:`K` background samples,

.. math::

   v_x(S)
   \approx
   \frac{1}{K}
   \sum_{k=1}^{K}
   f\left(
      x_S,
      x_{\bar{S}}^{(k)}
   \right),

where

* :math:`x_S` contains the feature values of the instance :math:`x` for the
  features inside coalition :math:`S`,
* :math:`\bar{S}` denotes the features outside the coalition,
* :math:`x_{\bar{S}}^{(k)}` contains the corresponding values from background
  sample :math:`k`.

Thus, **missing features are represented using values from a background
distribution, and the resulting model predictions are averaged to estimate
the coalition value** :math:`v_x(S)`.

The background data therefore plays an important role: it defines the
reference distribution used to represent feature information that is not
available in a coalition.


What Defines the Baseline?
-----------------------------

The general additive decomposition introduced above is

.. math::

   f(x)
   =
   v_x(\emptyset)
   +
   \sum_{i \in N}\phi_i.

The baseline is therefore the value of the **empty coalition**
:math:`v_x(\emptyset)`.

What does the empty coalition mean for the background-based tabular
formulation?

For

.. math::

   S = \emptyset,

none of the feature values from the individual instance :math:`x` are
available. The model is therefore evaluated using the background samples
alone.

The empty-coalition value is

.. math::

   v_x(\emptyset)
   \approx
   \frac{1}{K}
   \sum_{k=1}^{K}
   f\left(x^{(k)}\right).

Thus, for this formulation, the baseline is the **average model prediction
over the background data**.

The SHAP decomposition becomes

.. math::

   f(x)
   =
   \underbrace{
      \frac{1}{K}\sum_{k=1}^{K}f(x^{(k)})
   }_{\text{baseline}}
   +
   \sum_{i \in N}\phi_i.

The choice of background distribution therefore matters. A different
background dataset can change the empty-coalition value and hence the
baseline against which the individual prediction is explained. It can also
change the coalition values :math:`v_x(S)` and therefore the resulting SHAP
values.


The Challenge of Correlated Features
--------------------------------------

Using a background distribution to represent missing features introduces an
important challenge when input features are **dependent or correlated**.

Consider, for example, the features ``Age`` and ``Experience``:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Age
     - Experience
     - ...
   * - 25
     - 3
     - ...

Suppose ``Age`` is part of the coalition but ``Experience`` is not. The age
of 25 is therefore retained from the instance being explained, while
``Experience`` is replaced using values from the background data.

This might produce inputs such as

.. code-block:: text

   Age    Experience
   25     2
   25     35
   25     8

Some combinations may be unlikely or even impossible in the original data.
For example, an age of 25 combined with 35 years of work experience would
usually be unrealistic.

This illustrates two related challenges.


Feature dependencies can be broken
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Background replacement can combine the observed value of one feature with
background values of another feature without preserving the relationship
between them.

Consequently,

* background replacement can create **unrealistic feature combinations**, and
* the model may be evaluated outside the typical data distribution.

Since SHAP values are computed from differences between coalition values,
changes in these coalition values can directly affect the resulting
attributions.


Predictive information can be shared
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Correlated features may also contain **overlapping predictive information**.

If two features carry similar information, their contribution to a prediction
may be distributed across them. An individual feature may therefore receive a
smaller SHAP value even when the correlated feature group as a whole is
important to the model.

SHAP values should consequently not be interpreted as independent measures of
the intrinsic importance of individual features when substantial feature
dependence is present.


How Dependence Is Handled Matters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no single universal way to define missing feature information when
features are dependent. Different SHAP explainers and maskers can make
different assumptions about how missing and dependent features should be
handled.

The interpretation of a SHAP explanation therefore depends not only on the
model and the instance being explained, but also on the **definition of
missingness and the reference distribution used to evaluate feature
coalitions**.

