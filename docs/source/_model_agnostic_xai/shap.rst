Introduction to SHapley Additive exPlanations (SHAP)
====================================================

What does SHAP explain?
-----------------------

SHapley Additive exPlanations (**SHAP**) is a **local, model-agnostic XAI
method** that explains individual model predictions by assigning a
contribution to each input feature. SHAP is therefore primarily a 
**local explanation method**. However, global insights about the model can 
be obtained by aggregating SHAP values across many instances. Although SHAP 
is commonly introduced as model-agnostic, the SHAP framework also includes 
**model-specific algorithms** that exploit particular model structures to 
compute Shapley-based feature contributions more efficiently.

The main idea behind SHAP is to quantify how much each feature contributes to 
the difference between a **baseline prediction** and the prediction for a
particular instance.

Unlike methods such as permutation feature importance, SHAP does **not**
measure how model performance changes when a feature is perturbed. Instead,
SHAP explains the **model output itself**.

The quantity being explained depends on the prediction task:

* **Classification:** typically a predicted probability, log-odds, or model
  score for a class.
* **Regression:** the predicted numerical value.

Video Introduction
------------------

The following video provides a short introduction to SHAP and its main
concepts:

.. vimeo:: 745352008?h=3168320cef

    Short video introduction to SHAP.


Shapley Values
--------------

SHAP is built on **Shapley values** from cooperative game theory. To
understand how SHAP assigns contributions to individual features, we first
look at how Shapley values assign contributions to players in a cooperative
game.

Cooperative Game Theory
^^^^^^^^^^^^^^^^^^^^^^^

Shapley values originate from **cooperative game theory** and provide a fair
way to determine how much each player contributed to a shared outcome.

The most important concepts are:

* A **player** is a participant in the game.
* A **coalition** is any subset of players working together.
* The **coalition value** describes what a group of players achieves together.
* A player's **marginal contribution** describes how much the coalition value
  increases or decreases when that player joins.
* The **Shapley value** summarizes the player's marginal contributions to
  determine the player's overall contribution.

The important idea is that a player's contribution may depend on which other
players are already part of the coalition. Shapley values account for this by
considering the player's contribution in all possible situations.

Players, Coalitions, and Coalition Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To illustrate these concepts, consider a team building a table. The three
players are:

* **Alice (A):** a professional carpenter,
* **Bob (B):** an architect,
* **Charlie (C):** a cat.

Different combinations of these players produce different amounts of value.
The question is:

    **How should the total value be fairly attributed to Alice, Bob, and
    Charlie?**

A coalition is any subset of the available players. With three players,
there are :math:`2^3 = 8` possible coalitions. We assign each coalition a
value :math:`v(S)` describing what that group achieves:

.. list-table:: Coalition values for the table-building example
   :header-rows: 1
   :widths: 20 55 25

   * - Coalition :math:`S`
     - Description
     - :math:`v(S)` (€)
   * - :math:`\emptyset`
     - Nobody works
     - 0
   * - :math:`\{A\}`
     - Alice alone
     - 100
   * - :math:`\{B\}`
     - Bob alone
     - 40
   * - :math:`\{C\}`
     - Charlie alone
     - 0
   * - :math:`\{A,B\}`
     - Alice builds, Bob improves the design
     - 150
   * - :math:`\{A,C\}`
     - Charlie distracts Alice
     - 90
   * - :math:`\{B,C\}`
     - Bob and Charlie
     - 30
   * - :math:`\{A,B,C\}`
     - Charlie gets in the way
     - 140

For example,

.. math::

   v(\{A,B\}) = 150

means that Alice and Bob working together produce a value of €150.

The value of the complete coalition is

.. math::

   v(\{A,B,C\}) = 140.

Our goal is to determine how much of this value should be attributed to each
individual player.


Marginal Contributions
^^^^^^^^^^^^^^^^^^^^^^

A player's contribution cannot in general be determined by looking only at
what the player achieves alone. Their contribution may depend on which
players are already in the coalition.

To measure the contribution of player :math:`i` to a coalition :math:`S`,
we compare the coalition value **before and after the player joins**:

.. math::

   \Delta_i(S) = v(S \cup \{i\}) - v(S).

Here,

* :math:`i` is the player joining the coalition,
* :math:`S` is the coalition before the player joins,
* :math:`v(S)` is the value of that coalition.

For example, suppose Bob is already working and Alice joins. Before Alice
joins, the coalition :math:`\{B\}` has value 40. After Alice joins, the
coalition :math:`\{A,B\}` has value 150. Alice's marginal contribution is
therefore

.. math::

   \Delta_A(\{B\})
   = v(\{A,B\}) - v(\{B\})
   = 150 - 40
   = +110.

Alice's marginal contribution depends on the coalition she joins:

.. list-table:: Alice's marginal contributions
   :header-rows: 1
   :widths: 25 25 25 25

   * - Coalition :math:`S`
     - Alice joins
     - Calculation
     - Contribution
   * - :math:`\emptyset`
     - :math:`\{A\}`
     - :math:`100-0`
     - :math:`+100`
   * - :math:`\{B\}`
     - :math:`\{A,B\}`
     - :math:`150-40`
     - :math:`+110`
   * - :math:`\{C\}`
     - :math:`\{A,C\}`
     - :math:`90-0`
     - :math:`+90`
   * - :math:`\{B,C\}`
     - :math:`\{A,B,C\}`
     - :math:`140-30`
     - :math:`+110`

Marginal contributions can also be **negative**. For example, suppose Alice
is already working and Charlie joins:

.. math::

   \Delta_C(\{A\})
   = v(\{A,C\}) - v(\{A\})
   = 90 - 100
   = -10.

In this case, Charlie reduces the coalition value by €10.

His marginal contributions are:

.. list-table:: Charlie's marginal contributions
   :header-rows: 1
   :widths: 25 25 25 25

   * - Coalition :math:`S`
     - Charlie joins
     - Calculation
     - Contribution
   * - :math:`\emptyset`
     - :math:`\{C\}`
     - :math:`0-0`
     - :math:`0`
   * - :math:`\{A\}`
     - :math:`\{A,C\}`
     - :math:`90-100`
     - :math:`-10`
   * - :math:`\{B\}`
     - :math:`\{B,C\}`
     - :math:`30-40`
     - :math:`-10`
   * - :math:`\{A,B\}`
     - :math:`\{A,B,C\}`
     - :math:`140-150`
     - :math:`-10`

This leads to an important question: **Which of these marginal contributions
should represent a player's overall contribution?**


Computing the Shapley Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A player's marginal contribution depends on which players are already in the
coalition. The Shapley value accounts for this by considering **all possible
orders in which the players can join**.

For three players, there are

.. math::

   3! = 6

possible orders:

.. code-block:: text

   A -> B -> C
   A -> C -> B
   B -> A -> C
   B -> C -> A
   C -> A -> B
   C -> B -> A

For each order, we determine the coalition that exists immediately before
Alice joins and calculate her marginal contribution.

For example, consider

.. code-block:: text

   B -> A -> C

Bob is already present when Alice joins. The coalition before Alice joins is
therefore :math:`\{B\}`, and Alice contributes

.. math::

   v(\{A,B\}) - v(\{B\}) = 150 - 40 = 110.

Doing this for every possible order gives:

.. list-table:: Alice's marginal contribution across all possible orders
   :header-rows: 1
   :widths: 30 35 35

   * - Order
     - Coalition before Alice
     - Alice's contribution
   * - A -> B -> C
     - :math:`\emptyset`
     - :math:`+100`
   * - A -> C -> B
     - :math:`\emptyset`
     - :math:`+100`
   * - B -> A -> C
     - :math:`\{B\}`
     - :math:`+110`
   * - B -> C -> A
     - :math:`\{B,C\}`
     - :math:`+110`
   * - C -> A -> B
     - :math:`\{C\}`
     - :math:`+90`
   * - C -> B -> A
     - :math:`\{B,C\}`
     - :math:`+110`

The **Shapley value** is the average marginal contribution across all possible
player orders. For Alice,

.. math::

   \phi_A
   =
   \frac{100 + 100 + 110 + 110 + 90 + 110}{6}
   =
   103.33.

We can compute the Shapley values for Bob and Charlie in the same way:

.. math::

   \phi_B = 43.33,

.. math::

   \phi_C = -6.67.

The three Shapley values sum to the value of the complete coalition:

.. math::

   \phi_A + \phi_B + \phi_C
   =
   103.33 + 43.33 - 6.67
   =
   140.

This is exactly the value of the complete coalition,

.. math::

   v(\{A,B,C\}) = 140.

Notice that a **Shapley value can be negative**. Charlie has a negative
Shapley value because, on average, his presence reduces the value produced by
the team.


General Shapley Formula
^^^^^^^^^^^^^^^^^^^^^^^

The calculation above can be written compactly using the general Shapley
value formula:

.. math::

   \phi_i =
   \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!(M-|S|-1)!}{M!}
   \left[
      v(S \cup \{i\}) - v(S)
   \right].

Here,

* :math:`N` is the set of all players,
* :math:`M = |N|` is the total number of players,
* :math:`i` is the player whose Shapley value is being calculated,
* :math:`S` is a coalition that does not contain player :math:`i`,
* :math:`|S|` is the number of players in that coalition.

The expression

.. math::

   v(S \cup \{i\}) - v(S)

is the **marginal contribution** of player :math:`i` to coalition :math:`S`.

The expression

.. math::

   \frac{|S|!(M-|S|-1)!}{M!}

is the **coalition weight**. The weights ensure that all possible player
orderings are considered equally.

The Shapley value can therefore be understood as the **weighted average of a
player's marginal contributions across all possible coalitions**.


Shapley Axioms
^^^^^^^^^^^^^^

Why are Shapley values considered a fair way to assign contributions?
Shapley values are characterized by a set of properties, or **axioms**, that
describe how a reasonable attribution should behave.

**Efficiency**

The contributions assigned to all players add up to the value created by the
complete coalition (relative to the value of the empty coalition):

.. math::

   \sum_{i \in N} \phi_i = v(N) - v(\emptyset).

In our example, :math:`v(\emptyset)=0`, so

.. math::

   103.33 + 43.33 - 6.67 = 140.

Thus, the complete outcome is distributed among the players without leaving
any value unexplained.

**Symmetry**

If two players always contribute the same amount to every coalition, they
receive the same Shapley value.

In other words, players that play identical roles in the game are treated
equally.

**Dummy**

If a player never changes the value of any coalition, that player's Shapley
value is zero.

Formally, if

.. math::

   v(S \cup \{i\}) = v(S)

for every coalition :math:`S`, then

.. math::

   \phi_i = 0.

**Additivity**

If two cooperative games are combined, the Shapley value of each player in
the combined game is the sum of that player's Shapley values in the
individual games.

For two games with coalition-value functions :math:`v` and :math:`w`,

.. math::

   \phi_i(v+w) = \phi_i(v) + \phi_i(w).

Together, these properties make Shapley values a **principled way to assign
the outcome of a cooperative game to the individual players**.

In SHAP, this game-theoretic setting is transferred to machine learning:
**players become input features, the coalition value becomes the model
output, and each Shapley value becomes a feature contribution.**

From Shapley Values to SHAP
---------------------------

We now transfer the idea of Shapley values from cooperative game theory to
machine learning. Instead of asking how much each player contributed to a
shared outcome, we ask how much each input feature contributed to an
individual model prediction.

SHAP uses the same basic idea as before: a feature's contribution is evaluated
for different coalitions and these marginal contributions are combined into a
Shapley value. The resulting **SHAP value** describes the contribution of that
feature to the prediction.


Mapping Game Theory to Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The concepts introduced for cooperative games have direct counterparts in
SHAP:

.. list-table:: From Shapley values to SHAP
   :header-rows: 1
   :widths: 40 60

   * - Cooperative game theory
     - SHAP
   * - Player
     - Input feature
   * - Coalition
     - Subset of input features
   * - Coalition value
     - Model output for a feature coalition
   * - Marginal contribution
     - Change in model output when a feature is added to a coalition
   * - Shapley value
     - Feature contribution (SHAP value)

Consider an individual instance :math:`x` with :math:`M` input features. The
players of the SHAP game are the features

.. math::

   N = \{1, 2, \ldots, M\}.

For a particular feature :math:`i`, SHAP asks how the model output changes
when that feature is added to different subsets of the remaining features.

Just as in the cooperative game, the contribution of a feature can depend on
which other features are already present. SHAP therefore considers the
feature's marginal contribution across different feature coalitions.


Feature Coalitions
^^^^^^^^^^^^^^^^^^

A **feature coalition** :math:`S` is a subset of the input features whose
information is considered when evaluating the model.

Suppose, for example, that a model uses the features

.. code-block:: text

   Age   BMI   Income   Smoking

A possible feature coalition is

.. math::

   S = \{\text{Age}, \text{BMI}\}.

To determine the marginal contribution of ``Income``, we compare the model
output for the coalition without ``Income`` with the model output after
``Income`` is added:

.. math::

   \Delta_{\text{Income}}(S)
   =
   v(S \cup \{\text{Income}\}) - v(S).

This is exactly the same marginal-contribution calculation used in the
cooperative-game example. The difference is that :math:`v(S)` now represents
a **model output** rather than the value produced by a group of people.

There is, however, an important practical question: most machine-learning
models expect values for all input features. What does it therefore mean for
a feature to be *absent* from a coalition?

In SHAP, features outside the coalition are treated as **unknown**. Their
effect on the model output is accounted for using a **background or reference
distribution**. In practice, this is commonly represented by a background
dataset, such as the training data or a representative subset of it.

How the unknown features are handled can differ between SHAP explainers and
is particularly important when features are dependent or correlated.


Additive Decomposition of a Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After computing a Shapley value for every input feature, SHAP expresses an
individual prediction as an **additive decomposition**:

.. math::

   f(x)
   =
   \phi_0
   +
   \sum_{i=1}^{M} \phi_i,

where

* :math:`f(x)` is the model prediction for the instance :math:`x`,
* :math:`\phi_0` is the **baseline value**, and
* :math:`\phi_i` is the **SHAP value** of feature :math:`i`.

Equivalently,

.. math::

   f(x) - \phi_0
   =
   \sum_{i=1}^{M} \phi_i.

The SHAP values therefore explain the **difference between the baseline and
the individual prediction**.

A positive SHAP value means that the feature pushes the prediction **above**
the baseline, while a negative SHAP value means that the feature pushes the
prediction **below** the baseline.

For example, a prediction could be decomposed as

.. code-block:: text

   Baseline                    0.40
   Age                        +0.15
   BMI                        +0.08
   Income                     -0.04
   Smoking                    +0.11
                              -----
   Prediction                  0.70

The individual feature contributions sum to

.. math::

   0.15 + 0.08 - 0.04 + 0.11 = 0.30,

which is exactly the difference between the prediction and the baseline:

.. math::

   0.70 - 0.40 = 0.30.

This property is sometimes referred to as **local accuracy** or
**efficiency**: the feature contributions account for the complete difference
between the baseline and the prediction.


Baseline Value
^^^^^^^^^^^^^^

The **baseline value** :math:`\phi_0` represents the model output before any
information about the individual instance is taken into account. It is
typically defined as the expected model output over a background distribution:

.. math::

   \phi_0 = \mathbb{E}[f(X)].

In practice, this expectation is commonly approximated using a **background
dataset**, for example the training data or a representative subset of it.

The SHAP values then explain how the features of an individual instance move
the model output away from this baseline:

.. math::

   \underbrace{f(x)}_{\text{prediction}}
   =
   \underbrace{\mathbb{E}[f(X)]}_{\text{baseline}}
   +
   \underbrace{\sum_{i=1}^{M}\phi_i}_{\text{feature contributions}}.

The baseline is therefore an important part of the interpretation of a SHAP
explanation. A SHAP value should not be interpreted as saying that a feature
is inherently "positive" or "negative". Instead, it describes whether and by
how much the feature moves the prediction **relative to the chosen
baseline**.

Changing the background data can change the baseline and, consequently, the
resulting feature attributions.

Computational Challenge
-----------------------

The definition of Shapley values requires considering all possible coalitions.
For :math:`M` features, there are

.. math::

   2^M

possible feature coalitions.

The number of coalitions therefore grows **exponentially** with the number of
features:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Number of features
     - Number of possible coalitions
   * - 10
     - 1,024
   * - 20
     - 1,048,576
   * - 30
     - 1,073,741,824

For only 20 features, more than one million feature coalitions are already
possible. Exhaustively evaluating all coalitions therefore quickly becomes
infeasible for real-world machine-learning problems.

Practical SHAP algorithms address this computational challenge in different
ways. **KernelSHAP** provides a model-agnostic approach that can approximate
Shapley values by evaluating only a sample of feature coalitions. Other SHAP
algorithms exploit properties of particular model classes. For example,
**TreeSHAP** uses the structure of decision trees to compute SHAP values much
more efficiently than a general model-agnostic approach.

Thus, SHAP should not be understood as a single algorithm for computing
Shapley values. It is a framework that includes different algorithms for
computing or approximating Shapley-based feature contributions.

Advantages and Limitations
--------------------------

Advantages
^^^^^^^^^^

**Principled feature attribution**

SHAP is based on **Shapley values** from cooperative game theory. The resulting
feature contributions satisfy well-defined properties such as efficiency,
symmetry, dummy, and additivity. This provides a principled basis for
attributing a model prediction to its input features.

**Local explanations**

SHAP provides instance-level explanations by decomposing an individual
prediction into a baseline value and feature-specific contributions. Each
SHAP value describes how much a feature pushes the prediction above or below
the baseline.

**Local and global interpretation**

Although SHAP values explain individual predictions, they can be aggregated
across many instances to obtain global insights into model behavior. The same
feature contributions therefore form the basis of both local and global
interpretations.

**Applicable to different models and data modalities**

The general idea of SHAP is not restricted to a particular model class or
type of input data. Different SHAP explainers adapt the computation to
different models and data modalities, including tabular, image, and text
data.


Limitations
^^^^^^^^^^^

**Computational complexity**

Exact Shapley computation grows exponentially with the number of features,
since up to :math:`2^M` feature coalitions must be considered. Approximation
methods such as KernelSHAP and specialized algorithms such as TreeSHAP are
therefore used to make the computation practical.

**Feature dependence**

The treatment of features that are not part of a coalition affects the
resulting explanation. This becomes particularly important when features are
dependent or correlated: different ways of accounting for the missing
features can lead to different feature attributions.

**Dependence on the background data**

SHAP values explain a prediction relative to a baseline and a background
distribution. The choice of background data therefore influences both the
baseline and the resulting feature contributions. SHAP values should always
be interpreted relative to this reference.

**SHAP explains the model, not necessarily the underlying data-generating
process**

A large positive or negative SHAP value indicates how a feature contributes
to the model's prediction. It does not imply that the feature has a causal
effect on the predicted outcome.


References
----------

The following resources provide further information on Shapley values and
SHAP:

* Lundberg, S. M. and Lee, S.-I. (2017).
  `A Unified Approach to Interpreting Model Predictions
  <https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_.
  *Advances in Neural Information Processing Systems 30*.

* Shapley, L. S. (1953).
  `A Value for n-Person Games
  <https://doi.org/10.1515/9781400881970-018>`_.
  In *Contributions to the Theory of Games II*, pp. 307--317.

* Lundberg, S. M., Erion, G., Chen, H., et al. (2020).
  `From local explanations to global understanding with explainable AI for
  trees
  <https://doi.org/10.1038/s42256-019-0138-9>`_.
  *Nature Machine Intelligence*, 2, 56--67.

* Molnar, C.
  `Interpretable Machine Learning: SHAP
  <https://christophm.github.io/interpretable-ml-book/shap.html>`_.

* `SHAP Documentation <https://shap.readthedocs.io/>`_.



