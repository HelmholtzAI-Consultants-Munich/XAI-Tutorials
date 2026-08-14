Introduction to SHapley Additive exPlanations (SHAP)
====================================================

What does SHAP explain?
-----------------------

SHapley Additive exPlanations (**SHAP**) explains an **individual model
prediction** by assigning a contribution to each input feature. The main idea
is to quantify how much each feature contributes to the difference between a
**baseline prediction** and the prediction for the individual instance.

SHAP is therefore primarily a **local model-agnostic method**. Global
insights can later be obtained by aggregating SHAP values across many
instances.

Importantly, SHAP explains the **model output itself**, not changes in model
performance. This distinguishes SHAP from methods such as permutation feature
importance, which measure how predictive performance changes when feature
information is perturbed.

The quantity explained by SHAP depends on the prediction task:

* **Classification:** the explained model output may be a predicted
  probability, log-odds, or model score for a particular class.
* **Regression:** the explained model output is the predicted numerical value.

In both cases, the SHAP values describe how the input features move the model
output from a **baseline** to the prediction for the individual instance. What
this baseline represents will be introduced below and depends on how
``missing`` feature information is defined.

**What SHAP Does Not Imply**

SHAP explains the behavior of the **model**, not necessarily the underlying
data-generating process.

A large positive or negative SHAP value indicates how a feature contributes
to the model's prediction relative to the chosen baseline. It does not imply
that the feature has a causal effect on the predicted outcome.

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

Shapley values originate from **cooperative game theory** and provide a
principled way to determine how much each player contributed to a shared
outcome.

The most important concepts are:

* A **player** is a participant in the game.
* A **coalition** :math:`S` is any subset of players working together.
* The **coalition value** :math:`v(S)` describes what that group achieves.
* A player's **marginal contribution** describes how much the coalition value
  increases or decreases when that player joins.
* The **Shapley value** combines these marginal contributions to determine the
  player's overall contribution.

The important idea is that a player's contribution can depend on which other
players are already part of the coalition. Shapley values account for this by
considering the player's contribution across all possible situations.


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

The value of the full coalition is

.. math::

   v(\{A,B,C\}) = 140.

Our goal is to determine how much of this value should be attributed to each
individual player.


Marginal Contributions
^^^^^^^^^^^^^^^^^^^^^^

To measure a player's contribution to a coalition, we compare the coalition
value **before and after the player joins**.

For player :math:`i` joining coalition :math:`S`, the marginal contribution is

.. math::

   \Delta_i(S) = v(S \cup \{i\}) - v(S).

Here,

* :math:`i` is the player of interest,
* :math:`S` is the coalition before the player joins,
* :math:`v(S)` is the value of that coalition.

For example, suppose Bob is already working and Alice joins. Before Alice
joins, coalition :math:`\{B\}` has value 40. After Alice joins, coalition
:math:`\{A,B\}` has value 150. Alice's marginal contribution is therefore

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

Charlie's marginal contributions are:

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

This leads to an important question:

**Which of these marginal contributions represents the player's overall
contribution?**


Average Contribution Across Player Orderings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A player's contribution depends on which players are already in the
coalition. The Shapley value accounts for this by averaging the player's
marginal contribution over **all possible orders in which the players can
join**.

With three players, there are

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

For each order, we ask:

**What is the marginal contribution of Alice when she joins?**

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

In this example, the value of the empty coalition is zero. The three Shapley
values therefore sum to the value of the full coalition:

.. math::

   \phi_A + \phi_B + \phi_C
   =
   103.33 + 43.33 - 6.67
   =
   140,

which is exactly

.. math::

   v(\{A,B,C\}) = 140.

Notice that a **Shapley value can be negative**. Charlie has a negative
Shapley value because, on average, his presence reduces the value produced by
the team.


From Player Orderings to Coalitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of averaging marginal contributions over all possible player
orderings, the Shapley value can equivalently be computed as a **weighted
average over all possible coalitions**.

The coalitions represent the different subsets of players that could already
be present before the player of interest joins.

For example, several different player orderings can lead to the same
coalition being present before Alice joins. Rather than treating these
equivalent situations separately, the coalition formulation groups them
together and assigns the coalition an appropriate weight.

The general Shapley formula is

.. math::

   \phi_i =
   \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!(|N|-|S|-1)!}{|N|!}
   \left[
      v(S \cup \{i\}) - v(S)
   \right].

Here,

* :math:`N` is the set of all players,
* :math:`i` is the player of interest,
* :math:`S` is a coalition that does not contain player :math:`i`,
* :math:`|S|` is the number of players in coalition :math:`S`,
* :math:`v(S)` is the value achieved by coalition :math:`S`.

The expression

.. math::

   v(S \cup \{i\}) - v(S)

is the **marginal contribution**: the change in coalition value when player
:math:`i` joins coalition :math:`S`.

The expression

.. math::

   \frac{|S|!(|N|-|S|-1)!}{|N|!}

is the **coalition weight**. It accounts for how often coalition :math:`S`
is already present before player :math:`i` joins across all possible player
orderings.

The Shapley value can therefore be understood in two equivalent ways:

* as the **average marginal contribution across all possible player
  orderings**, or
* as the **weighted average of marginal contributions across all possible
  coalitions**.

Using coalitions avoids treating equivalent player orderings separately by
grouping situations where the same players are present before player
:math:`i` joins.


Shapley Axioms
^^^^^^^^^^^^^^

Why are Shapley values considered a principled way to assign contributions?
They are characterized by a set of properties, or **axioms**, describing how
an attribution should behave.

**Efficiency**

The contributions assigned to all players explain the difference between the
value of the full coalition and the value of the empty coalition:

.. math::

   \sum_{i \in N} \phi_i = v(N) - v(\emptyset).

In our example,

.. math::

   v(\emptyset)=0,

so

.. math::

   103.33 + 43.33 - 6.67 = 140.

Thus, the complete outcome relative to the empty coalition is distributed
among the players without leaving any value unexplained.

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

In the next section, we transfer this game-theoretic setting to machine
learning: players become input features, coalition values become model
outputs, and Shapley values become feature contributions.

From Shapley Values to SHAP
---------------------------

We now transfer the idea of Shapley values from cooperative game theory to
machine learning. Instead of asking how much each player contributed to a
shared outcome, we ask how much each input feature contributed to an
individual model prediction.

SHAP applies the same Shapley value formula introduced above. The formula
remains unchanged; only the interpretation of its components changes.


Mapping Game Theory to Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The concepts introduced for cooperative games have direct counterparts in
SHAP:

.. list-table:: From Shapley values to SHAP
   :header-rows: 1
   :widths: 25 35 40

   * - Symbol
     - Shapley value
     - SHAP
   * - :math:`i`
     - Player of interest
     - Feature of interest
   * - :math:`N`
     - Set of all players
     - Set of all input features
   * - :math:`S`
     - Coalition of players
     - Subset of input features
   * - :math:`v_x(S)`
     - Value achieved by coalition :math:`S`
     - Model output when the information in feature coalition :math:`S`
       is available for instance :math:`x`
   * - :math:`\phi_i`
     - Player's Shapley value
     - Feature's SHAP value

For an individual instance :math:`x`, the SHAP value of feature :math:`i`
can therefore be written as

.. math::

   \phi_i(x) =
   \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!(|N|-|S|-1)!}{|N|!}
   \left[
      v_x(S \cup \{i\}) - v_x(S)
   \right].

The term

.. math::

   v_x(S \cup \{i\}) - v_x(S)

is the feature's **marginal contribution**. It describes how much the model
output changes when the information from feature :math:`i` is added to
coalition :math:`S`.

SHAP therefore uses exactly the same principle as the cooperative-game
example: a feature's contribution is evaluated in different feature
coalitions and these marginal contributions are combined into its SHAP value.

There is, however, an important practical question. A machine-learning model
usually expects a complete input, even when only the information from a
subset of features is considered available. Features outside the coalition
must therefore somehow be represented as **"missing"**.

How this missing information is represented depends on the data modality and
the SHAP explainer. This choice determines how coalition values
:math:`v_x(S)` are evaluated and will be discussed separately for tabular,
image, and text data.


Additive Decomposition
^^^^^^^^^^^^^^^^^^^^^^

The Shapley efficiency property states that the Shapley values explain the
difference between the value of the full coalition and the value of the empty
coalition:

.. math::

   \sum_{i \in N} \phi_i
   =
   v(N) - v(\emptyset).

In SHAP, the **full coalition** corresponds to the complete instance being
explained. Its coalition value is therefore the model prediction

.. math::

   v_x(N) = f(x).

The **empty coalition** represents the situation in which none of the
feature information from the instance is available. Its value

.. math::

   v_x(\emptyset)

defines the **baseline** against which the prediction is explained.

The efficiency property therefore becomes

.. math::

   \sum_{i \in N} \phi_i
   =
   f(x) - v_x(\emptyset).

Rearranging gives the additive decomposition of an individual prediction:

.. math::

   f(x)
   =
   v_x(\emptyset)
   +
   \sum_{i \in N} \phi_i.

Here,

* :math:`f(x)` is the model prediction for the individual instance,
* :math:`v_x(\emptyset)` is the **baseline**, and
* :math:`\phi_i` is the contribution of feature :math:`i`.

The SHAP values therefore collectively explain the complete difference
between the baseline and the individual prediction.

A positive SHAP value indicates that a feature contributes toward increasing
the model output relative to the baseline, while a negative SHAP value
contributes toward decreasing it. The magnitude of the SHAP value indicates
the strength of the contribution.

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


Contrastive Explanations
^^^^^^^^^^^^^^^^^^^^^^^^

SHAP is therefore a **contrastive explanation method**. It does not explain
the prediction :math:`f(x)` in isolation. Instead, it explains how the input
features move the model output from the baseline to the prediction:

.. math::

   \underbrace{f(x)}_{\text{prediction}}
   =
   \underbrace{v_x(\emptyset)}_{\text{baseline}}
   +
   \underbrace{\sum_{i \in N}\phi_i}_{\text{feature contributions}}.

The baseline is the value of the **empty feature coalition**. Importantly,
there is no single universal way to construct this empty coalition for every
SHAP explanation.

What :math:`v_x(\emptyset)` represents depends on how "missing" feature
information is defined for the particular data modality and explainer.
Consequently, SHAP values must always be interpreted relative to the
corresponding baseline.

For example, later sections will show that an empty coalition can be
represented differently for tabular, image, and text data. These different
definitions of missingness determine both the coalition values and the
baseline used by the explanation.


Computational Challenge
-----------------------

The definition of Shapley values requires considering all possible feature
coalitions. For :math:`M` input features, there are

.. math::

   2^M

possible coalitions.

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

In practice, different SHAP algorithms reduce this computational cost through
**approximation** or by exploiting properties of the **model and data**.

SHAP should therefore not be understood as a single algorithm. It is a
framework for Shapley-based feature attribution that can be implemented using
different computational strategies. The practical notebooks introduce
examples of these different explainers.


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

SHAP decomposes an individual prediction into feature-specific contributions
relative to a defined baseline. Each SHAP value describes how a feature
contributes to the difference between the baseline and the individual
prediction.

**Local and global interpretation**

Although SHAP values explain individual predictions, they can be aggregated
across many instances to obtain global insights into model behavior. The same
underlying feature contributions therefore form the basis of both local and
global interpretations.


Limitations
^^^^^^^^^^^

**Missingness and feature dependence**

SHAP requires defining how features outside a coalition are represented as
"missing". This choice determines how coalition values are evaluated and can
therefore affect the resulting feature attributions.

The problem is particularly important for dependent features. Treating one
feature as missing while retaining a related feature can break dependencies,
produce unrealistic inputs, or distribute shared information across multiple
features.

**Computational complexity**

Exact Shapley computation grows exponentially with the number of features,
since up to :math:`2^M` feature coalitions must be considered. Practical SHAP
algorithms therefore rely on approximation or exploit properties of the model
and data to reduce this computational cost.

**Reference dependence**

SHAP values are contrastive: they explain a prediction relative to the
baseline :math:`v_x(\emptyset)`. How the empty coalition is represented
therefore matters. Changing the definition of missingness or the reference
used to construct the empty coalition can change both the baseline and the
resulting SHAP values.


SHAP for Tabular Data
---------------------

We now apply the general SHAP framework to **tabular data**. The Shapley
formula itself does not change. Instead, we need to define what the players
are, what it means for a feature to be "missing", and how the corresponding
coalition values :math:`v_x(S)` are evaluated.

What Are the Players?
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Correlated features may also contain **overlapping predictive information**.

If two features carry similar information, their contribution to a prediction
may be distributed across them. An individual feature may therefore receive a
smaller SHAP value even when the correlated feature group as a whole is
important to the model.

SHAP values should consequently not be interpreted as independent measures of
the intrinsic importance of individual features when substantial feature
dependence is present.


How Dependence Is Handled Matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no single universal way to define missing feature information when
features are dependent. Different SHAP explainers and maskers can make
different assumptions about how missing and dependent features should be
handled.

The interpretation of a SHAP explanation therefore depends not only on the
model and the instance being explained, but also on the **definition of
missingness and the reference distribution used to evaluate feature
coalitions**.

SHAP for Image Data
-------------------

We now apply the same SHAP framework to **image data**. Again, the Shapley
formula itself does not change. Instead, we need to define what the players
are, what it means for an image feature to be "missing", and how the
corresponding coalition values :math:`v_x(S)` are evaluated.

What Are the Players?
^^^^^^^^^^^^^^^^^^^^^

For image data, the input **pixels** are the players in the Shapley game. A
coalition :math:`S` is therefore a subset of the pixels of the image.

For an image with :math:`M` pixel features, the set of players can be written
as

.. math::

   N = \{p_1, p_2, \ldots, p_M\}.

For an individual image :math:`x`, SHAP assigns a contribution
:math:`\phi_i(x)` to each pixel :math:`p_i`.

As introduced above, the SHAP value of pixel :math:`p_i` is obtained by
combining its marginal contributions across different pixel coalitions:

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

We therefore determine a pixel's marginal contribution by comparing the
coalition value **with and without that pixel**.

This raises the same practical question as for tabular data: how can we
evaluate :math:`v_x(S)` when pixels outside the coalition cannot simply be
removed from the image?


What Does a "Missing" Pixel Mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image models generally expect a complete image as input. A pixel outside a
coalition therefore cannot simply be removed. Instead, an **image masker**
defines how the information contained in a "missing" pixel is replaced.

Suppose, for example, that :math:`S` is a coalition containing a subset of
the pixels of image :math:`x`.

Pixels inside the coalition retain their original values, while pixels
outside the coalition are treated as missing:

.. math::

   x_j^{(S)}
   =
   \begin{cases}
      x_j, & p_j \in S,\\
      m_j(x), & p_j \notin S,
   \end{cases}

where

* :math:`x_j` is the original value of pixel :math:`p_j`,
* :math:`m_j(x)` is the replacement value produced by the image masker, and
* :math:`x^{(S)}` is the resulting complete, masked image.

The model can then be evaluated on this masked image. The corresponding
coalition value is

.. math::

   v_x(S) = f\left(x^{(S)}\right).

Thus, unlike the background-based tabular formulation, a coalition does not
need to be completed using multiple background samples. The image masker
directly constructs a complete image corresponding to the coalition.


Image Masking Strategies
~~~~~~~~~~~~~~~~~~~~~~~~

Different masking strategies can be used to represent missing image
information. Two common approaches are **blurring** and **inpainting**.

**Blurring**

With a blur masker, a missing pixel is replaced with the corresponding value
from a blurred version of the image.

For example, a masker using a :math:`16 \times 16` blur constructs a blurred
image in which each replacement pixel is determined from its local
:math:`16 \times 16` neighborhood. Conceptually,

.. math::

   x_j^{(S)}
   =
   \begin{cases}
      x_j, & p_j \in S,\\
      x_j^{\mathrm{blur}}, & p_j \notin S.
   \end{cases}

Pixels in the coalition therefore retain the original image information,
while pixels outside the coalition contain only the corresponding blurred
information.

Importantly, the :math:`16 \times 16` window specifies how the blurred
replacement values are constructed. It does **not** divide the image into
:math:`16 \times 16` SHAP regions.

**Inpainting**

Inpainting instead reconstructs the missing image information from the
surrounding visible image. Rather than replacing a missing pixel with a value
from a precomputed blurred image, the masked area is filled based on the
image information around it.

Different inpainting algorithms can use different reconstruction strategies,
but their role within SHAP is the same: they define what the model receives
when particular pixel information is considered unavailable.

The choice of masker therefore defines what "missing" means in the
corresponding Shapley game.


What Defines the Baseline?
^^^^^^^^^^^^^^^^^^^^^^^^^^

The general SHAP decomposition is

.. math::

   f(x)
   =
   v_x(\emptyset)
   +
   \sum_{i \in N}\phi_i.

The baseline is therefore the value of the **empty coalition**
:math:`v_x(\emptyset)`.

What does the empty coalition mean for masking-based image SHAP?

For

.. math::

   S = \emptyset,

none of the original pixel values are available. Every pixel is therefore
replaced according to the chosen image masker:

.. math::

   x_j^{(\emptyset)} = m_j(x)
   \qquad \forall j.

The fully masked image is denoted by :math:`x^{(\emptyset)}`. Its model
output defines the baseline:

.. math::

   v_x(\emptyset)
   =
   f\left(x^{(\emptyset)}\right).

For example, when using a blur masker, :math:`x^{(\emptyset)}` is the fully
blurred version of the image. The baseline is therefore

.. math::

   v_x(\emptyset)
   =
   f\left(x^{\mathrm{blur}}\right).

The SHAP decomposition can then be written as

.. math::

   f(x)
   =
   \underbrace{
      f\left(x^{(\emptyset)}\right)
   }_{\text{baseline}}
   +
   \underbrace{
      \sum_{i \in N}\phi_i
   }_{\text{pixel contributions}}.

For masking-based image SHAP, the **chosen masker defines the empty coalition
and therefore the baseline**. Changing how missing pixels are represented can
change the model output for the empty coalition, the values of intermediate
coalitions, and consequently the resulting SHAP values.

SHAP explanations for images should therefore always be interpreted relative
to the masking strategy used to define missing image information.


SHAP for Text Data
------------------

We now apply the same SHAP framework to **text data**. As for tabular and
image data, the Shapley formula itself does not change. Instead, we need to
define what the players are, what it means for a text feature to be
"missing", and how the corresponding coalition values :math:`v_x(S)` are
evaluated.


What Are the Players?
^^^^^^^^^^^^^^^^^^^^^

For text data, the input **tokens** are the players in the Shapley game. A
coalition :math:`S` is therefore a subset of the tokens in the input text.

Suppose, for example, that a tokenizer represents a sentence as

.. code-block:: text

   [I] [really] [enjoyed] [this] [wonder] [ful] [movie] [.]

The set of players is then

.. math::

   N = \{t_1, t_2, \ldots, t_M\},

where each :math:`t_i` represents one input token.

Notice that tokens do not necessarily correspond to complete words. Depending
on the tokenizer, a word may be split into several tokens. In the example
above, ``wonderful`` is represented by the two tokens ``wonder`` and ``ful``.

For an individual text :math:`x`, SHAP assigns a contribution
:math:`\phi_i(x)` to each token :math:`t_i`.

As introduced above, the SHAP value of token :math:`t_i` is obtained by
combining its marginal contributions across different token coalitions:

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

We therefore determine a token's marginal contribution by comparing the
coalition value **with and without that token**.

This again raises a practical question: how can we evaluate :math:`v_x(S)`
when tokens outside the coalition cannot simply be removed from the text?


What Does a "Missing" Token Mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text models generally expect an input that follows the representation defined
by their tokenizer. Tokens outside a coalition therefore need to be
represented in a way that keeps the input compatible with the model.

A **text masker** defines how these "missing" tokens are represented.

Consider again the tokenized sentence

.. code-block:: text

   [I] [really] [enjoyed] [this] [wonder] [ful] [movie] [.]

and suppose we want to evaluate a coalition containing the tokens

.. math::

   S = \{\text{really}, \text{enjoyed}, \text{wonder}\}.

Conceptually, only the information from these tokens is retained:

.. code-block:: text

   [?] [really] [enjoyed] [?] [wonder] [?] [?] [?]

The text masker then replaces or hides the tokens outside the coalition
according to its masking strategy. For a model with an appropriate mask
token, the resulting input could for example be represented as

.. code-block:: text

   [MASK] [really] [enjoyed] [MASK] [wonder] [MASK] [MASK] [MASK]

More generally, the masked text can be denoted by :math:`x^{(S)}`. Its token
values are defined as

.. math::

   x_j^{(S)}
   =
   \begin{cases}
      x_j, & t_j \in S,\\
      m_j(x), & t_j \notin S,
   \end{cases}

where

* :math:`x_j` is the original token information for :math:`t_j`,
* :math:`m_j(x)` is the representation produced by the text masker for a
  missing token, and
* :math:`x^{(S)}` is the resulting masked text input.

The model can then be evaluated on this masked input. The corresponding
coalition value is

.. math::

   v_x(S)
   =
   f\left(x^{(S)}\right).

Thus, the **text masker defines what it means for token information to be
missing**. The exact masking strategy depends on the model and tokenizer. For
models that support a dedicated mask token, this may involve replacing
missing tokens with that token.


What Defines the Baseline?
^^^^^^^^^^^^^^^^^^^^^^^^^^

The general SHAP decomposition is

.. math::

   f(x)
   =
   v_x(\emptyset)
   +
   \sum_{i \in N}\phi_i.

The baseline is therefore the value of the **empty coalition**
:math:`v_x(\emptyset)`.

For text data, what does the empty coalition :math:`S=\emptyset` mean?

For the empty coalition, none of the original tokens are available. The text
masker therefore represents all input tokens as missing.

For example,

.. code-block:: text

   [I] [really] [enjoyed] [this] [wonder] [ful] [movie] [.]

could conceptually become

.. code-block:: text

   [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]

when a mask-token-based strategy is used.

More generally,

.. math::

   x_j^{(\emptyset)}
   =
   m_j(x)
   \qquad \forall j.

The resulting fully masked text is denoted by
:math:`x^{(\emptyset)}`. Its model output defines the baseline:

.. math::

   v_x(\emptyset)
   =
   f\left(x^{(\emptyset)}\right).

The SHAP decomposition can therefore be written as

.. math::

   f(x)
   =
   \underbrace{
      f\left(x^{(\emptyset)}\right)
   }_{\text{baseline}}
   +
   \underbrace{
      \sum_{i \in N}\phi_i
   }_{\text{token contributions}}.

For masking-based text SHAP, the **text masker defines the empty coalition and
therefore the baseline**. Changing how missing tokens are represented can
change the model output for the empty coalition, the values of intermediate
coalitions, and consequently the resulting SHAP values.

SHAP explanations for text should therefore always be interpreted relative
to the masking strategy used to define missing token information.


The Challenge of Dependent Tokens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text introduces an additional challenge because tokens are strongly
**context-dependent**. The meaning and contribution of a token often depend
on the tokens surrounding it.

Consider, for example,

.. code-block:: text

   [The] [movie] [was] [not] [good] [.]

The tokens ``not`` and ``good`` cannot easily be interpreted independently.
Together, the phrase ``not good`` expresses a meaning that neither token
captures on its own.

Masking one of these tokens changes the linguistic context substantially:

.. code-block:: text

   [The] [movie] [was] [MASK] [good] [.]

or

.. code-block:: text

   [The] [movie] [was] [not] [MASK] [.]


Tokens Are Context-Dependent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A token's meaning and contribution can depend strongly on the surrounding
tokens. Information may therefore be **shared across interacting tokens**.

For example, the contribution associated with ``not good`` arises partly
from the interaction between the two tokens. Assigning this shared
information to individual tokens is therefore not always straightforward.

As a result, SHAP values should not necessarily be interpreted as independent
measures of the intrinsic importance of individual tokens. They describe how
token information contributes within the coalition game defined by the
chosen masking strategy.


Masked Text Can Be Unnatural
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Masking tokens can also produce sequences that are unlikely, grammatically
incomplete, or different from the natural-language inputs on which the model
was trained.

The model may therefore be evaluated on inputs that differ from natural
language. This can affect the coalition values :math:`v_x(S)` and,
consequently, the resulting SHAP values.

As with the other data modalities, the definition of **missingness** is
therefore an important part of the interpretation of a text SHAP explanation.

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



