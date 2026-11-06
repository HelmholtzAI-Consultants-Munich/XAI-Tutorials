SHAP for Text Data
=====================

We now apply the same SHAP framework to **text data**. As for tabular and
image data, the Shapley formula itself does not change. Instead, we need to
define what the players are, what it means for a text feature to be
"missing", and how the corresponding coalition values :math:`v_x(S)` are
evaluated.


What Are the Players?
----------------------

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
-----------------------------------

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
----------------------------

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
----------------------------------

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Masking tokens can also produce sequences that are unlikely, grammatically
incomplete, or different from the natural-language inputs on which the model
was trained.

The model may therefore be evaluated on inputs that differ from natural
language. This can affect the coalition values :math:`v_x(S)` and,
consequently, the resulting SHAP values.

As with the other data modalities, the definition of **missingness** is
therefore an important part of the interpretation of a text SHAP explanation.


