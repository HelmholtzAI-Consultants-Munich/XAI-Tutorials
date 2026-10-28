Attention is not Explanation
=========================================

Attention mechanisms are widely used in NLP models and their weights are often presented as an explanation of the model's prediction:
tokens that receive high attention weights are interpreted as the most important ones for the decision.

Jain and Wallace (2019) questioned this interpretation. If attention weights provide a faithful explanation of the model, two properties should hold:

1. Alternative attention distributions should lead to different predictions.
2. Attention weights should agree with other feature importance measures.

They tested both properties on text classification, question answering and natural language inference tasks.

**Notation**

The input :math:`\mathbf{x}` is a sequence of :math:`T` tokens, each one-hot encoded over a vocabulary :math:`V`.
The encoder maps the input to a sequence of hidden states :math:`\mathbf{h} = \text{Enc}(\mathbf{x}) = \{h_1, \dots, h_T\}`.
The attention weights are computed as

.. math::

   \hat{\alpha} = \text{softmax}(\phi(\mathbf{h}, \mathbf{Q}))

where :math:`\phi` assigns a score to each hidden state and :math:`\mathbf{Q}` is an optional query (e.g. the question in question answering).
The decoder combines the hidden states weighted by the attention, :math:`h_{\alpha} = \sum_{t=1}^{T} \alpha_t h_t`,
and returns the prediction :math:`\hat{y} = \text{Dec}(\mathbf{h}, \alpha)`.


Attention Permutation: the algorithm
-------------------------------------

If attention weights explain the prediction, assigning the high weights to different tokens should change the output.
The first experiment tests this by randomly permuting the attention weights and measuring how much the prediction changes.

**Step 1: Forward Pass**

Encode the input, compute the attention weights and obtain the model's prediction:

.. math::

   \mathbf{h} \leftarrow \text{Enc}(\mathbf{x}), \quad \hat{\alpha} \leftarrow \text{softmax}(\phi(\mathbf{h}, \mathbf{Q})), \quad \hat{y} \leftarrow \text{Dec}(\mathbf{h}, \hat{\alpha})

**Step 2: Permute the Attention Weights**

For :math:`p = 1, \dots, 100`, randomly shuffle the attention weights and compute the new prediction:

.. math::

   \alpha^p \leftarrow \text{Permute}(\hat{\alpha}), \quad \hat{y}^p \leftarrow \text{Dec}(\mathbf{h}, \alpha^p)

The hidden states :math:`\mathbf{h}` are not changed: the same attention values are only assigned to different tokens.

**Step 3: Measure the Change in the Output**

Compare each new prediction with the original one using the Total Variation Distance (TVD):

.. math::

   \Delta\hat{y}^p \leftarrow \text{TVD}(\hat{y}^p, \hat{y}), \qquad \text{TVD}(\hat{y}_1, \hat{y}_2) = \frac{1}{2} \sum_{i=1}^{|\mathcal{Y}|} |\hat{y}_{1i} - \hat{y}_{2i}|

where :math:`|\mathcal{Y}|` is the number of classes and :math:`\hat{y}_{1i}` is the predicted probability of class :math:`i`.
The TVD ranges from 0 (identical predictions) to 1 (completely different predictions).

**Step 4: Aggregate**

Summarize the effect of the permutations with the median output difference:

.. math::

   \Delta\hat{y}^{med} \leftarrow \text{Median}_p(\Delta\hat{y}^p)

A small :math:`\Delta\hat{y}^{med}` means that the prediction barely depends on which tokens receive the high attention weights.
For many datasets the authors observed only small changes in the output, even when the original attention was concentrated on a few tokens.


Adversarial Attention
-----------------------

Random permutations only explore a limited set of alternative attention distributions.
In the second experiment, the authors explicitly search for an adversarial attention distribution :math:`\alpha^{adv}`
that is as different as possible from the original attention while leaving the prediction (almost) unchanged:

.. math::

   \max_{\alpha^{adv}} \; \text{JSD}(\alpha^{adv}, \hat{\alpha}) \quad \text{subject to} \quad \text{TVD}(\hat{y}^{adv}, \hat{y}) < \epsilon

where :math:`\hat{y}^{adv} = \text{Dec}(\mathbf{h}, \alpha^{adv})` and :math:`\epsilon` is a small tolerance.

The dissimilarity between two attention distributions is measured with the Jensen-Shannon Divergence (JSD):

.. math::

   \text{JSD}(\alpha_1, \alpha_2) = \frac{1}{2} \text{KL}\left(\alpha_1 \,\Big\|\, \frac{\alpha_1 + \alpha_2}{2}\right) + \frac{1}{2} \text{KL}\left(\alpha_2 \,\Big\|\, \frac{\alpha_1 + \alpha_2}{2}\right)

with the Kullback-Leibler divergence :math:`\text{KL}(p \,\|\, q) = \sum_{t} p_t \log \frac{p_t}{q_t}`.

The JSD compares each distribution with their average :math:`\frac{\alpha_1 + \alpha_2}{2}`.
Unlike the KL divergence, it is symmetric and always finite: it is 0 when the two distributions are identical
and reaches its maximum value :math:`\log 2 \approx 0.69` when they do not overlap.

In many cases the authors found adversarial attention distributions with a JSD close to the maximum that produce the same prediction.
Very different attention maps can therefore be equally compatible with the model's output,
so a single attention map cannot be taken as *the* explanation of the prediction.


Concordance with Feature Importance Measures
---------------------------------------------

If attention weights indicate which tokens are important, their ranking should agree with established feature importance measures.
The authors compare attention with a gradient-based measure and with leave-one-out, using the Kendall :math:`\tau` correlation.

**Step 1: Forward Pass**

.. math::

   \mathbf{h} \leftarrow \text{Enc}(\mathbf{x}), \quad \hat{\alpha} \leftarrow \text{softmax}(\phi(\mathbf{h}, \mathbf{Q})), \quad \hat{y} \leftarrow \text{Dec}(\mathbf{h}, \hat{\alpha})

**Step 2: Compute the Gradient-based Feature Importance**

For each position :math:`t`, compute the gradient of the output with respect to the input token:

.. math::

   g_t \leftarrow \left| \sum_{w=1}^{|V|} \mathbb{1}[x_{tw} = 1] \frac{\partial \hat{y}}{\partial x_{tw}} \right|, \quad \forall t \in [1, T]

The indicator :math:`\mathbb{1}[x_{tw} = 1]` selects the gradient with respect to the word :math:`w` that actually appears at position :math:`t`.
The gradient does not flow through the attention layer, so :math:`g_t` measures the importance of the token independently of the attention weights.

Similarly, the leave-one-out importance of a token is the change in the output when the token is removed from the input:
:math:`\Delta\hat{y}_t = \text{TVD}(\hat{y}(\mathbf{x}_{-t}), \hat{y}(\mathbf{x}))`.

**Step 3: Compute the Kendall** :math:`\tau` **Correlation**

.. math::

   \tau_g \leftarrow \text{Kendall-}\tau(\hat{\alpha}, g)

The Kendall :math:`\tau` compares two rankings by looking at all pairs of tokens :math:`(i, j)`.
A pair is concordant if both rankings order the two tokens in the same way and discordant otherwise:

.. math::

   \tau = \frac{n_c - n_d}{T(T-1)/2}

where :math:`n_c` and :math:`n_d` are the numbers of concordant and discordant pairs.
:math:`\tau` ranges from -1 (opposite rankings) to 1 (identical rankings), and values close to 0 indicate no agreement.


References
----------

- **Attention is not Explanation:** Jain, S., & Wallace, B. C. `Attention is not Explanation. <https://doi.org/10.18653/v1/N19-1357>`_ NAACL. 2019
