Computing SHAP Values
=====================

SHAP is an attribution framework, not a single algorithm
--------------------------------------------------------

So far, we have learned **what SHAP values represent**: feature contributions that explain the difference between a baseline and an individual prediction.

However, SHAP itself is an **attribution framework, not a single algorithm**. Computing SHAP values requires evaluating the underlying Shapley attribution game, which can become computationally expensive because the number of possible feature coalitions grows exponentially with the number of features.

Different **SHAP explainers** use different strategies to make this computation feasible. Some explainers are **model-agnostic** and treat the model as a black box, while others are **model-specific** and exploit the internal structure of particular model types to compute SHAP values more efficiently. Model-agnostic explainers only require access to the model's inputs and predictions and can therefore be applied to essentially any model. 

Two important model-agnostic SHAP explainers are **KernelExplainer** and **PartitionExplainer**. Both can be applied independently of the model type but use different strategies to make the SHAP computation feasible. For tabular data, we focus on KernelExplainer as our model-agnostic reference. PartitionExplainer is particularly useful for structured inputs and will be introduced in more detail in the image and text tutorials.

*For further details, see the [SHAP Explainer API](https://shap.readthedocs.io/en/latest/api.html#explainers) and [Lundberg et al. (2020)](https://doi.org/10.1038/s42256-019-0138-9).*

KernelExplainer
-----------------

**KernelExplainer** is a model-agnostic method that approximates SHAP values by evaluating a sample of feature coalitions rather than all $2^M$ possible coalitions.

Each sampled coalition is represented by a **binary mask**, where 1 means that a feature is present in the coalition and 0 means that it is treated as missing. For example, for three features $A$, $B$, and $C$, the coalition $\{A,B\}$ is represented as $(1,1,0)$. For each sampled coalition, KernelExplainer evaluates the original model to obtain its **coalition value** $v_x(S)$. This creates a new dataset in which the binary coalition masks are the inputs and the corresponding coalition values are the targets. In addition, each coalition receives a **SHAP kernel weight** $\pi(S)$ that determines how strongly this coalition influences the subsequent regression. A simplified dataset could therefore look like this:

| $z_A$ | $z_B$ | $z_C$ | Target | Weight |
| ----: | ----: | ----: | ----: | ----: |
| 0 | 0 | 0 | $v_x(\emptyset)$ | special case |
| 1 | 0 | 0 | $v_x(\{A\})$ | $\pi(\{A\})$ |
| 0 | 1 | 0 | $v_x(\{B\})$ | $\pi(\{B\})$ |
| 1 | 1 | 0 | $v_x(\{A,B\})$ | $\pi(\{A,B\})$ |
| 0 | 1 | 1 | $v_x(\{B,C\})$ | $\pi(\{B,C\})$ |
| 1 | 1 | 1 | $v_x(\{A,B,C\})$ | special case |

KernelExplainer then fits an **additive linear model** to these coalition values:

$$
g(z)=\phi_0+\phi_A z_A+\phi_B z_B+\phi_C z_C
$$

Here, $z_i$ indicates whether feature $i$ is present in the coalition. The coefficient $\phi_i$ will become the estimated SHAP value of feature $i$. The important part is **how this linear model is fitted**. KernelExplainer does not treat every evaluated coalition equally. Instead, it minimizes a **weighted regression error**:

$$
\min_{\phi}\sum_S\pi(S)\left[v_x(S)-g(z_S)\right]^2
$$

The SHAP kernel weight $\pi(S)$ therefore controls **how strongly the error for coalition $S$ influences the fitted coefficients**. A coalition with a larger weight has a stronger influence on the regression solution than a coalition with a smaller weight. For a coalition $S$ with $|S|$ present features out of $M$ total features, the SHAP kernel weight is 

$$
\pi(S)=\frac{M-1}{\binom{M}{|S|}\,|S|\,(M-|S|)}
$$

The weighting gives greater importance to very small and very large coalitions, while the empty and full coalitions are handled specially so that the explanation is anchored at the baseline and the prediction: $g(0,\ldots,0)=v_x(\emptyset)$ and $g(1,\ldots,1)=f(x)$. The **SHAP kernel weights connect the linear regression back to the coalition weights in the original Shapley-value calculation**. With many features, the number of possible coalitions is very uneven across coalition sizes. There are only a few very small and very large coalitions, but many more medium-sized coalitions. For example, with 10 features, there are only 10 coalitions containing one feature, but 252 coalitions containing five features. If every coalition were treated equally in the regression, the medium-sized coalitions could dominate simply because there are many more of them. The SHAP kernel weights correct for this combinatorial imbalance: rarer, very small and very large coalitions receive more weight, while the numerous medium-sized coalitions receive less weight. In this way, the weighted regression reflects the coalition weighting required by the original Shapley-value calculation rather than being dominated by the most common coalition sizes. This allows the fitted coefficients $\phi_i$ to estimate the SHAP values.

KernelExplainer can therefore approximate SHAP values without explicitly evaluating every possible coalition. However, it still requires repeated evaluations of the original model and can become computationally expensive when the number of features is large.

<p align="center">
  <img src="https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/main/docs/source/_figures/shap_kernelshap_overview.png?raw=true" width="95%">
</p>

<p align="center">
  <em>Overview of KernelSHAP. KernelSHAP samples feature coalitions, evaluates their model outputs, and uses SHAP kernel weights in a weighted linear model to estimate the SHAP values.</em>
</p>

PartitionExplainer
-------------------

TreeExplainer
---------------

**TreeExplainer** is a model-specific method for tree-based models that computes SHAP values efficiently by exploiting the tree structure and using dynamic programming to account for many feature coalitions simultaneously, rather than evaluating each coalition separately.

To understand how TreeSHAP works, consider a very small decision tree:

```text
                         Age < 50?
                        /         \
                     Yes           No
                     /              \
               BMI < 25?         Income < 50k?
                /    \              /     \
              Yes    No           Yes      No
               |      |            |        |
              20     60           40       80
            Leaf 1  Leaf 2      Leaf 3   Leaf 4
```

Suppose we want to explain the prediction for the following sample:

| Age | BMI | Income | Smoke |
|---:|---:|---:|---|
| 42 | 28 | 40k | Yes |

`Smoke` is an input feature of the model but is not used anywhere in this tree. With all feature values available, the prediction is:

```text
Age = 42 → Age < 50? → Yes
   ↓
BMI = 28 → BMI < 25? → No
   ↓
Leaf 2 = 60
```

TreeSHAP now has to answer the same question as the Shapley formulation introduced earlier: **How much did each feature contribute to moving the prediction from the baseline to 60?**

***Baseline and Leaf Contributions***

As before, the **baseline** is the model output for the empty coalition: $v_x(\emptyset)$. It represents the model output when none of the feature information from the sample being explained is available. The **sample being explained and the background data have different roles**. The sample provides the feature values whose prediction we want to explain, while the background data provides the reference used when feature information is considered missing.

Assume that our background data contains two different samples:

| Background sample | Age | BMI | Income | Smoke | Tree prediction |
|---|---:|---:|---:|---|---:|
| $b_1$ | 35 | 22 | 70k | No | 20 |
| $b_2$ | 65 | 30 | 60k | Yes | 80 |

For the empty coalition, none of the feature values from the sample being explained are retained. The model is therefore evaluated using the background samples:

```text
b₁ → Leaf 1 → 20
b₂ → Leaf 4 → 80
```

The overall baseline is: $v_x(\emptyset) = \frac{20+80}{2} = 50$. Our sample has a prediction of 60, so the SHAP values ultimately have to explain: $60-50=10$.

To understand how TreeSHAP obtains these contributions, it is useful to look at the individual **leaves of the tree**. Under the empty coalition, one of the two background samples reaches Leaf 1 and one reaches Leaf 4. Since each background sample represents $\frac{1}{2}$ of our background data, the probability of reaching either Leaf 1 or Leaf 4 under the empty coalition is $\frac{1}{2}$. A leaf's contribution is its **leaf value multiplied by the probability of reaching that leaf**. Therefore,

$$
50
=
\underbrace{\frac{1}{2}\cdot20}_{\text{Leaf 1}}
+
\underbrace{\frac{1}{2}\cdot80}_{\text{Leaf 4}}
=
10+40.
$$

Leaf 2 and Leaf 3 contribute 0 because neither background sample reaches these leaves. Thus, the baseline can also be decomposed into contributions from the individual leaves:

$$
\underbrace{50}_{v_x(\emptyset)}
=
\underbrace{10}_{L_1}
+
\underbrace{0}_{L_2}
+
\underbrace{0}_{L_3}
+
\underbrace{40}_{L_4}.
$$

Importantly, these leaf contributions are **not separate baselines**. The overall baseline is still 50. They simply describe how much each leaf contributes to the baseline prediction. When feature information from the sample becomes available, the probabilities of reaching the different leaves can change. TreeSHAP attributes the resulting changes in the model output to the corresponding features.

**Example: Contribution of Leaf 2**

Let's look at how the contribution of **Leaf 2** changes as information about our sample becomes available. Under the empty coalition, neither background sample reaches Leaf 2, so its contribution is: $0\cdot60=0$. When both Age and BMI are available, their values are fixed to those of the sample being explained:

```text
Age = 42 → left
BMI = 28 → right
```

The sample therefore reaches Leaf 2 with certainty and its contribution becomes: $1\cdot60=60$. Thus, the contribution associated with Leaf 2 changes by $60-0=60$. How much of this change should be attributed to **Age**, and how much to **BMI**? We use the same Shapley marginal-contribution principle introduced earlier. First, consider the contribution of Leaf 2 for the different combinations of available path features:

| Available features | Probability of reaching Leaf 2 | Leaf 2 contribution |
|---|---:|---:|
| neither Age nor BMI | $0$ | $0$ |
| Age only | $\frac{1}{2}$ | $30$ |
| BMI only | $\frac{1}{2}$ | $30$ |
| Age and BMI | $1$ | $60$ |

If neither feature is available, the two background samples are used unchanged and neither reaches Leaf 2. If only **Age** is available, Age is fixed to 42 for both background samples. Both therefore enter the left subtree. BMI remains missing and takes its background values:

```text
Age = 42, BMI = 22 → Leaf 1
Age = 42, BMI = 30 → Leaf 2
```

One of the two completed inputs therefore reaches Leaf 2: $\frac{1}{2}\cdot60=30$. If only **BMI** is available, BMI is fixed to 28 while Age takes its background values:

```text
Age = 35, BMI = 28 → Leaf 2
Age = 65, BMI = 28 → right subtree
```

Again, one of the two inputs reaches Leaf 2: $\frac{1}{2}\cdot60=30$. 


One of the two completed inputs therefore reaches Leaf 2, so **Leaf 2's contribution** is $\frac{1}{2}\cdot60=30$. If only **BMI** is available, BMI is fixed to 28 while Age and the other missing features take their values from the background samples:

```text
Age = 35, BMI = 28, Income = 70k → Leaf 2 → 60
Age = 65, BMI = 28, Income = 60k → Leaf 4 → 80
```

Again, one of the two completed inputs reaches Leaf 2, so **Leaf 2's contribution** is $\frac{1}{2}\cdot60=30$. The other input reaches Leaf 4 and contributes $\frac{1}{2}\cdot80=40$. Thus, the complete coalition value for $S=\{\text{BMI}\}$ is $v_x(\{\text{BMI}\})=30+40=70$. Here, however, we focus only on the **30 contributed by Leaf 2**, because we are illustrating how the feature contributions associated with this particular leaf are calculated. The contributions from the other leaves are handled in the same way and combined later. Finally, when both Age and BMI are available, the sample reaches Leaf 2 with certainty: $1\cdot60=60$.

***From Leaf Contributions to SHAP Values***

We can now calculate the contribution of **Age to Leaf 2**. There are two possible coalition contexts for Age because BMI is the only other feature on this path. 

* If BMI is absent, we compare: $\emptyset\rightarrow\{\text{Age}\}$, giving the marginal contribution $30-0=30$.
* If BMI is already present, we compare $\{\text{BMI}\}\rightarrow\{\text{Age},\text{BMI}\}$, giving $60-30=30$.

Age's contribution associated with Leaf 2 is therefore $\phi_{\text{Age}}^{(\text{Leaf 2})} = \frac{1}{2}(30) + \frac{1}{2}(30) = 30$.

Why do both contexts receive a Shapley weight of $\frac{1}{2}$? For the two path features Age and BMI, there are only two possible orders:

```text
Age → BMI
BMI → Age
```

Age is first in one ordering and second in the other. Therefore, the context in which BMI is absent and the context in which BMI is already present each occur in **one of the two possible orderings** and receive weight $\frac{1}{2}$. 

The same calculation for BMI gives $\phi_{\text{BMI}}^{(\text{Leaf 2})} = 30$. Thus, the change in the contribution of Leaf 2 is fully attributed to Age and BMI: $60 = 0 + 30_{\text{Age}} + 30_{\text{BMI}}$. These are **not yet the final SHAP values** of Age and BMI. TreeSHAP also accounts for the other leaves that can become relevant when features are considered present or missing. The leaf-wise contributions are then summed: $\phi_i(x) = \sum_{\text{leaves }L} \phi_i^{(L)}(x)$.

Importantly, TreeSHAP does not only consider the leaf reached by the fully observed sample. For example, when `Age` is missing, the right subtree can also become relevant. Therefore, `Income` can receive a non-zero SHAP value even though it is not part of the sample's ordinary prediction path.


***What About a Feature That Is Never Used?***

`Smoke` does not appear in any split of our example tree. Whether `Smoke` is available or missing can therefore never change which leaf is reached or the resulting prediction. For every coalition $S$, $v_x(S\cup\{\text{Smoke}\})=v_x(S)$. Its marginal contribution is always zero, and therefore $\phi_{\text{Smoke}}(x)=0$. This is exactly the **dummy property** of Shapley values.

***From One Tree to a Random Forest***

For a Random Forest, TreeSHAP performs this calculation across all trees. Since a Random Forest regressor averages its tree predictions, the corresponding tree-level SHAP values are also averaged:

$$
\phi_i^{RF}(x)
=
\frac{1}{T}
\sum_{t=1}^{T}\phi_i^{(t)}(x).
$$

The resulting SHAP values still satisfy

$$
f_{RF}(x)
=
v_x(\emptyset)
+
\sum_i\phi_i^{RF}(x).
$$

The calculation is performed separately for every sample we want to explain because SHAP values describe contributions to an **individual prediction**.

***How Does TreeSHAP Make This Efficient?***

The calculations above explicitly considered coalitions to illustrate the idea. Doing this for every feature and every possible coalition would again be computationally expensive. TreeSHAP avoids this explicit enumeration using **dynamic programming**. While traversing the tree, it stores and updates the information required to represent many feature-present and feature-missing cases simultaneously. When it reaches a leaf, this information is used to calculate the corresponding Shapley-weighted feature contributions. Thus, TreeSHAP computes the same type of marginal contributions without explicitly constructing and evaluating every feature coalition.

<p align="center">
  <img src="https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/main/docs/source/_figures/shap_treeshap_overview.png?raw=true" width="95%">
</p>

<p align="center">
  <em>Overview of TreeSHAP. TreeSHAP exploits the structure of tree-based models to efficiently account for features being present or missing and compute their Shapley-based contributions without explicitly evaluating every feature coalition.</em>
</p>

