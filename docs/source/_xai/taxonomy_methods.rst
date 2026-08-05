Taxonomy of XAI methods
=======================

Explainability methods can be categorized along two complementary dimensions:

1. **Model-specific vs. model-agnostic** – whether a method is tied to a
   particular model class or can be applied to any machine learning model.
2. **Global vs. local** – whether a method explains the overall behaviour
   of a model or a single prediction.

Together, these dimensions form four categories.

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * -
     - **Global**
     - **Local**
   * - **Model-agnostic**
     - Explain the overall behaviour of any model.
     - Explain an individual prediction of any model.
   * - **Model-specific**
     - Explain the overall behaviour of a particular model family.
     - Explain an individual prediction of a particular model family.


**Model-agnostic methods**

Model-agnostic methods treat the machine learning model as a black box.
They only require access to the model inputs and outputs, making them
applicable to virtually any predictive model. Their flexibility comes at
the cost of potentially higher computational effort, and the resulting
explanations are approximations of the model behaviour.

**Model-specific methods**

Model-specific methods exploit the internal structure of a particular
model class. They are often computationally more efficient and can
produce explanations that are closely aligned with the underlying model,
but they cannot easily be transferred to other model architectures.

**Global explanations**

Global explanations describe the behaviour of a model over an entire
dataset. They help answer questions such as:

- Which features are generally most influential?
- How does the model respond to changes in a feature?
- What decision strategy has the model learned?

Global explanations are particularly useful for model validation,
debugging, and scientific interpretation.


**Local explanations**

Local explanations focus on a single prediction. They answer questions
such as:

- Why did the model predict this particular outcome?
- Which features contributed most to this prediction?
- How would the prediction change if the input were slightly different?

Local explanations are especially useful for decision support,
debugging individual predictions, and communicating results to end users.
