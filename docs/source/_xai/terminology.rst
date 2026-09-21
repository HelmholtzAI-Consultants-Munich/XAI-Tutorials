Terminology
==============

**Explainability or Interpretability?**

The distinction between *interpretability* and *explainability* is not universally agreed upon. 
Different authors, research communities, and software libraries may use these terms differently. 

**Interpretability**

Interpretability is the degree to which a human can understand the internal mechanics of a model 
without external tools. It focuses on the transparency of the model itself. Interpretable models, 
often referred to as *glass-box models*, allow humans to directly inspect how predictions are generated.

**Explainability**

Explainability is the extent to which the behaviour or internal mechanics of a machine learning model 
can be explained in human-understandable terms, often using *post hoc* methods. It focuses on providing 
insights into the behaviour of *black-box models*, whose internal decision-making process is not directly accessible.

In practice, explainability methods aim to answer questions such as *Why did the model make this prediction?*, 
*Which features were most influential?*, or *How would the prediction change if the input were different?* 
These explanations are generated after the model has been trained and do not require modifying the underlying model.

In this course, we primarily focus on explainability techniques for black-box models, with an emphasis on widely used post hoc methods. While intrinsically interpretable models are an important area of machine learning, they are outside the scope of this course.

References
-----------

The Royal Society. `Explainable AI: The basics. <https://royalsociety.org/-/media/policy/projects/explainable-ai/AI-and-interpretability-policy-briefing.pdf>`_ Policy Briefing. 2019. 
