Why is explainability important?
=================================

 *"The problem is that a single metric, such as classification accuracy, is an incomplete description of most real-world tasks." — (Doshi-Velez et al., 2017)*

Bias in AI systems
-------------------

Real-world data is messy! It contains missing entries, it can be skewed or subject to sampling errors, and it is often collected for purposes other than the analysis at hand.
Sampling errors or other issues in data collection can influence how well the resulting machine learning system works for different users. 
There have been a number of high profile instances of image recognition systems failing to work accurately for users from minority ethnic groups, for example.

The models created by a machine learning system can also generate issues of fairness or bias, even if trained on accurate data, and users need to be aware of the limitations of the systems they use. 
In recruitment, for example, systems that make predictions about the outcomes of job offers or training can be influenced by biases arising from social 
structures that are embedded in data at the point of collection. The resulting models can then reinforce these social biases, unless corrective actions are taken.

Concepts like fairness can have different meanings to different communities, and there can be trade-offs between these different interpretations. 
Questions about how to build "fair" algorithms are the subject of increasing interest in technical communities and ideas about how to create technical fixes 
to tackle these issues are evolving, but fairness remains a challenging issue. 
Fairness typically involves enforcing equality of some measure across individuals and/or groups, 
but many different notions of fairness are possible – these different notions can often be incompatible, requiring more discussions to negotiate inevitable trade-offs. [1]

Science
----------

Data collection and analysis is a core element of the scientific method, and scientists have long used statistical techniques to aid their work. 
In the early 1900s, for example, the development of the t-test gave researchers a new tool to extract insights from data in order to test the veracity of their hypotheses.

Today, machine learning has become a key tool for researchers across domains to analyse large datasets, detecting previously unforeseen patterns or extracting unexpected insights. 
Current application areas include:

- Analysing genomic data to predict protein structures, using machine learning approaches that can predict the three-dimensional structure of proteins from DNA sequences
- Understanding the effects of climate change on cities and regions, combining local observational data and large-scale climate models to provide a more detailed picture of the local impacts of climate change
- Finding patterns in astronomical data, detecting interesting features or signals from vast amounts of data that might include large amounts of noise, and classifying these features to understand the different objects or patterns being detected

In some contexts, the accuracy of these methods alone is sufficient to make AI useful – filtering telescope observations to identify likely targets for further study, for example. 
However, the goal of scientific discovery is to understand. Researchers want to know not just what the answer is but why.

Explainable AI can help researchers to understand the insights that come from research data, by providing accessible interpretations of how AI systems conduct their analysis. 
The Automated Statistician project, for example, has created a system which can generate an explanation of its forecasts or predictions, by breaking complicated datasets into 
interpretable sections and explaining its findings to the user in accessible language. This both helps researchers analyse large amounts of data, and helps enhance their understanding 
of the features of that data. [1]

Safety
----------

Medical imaging is an important tool for physicians in diagnosing a range of diseases, and informing decisions about treatment pathways. 
The images used in these analyses – scans of tissue samples, for example – require expertise to analyse and interpret. 
As the use of such imaging increases across different medical domains, this expertise is in increasingly high demand.

The use of AI to analyse patterns in medical images, and to make predictions about the likely presence or absence of disease, is a promising area of research. 
To be truly useful in clinical settings, however, these AI systems will need to work well in clinical practice – and clinicians and patients may both want 
to understand the reasoning behind a decision. If doctors or patients are unable to understand why an AI has made a specific prediction, 
there may be dilemmas about how much confidence to have in that system, especially when the treatments that follow can have life-altering effects.

A recent research project by DeepMind and Moorfield's Eye hospital points to new methods that can allow doctors to better understand AI systems in the context of medical imaging. 
This project looked at over 14,000 retinal scans, creating an AI system that analysed these images to detect retinal disease. 
Despite using deep learning techniques that would usually be considered "black box", researchers built the system so that human users were able to understand 
why it had made a recommendation about the presence or absence of disease. This explainability was built into the system by making it decomposable.

The system itself consists of two neural networks, each performing different functions:

- The first analyses a scan, using deep learning to detect features in the image that are illustrative of the presence (or absence) of disease – haemorrhages in the tissue, for example. This creates a map of the features in the image
- The second analyses this map, using the features identified by the first to present clinicians with a diagnosis, while also presenting a percentage to illustrate confidence in the analysis.

At the interface of these two systems, clinicians are able to access an intermediate representation that illustrates which areas of an image might suggest the presence of eye disease. 
This can be integrated into clinical workflows and interrogated by human experts wishing to understand the patterns in a scan and why a recommendation has been made, 
before confirming which treatment process is suitable. Clinicians therefore remain in the loop of making a diagnosis and can work with patients to confirm treatment pathways. [1]

Ethics
----------

Criminal justice risk assessment tools analyse the relationship between an individual's characteristics (demographics, record of offences, and so on) and their likelihood of 
committing a crime or being rehabilitated. Risk assessment tools have a long history of use in criminal justice, often in the context of making predictions about the likely 
future behaviour of repeat offenders. For some, such tools offer the hope of a fairer system, in which human bias or socially-influenced perceptions about who is a "risk" are 
less likely to influence how an individual is treated by the justice system.

The use of AI-enabled risk assessment tools therefore offers the possibility of increasing the accuracy and consistency of these predictive systems. 
However, the opacity of such tools has raised concerns in recent years, particularly in relation to fairness and the ability to contest a decision.

In some jurisdictions, there already exists legislation against the use of protected characteristics – such as race or gender – when making decisions about an 
individual's likelihood of reoffending. These features can be excluded from analysis in an AI-enabled system. however, even when these features are excluded, 
their association with other features can "bake in" unfairness in the system; for example, excluding information about ethnicity but including postcode data that 
might correlate with districts with high populations from minority communities. Without some form of transparency, it can be difficult to assess how such biases might 
influence an individual's risk score.

In the US, there have already been examples of AI-enabled systems being associated with unfair judicial outcomes, and of those affected by its outputs seeking to contest its results. 
In the debates that followed, the lack of transparency surrounding the use of AI – due to IP protections and trade secrets – were front and centre. 
This raised questions about whether a "black box" algorithm violates a right to due process; what provisions for explainability or other forms of 
public scrutiny are necessary when developing AI tools for deployment in public policy domains; about how more explainable AI tools could balance the desire for transparency 
with the risk of revealing sensitive personal information about an individual; and about the ways in which technological tools that appear neutral or 
authoritative could unduly influence their users. These are important areas for more research. [1]

References
-----------

[1] Explainable AI: the basics, The Royal Society, 2019. `Link <https://royalsociety.org/-/media/policy/projects/explainable-ai/AI-and-interpretability-policy-briefing.pdf>`_

[2] Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, Christoph Molnar, 2022. `Link <https://christophm.github.io/interpretable-ml-book/>`_


Additionally, Christoph Molnar's book and Tim Miller's paper can provide further insight into the challenges and promise of machine learning interpretability:

- `Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ -- Christoph Molnar, 2019-12-17
- `Explanation in Artificial Intelligence: Insights from the Social Sciences. <https://arxiv.org/abs/1706.07269>`_ -- Tim Miller