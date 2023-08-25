Why is explainability important?
=================================

 *"The problem is that a single metric, such as classification accuracy, is an incomplete description of most real-world tasks." — (Doshi-Velez et al., 2017)*

To Mitigate Ethical Issues
-------------------

Real-world data is messy! Databases created from real-world data are incomplete, skewed, and subject to sampling errors. Additionally, data is often collected for a particular purpose, when such databases are made publicly available, they often only limitedly fit the use case of a new ML model. Such, and other, issues influence how well a machine learning system works. 

Even though machine learning models’ performances are usually output as one – or few – metrics, a single model likely works unequally well for different people. That is, because people are different, and their individual characteristics map the statistical model unequally well. Even subtle differences in performance for different people or population groups can lead to issues, such as biases or fairness issues, and users need to be aware of the limitations of the systems they use. Such issues have been found in several high-profile instances of image recognition, where systems failed to work accurately for users from ethnic minority groups.  Another well-known example would be machine learning-aided job recruitment: when socially biased data is used to train machine learning models that are built to predict who should or should not be recruited, the resulting models can reinforce these social biases and, for example, favor males over females in the recruitment process. To prevent machine learning models from perpetuating or even reinforcing adverse effects, actions must be taken. 

Concepts like fairness can have different meanings to different communities, and there can be trade-offs between these different interpretations. 
Questions about how to build "fair" algorithms are the subject of increasing interest in technical communities and ideas about how to create technical fixes 
to tackle these issues are evolving, but fairness remains a challenging issue. 
Fairness typically involves enforcing equality of some measure across individuals and/or groups, 
but many different notions of fairness are possible – these different notions can often be incompatible, requiring more discussions to negotiate inevitable trade-offs. [1]

For Science
----------

The goal of scientific discovery is to understand. Data collection and analysis is a core element of scientific methods, and scientists have long used statistical techniques to aid their work. As early as the 1900s the development of the t-test gave researchers a new tool to extract insights from data to test the veracity of their hypotheses. Today, machine learning has become a vital tool for researchers across domains to analyze large datasets, detect previously unforeseen patterns, or extract unexpected insights. 

For example, machine learning is used in these areas:

- In computational biology, machine learning is used to analyze genomic data, or to predict the three-dimensional structures of proteins 
- In computational climate science, machine learning is applied to understand the effects of climate change on cities and regions. By combining local observational data to large-scale climate models, researchers hope to acquire more detailed pictures of the local impacts of climate change
- In computational physics, machine learning is used to find patterns in astronomical data, to detect interesting features or signals from vast amounts of, sometimes very noisy data, and to classify these features to understand the different objects or patterns being detected

In some contexts, the accuracy of these methods alone is sufficient to make AI useful – filtering telescope observations to identify likely targets for further study, for example. However, researchers want to know not just what the answer is but why! Explainable AI can help researchers to understand the insights that come from research data, by providing accessible explanations of which features have been particularly relevant in the creation of the system’s output. A project that is working on explainable AI for science is, for example, the Automated Statistician project. They have created a system that can generate an explanation of its forecasts or predictions, by breaking complicated datasets into interpretable sections and explaining its findings in accessible language. Together, the machine learning system and the explanations help researchers analyze large amounts of data and to enhance their understanding of the features of that data. [1]

For technology acceptance 
----------

It is a long way from research to application! Having machine learning tools at hand - even such that work very well - does not necessarily mean they will be used in practice. Let’s look at medicine as an example: Medical imaging is an essential tool for physicians in diagnosing a range of diseases – and there are a lot of excellent machine learning tools available that can assist physicians in interpreting medical images such as X-rays, retina scans, or ultrasounds. But, in practice, high-performance scores are not enough to ensure a successful implementation of machine learning algorithms in clinical practice. A core element for doctors to use any technology in their work is trust and in order to trust, they need to be able to understand why an AI has made a specific prediction. This understanding is crucial for doctors to identify edge cases in which they should reject a model’s prediction. The identification of such a situation, however, might have life-altering effects for the individual affected patient. 
A recent research project by DeepMind and Moorfield’s Eye Hospital points to new methods that can allow doctors to better understand AI systems in the context of medical imaging. This project looked at over 14,000 retinal scans, creating an AI system that analyzed these images to detect retinal disease. On top of using deep learning techniques that would usually be considered “black box”, researchers built explainability mechanisms, so that human users were able to understand why the model had made a recommendation about the presence or absence of disease. 
This explainability was built into the system, making it decomposable. The system itself consists of two neural networks, each performing different functions:
- The first analyses a scan, using deep learning to detect features in the image that illustrate the presence (or absence) of disease – hemorrhages in the tissue, for example. This creates a map of the features in the image.
- The second analyses this map, using the features identified by the first to present clinicians with a diagnosis, while also presenting a percentage to illustrate confidence in the analysis.

At the interface of these two systems, clinicians can access an intermediate representation that illustrates which areas of an image might suggest the presence of eye disease. This can be integrated into clinical workflows and interrogated by human experts wishing to understand the patterns in a scan and why a recommendation has been made, before confirming which treatment process is suitable. Clinicians, therefore, remain in the loop of making a diagnosis and can work with patients to confirm treatment pathways. [1]


To meet regulation requirements (TBC)
----------



References
-----------

[1] Explainable AI: The basics, The Royal Society, 2019. `Link <https://royalsociety.org/-/media/policy/projects/explainable-ai/AI-and-interpretability-policy-briefing.pdf>`_

[2] Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, Christoph Molnar, 2022. `Link <https://christophm.github.io/interpretable-ml-book/>`_


Additionally, Christoph Molnar's book and Tim Miller's paper can provide further insight into the challenges and promise of machine learning interpretability:

- `Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable. <https://christophm.github.io/interpretable-ml-book/>`_ -- Christoph Molnar, 2019-12-17
- `Explanation in Artificial Intelligence: Insights from the Social Sciences. <https://arxiv.org/abs/1706.07269>`_ -- Tim Miller
