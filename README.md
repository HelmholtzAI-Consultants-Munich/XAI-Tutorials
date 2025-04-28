# Introduction to Explainable Deep Learning on Supercomputers

Join our comprehensive course to embark on an instructive journey into the world of eXplainable AI (XAI). Throughout the course, participants will develop a solid foundational understanding of XAI, primarily emphasizing how XAI methodologies can expose latent
biases in datasets and reveal valuable insights.
The course starts with a broad overview of XAI, setting the stage for a deep dive into cutting-edge model-agnostic interpretation techniques. As the course progresses, we shift our focus to model-specific post-hoc interpretation methods. Through immersive training, participants will learn to interpret machine learning algorithms and unravel the intricacies of deep neural networks, such as convolutional neural networks (CNN) and transformers. They will also become skilled in applying these techniques to various data formats, encompassing tabular data, images, and 1D data.
In addition to theoretical insights, participants will engage in hands-on practical sessions to apply these techniques effectively.
Take advantage of this opportunity to enhance your expertise in XAI and acquire the skills needed to navigate the intricate landscape of AI interpretability. Enroll now and unlock the potential of XAI!

Learning outcome:

- Gain an appreciation for the significance of XAI.
- Explore the available model-agnostic and model-specific XAI methodologies.
- Acquire the skills to interpret the results and visualizations of these methodologies through practical exercises.
- Master the skill of applying XAI techniques to diverse data types, including tabular data, images, and 1D data.
- Develop the ability to discern the most appropriate XAI method for a given task.

## Venue
The course will be fully online:
[Zoom link](https://fz-juelich-de.zoom.us/j/63066788688?pwd=bGpLZm9oN3lJSFJlbXRkQjdZMHRRQT09)  
Meeting ID: 630 6678 8688  
Passcode: 969908

## Schedule at a glance

#### Day 1 - XAI for Random Forest

|    Time     |       Session       |
|-------------|---------------------|
|09:00 - 09:30| Introduction to XAI |
|09:30 - 10:15|	Permutation Feature Importance|
|10:15 - 10:30| Break|
|10:30 - 11:30| SHAP |
|11:30 - 11:45| Break|
|11:45 - 12:15|	LIME |
|12:15 - 12:55| FGC |
|12:55 - 13:00| Conclusions |

Homework 1: Comparison notebook - [Tutorial_XAI_for_RandomForest](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2024/xai-for-tabular-data/Tutorial_XAI_for_RandomForests.ipynb)

Homework 2: SHAP exercise - [Compute Shapley values by hand](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2024/SHAP_exercise.pdf)

#### Day 2 - XAI for CNNs

|     Time     | Session |
|--------------|---------|
|09:00 - 09:15 | Welcome |
|09:15 - 09:30 | Homework Discussion |
|09:30 - 10:00 | Intro CNNs |
|10:00 - 10:15 | Break |
|10:15 - 11:05 | Grad-CAM for Images |
|11:05 - 11:45 | Grad-CAM for Signals |
|11:45 - 12:00 | Break |
|12:00 - 12:30 | LIME for Images |
|12:30 - 12:55 | SHAP for Images |
|12:55 - 13:00 | Conclusions |

Homework 1: Comparison notebook - [Tutorial_XAI_for_ImageAnalysis](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2024/xai-for-image-data/Tutorial_XAI_for_ImageAnalysis.ipynb)


#### Day 3 - XAI for Transformers

|     Time     | Session |
|--------------|---------|
|09:00 - 09:15 | Welcome |
|09:15 - 09:30 | Homework Discussion |
|09:30 - 10:15 | Intro to trasformers |
|10:15 - 10:30 | Break |
|10:30 - 11:00 | Attention for text |
|11:00 - 11:30 | Intro to Vision Transformers |
|11:30 - 11:45 | Break |
|11:45 - 12:45 | Attention map for image transformers |
|12:45 - 13:00 | Conclusions & Survey |


## Mentors

- Sabrina Benassou, JSC
- [Dr. Lisa Borros de Andrade e Sousa](mailto:lisa.barros@helmholtz-munich.de), Helmholtz Munich 
- Francesco Campi, Helmholtz Munich
- Isra Mekki, Helmholtz Munich

## Requirements and Setup

This course assumes you have minimal experience running Python and Machine Learning Frameworks like PyTorch and sklearn.

It is possible to either create an environment and install all the necessary packages locally (using the requirements.txt file) or to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This second option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your device, create a virtual environment using the requirements.txt file:
```
conda create -n xai python=3.9
conda activate xai
pip install -r requirements.txt
```

Once your environment is created, clone `2024-Juelich` brach branch of the repo using the following command:

```
git clone --branch 2024-Juelich https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
```

## Code of Conduct

Participants are expected to follow our code of conduct. In order to have a nice and collaborative environment, please follow these rules:

- respect the others
- turn camera on (if possible)
- mic turned off unless you want to speak/ask questions
- raise your hand to ask question or type them in the chat.

If you have any issues that you don’t want to share, send a private message to one of the mentors.

## Contributions

Comments and input are very welcome! If you have a suggestion or think something should be changed, please open an issue, submit a pull request or send an email to [Lisa Barros de Andrade e Sousa](mailto:lisa.barros@helmholtz-munich.de) or [Donatella Cea](mailto:donatella.cea@helmholtz-munich.de).

All content is publicly available under the Creative Commons Attribution License: https://creativecommons.org/licenses/by/4.0/
