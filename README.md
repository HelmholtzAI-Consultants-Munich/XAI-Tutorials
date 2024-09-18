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
[Zoom link](https://fz-juelich-de.zoom.us/j/68186324087?pwd=eUp5dEdyU2xSODhzMXNRVW9vNkh5QT09)  
Meeting ID: 681 8632 4087  
Passcode: 900523 

## Schedule at a glance

#### Day 1 - XAI for Tabular Data

|  Time | Session  | Duration  |
|---|---|---|
|9:00 - 9:30 |Introduction |30 min|
|9:30 - 10:15 |	Permutation Feature Importance|	45 min|
|10:15 - 10:30 | Break|	15 min|
| 10:30 - 11:30 | SHAP | 1 h| 
|11:30 - 11:45 | Break|	10 min|
|11:45- 12:15 |	LIME | 30 min|
|12:15 - 12:55 | FGC |40 min|
|12:55 - 13:00 | Conclusions |5 min|

Homework 1: Comparison notebook - [Tutorial_XAI_for_RandomForest](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2023/xai-for-tabular-data/Tutorial_XAI_for_RandomForests.ipynb)

Homework 2: SHAP exercise - [Compute Shapley values by hand](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2023/SHAP_exercise.pdf)


#### Day 2 - XAI for Image Data

|  Time | Session  | Duration  |
|---|---|---|
|9:00 - 9:15 |Welcome |15 min|
|9:15 - 9:30 |	Homework Discussion| 15 min|
|9:30 - 10:00 | Intro CNNs|	30 min|
|10:00 - 10:15 | Break | 15 min| 
|10:15 - 11:45 | Grad-CAM| 1 h 15 min|
|11:45 - 12:00 | Break | 15 min| 
|12:00- 12:30 |	LIME for Images | 45 min|
|12:30 - 12:55 | SHAP for Images | 30 min|
|12:55 - 13:00 | Conclusions |5 min|

Homework 1: Comparison notebook - [Tutorial_XAI_for_ImageAnalysis](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2023/xai-for-image-data/Tutorial_XAI_for_ImageAnalysis.ipynb)


#### Day 3 - XAI for 1D Data

|  Time | Session  | Duration  |
|---|---|---|
|9:00 - 9:15 |Welcome |15 min|
|9:15 - 9:30 |	Homework Discussion| 15 min|
|9:30 - 10:15 | CAM for signals | 45 min|
|10:15 - 10:30 | Break | 15 min| 
|10:30 - 11:15 | Intro to trasformers | 45 min|
|11:15 - 11:35 | Attention for text | 20 min |
|11:35 - 11:50 | Break | 15 min| 
|11:50 - 12:10 | Intro to Vision Transformers | 20 min |
|12:10- 12:55 |	Attention map for image transformers | 45 min|
|12:55 - 13:00 | Conclusions |5 min|

## Mentors

- Sabrina Benassou, JSC
- Dr. Donatella Cea, Helmholtz Munich ([donatella.cea@helmholtz-munich.de](mailto:donatella.cea@helmholtz-munich.de))
- Dr. Lisa Borros de Andrade e Sousa, Helmholtz Munich ([lisa.barros@helmholtz-munich.de](mailto:lisa.barros@helmholtz-munich.de))
- Dr. Alina Bazarova, JSC
- Dr. Elisabeth Georgii, Helmholtz Munich
- Francesco Campi, Helmholtz Munich
- Isra Mekki, Helmholtz Munich

## Requirements and Setup

This course assumes you have minimal experience running Python and Machine Learning Frameworks like Tensorflow and PyTorch.

It is possible to either create an environment and install all the necessary packages locally (using the requirements.txt file) or to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This second option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your device, create a virtual environment using the requirements.txt file:
```
conda create -n XAI-Course-2023 python=3.9
conda activate XAI-Course-2023
pip install -r requirements.txt
```

Once your environment is created, clone `ECCB-2024` brach branch of the repo using the following command:

```
git clone --branch Juelich-2023 https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
```

## Code of Conduct

Participants are expected to follow our code of conduct. In order to have a nice and collaborative environment, please follow these rules:

- respect the others
- turn camera on (if possible)
- mic turned off unless you want to speak/ask questions
- raise your hand to ask question or type them in the chat.

If you have any issues that you don’t want to share, send a private message to one of the mentors.

## Contributions

Comments and input are very welcome! If you have a suggestion or think something should be changed, please open an issue or submit a pull request. 

All content is publicly available under the Creative Commons Attribution License: https://creativecommons.org/licenses/by/4.0/

