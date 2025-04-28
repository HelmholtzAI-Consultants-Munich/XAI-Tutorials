# Introduction to Explainable AI

This 2-half-day course provides an introduction to the topic of Explainable AI (XAI). This fundamental knowledge is to be used as a starting point for self-guided learning during and beyond the course time. All course days cover alternating sequences of theoretical input and hands-on exercises, which are discussed with the instructors during the course.

The goal of the course is to help participants understand how XAI methods can help uncover biases in the data or provide interesting insights. After a general introduction to XAI, the course goes deeper into state-of-the-art model agnostic as well as model-specific interpretation techniques. The practical hands-on sessions will help to learn about strengths and weaknesses of these standard methods used in the field.

## Venue
The course will be fully online:  

[Zoom Link Day 1](https://zoom.us/j/93041152208?pwd=t2nePps1ib0g5DzGdeCT0X4QBDLPwQ.1)  
Meeting-ID: 930 4115 2208  
Kenncode: 795548  

[Zoom Link Day 2](https://zoom.us/j/93607982165?pwd=4yHLKPP2LYZu67vKf01tk87eZ3H9km.1)  
Meeting-ID: 936 0798 2165  
Kenncode: 967680  

## Schedule at a glance

#### Day 1 - XAI for Random Forest

|    Time     |       Session       |
|-------------|---------------------|
|10:00 - 10:30| Introduction to XAI |
|10:30 - 11:00|	Permutation Feature Importance|
|11:00 - 12:00| SHAP |
|12:00 - 13:00| Lunch Break|
|13:00 - 13:30| LIME |
|13:30 - 14:15| FGC|
|14:15 - 14:55|	Method Comparison |
|14:55 - 15:00| Conclusions |

Extra Material: SHAP exercise - [Compute Shapley values by hand]()

#### Day 2 - XAI for CNNs

|     Time     | Session |
|--------------|---------|
|10:00 - 10:15 | Welcome |
|10:15 - 10:45 | Intro CNNs |
|10:45 - 11:45 | Grad-CAM for Images |
|11:45 - 12:00 | Extra: Grad-CAM for Signals |
|12:00 - 13:00 | Lunch Break |
|13:00 - 13:30 | LIME for Images |
|13:30 - 14:00 | SHAP for Images |
|14:00 - 14:55 | Method Comparison |
|14:55 - 15:00 | Conclusions |


## Mentors


- [Dr. Lisa Borros de Andrade e Sousa](mailto:lisa.barros@helmholtz-munich.de), Helmholtz Munich
- Dr. Elisabeth Georgii, Helmholtz Munich
- Sabrina Benassou, JSC
- Francesco Campi, Helmholtz Munich
- Karol Szustakowski, Helmholtz Munich
- Serena Sritharan, Helmholtz Munich

## Requirements and Setup

This course assumes you have minimal experience running Python and Machine Learning Frameworks like PyTorch and sklearn.

It is possible to either create an environment and install all the necessary packages locally (using the requirements.txt file) or to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This second option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your device, create a virtual environment using the requirements.txt file:
```
conda create -n xai python=3.11
conda activate xai
pip install -r requirements_xai-for-cnn.txt
pip install -r requirements_xai-for-random-forest.txt
```

Once your environment is created, clone `2025-HIDA-Spring` brach branch of the repo using the following command:

```
git clone --branch 2025-HIDA-Spring https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
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
