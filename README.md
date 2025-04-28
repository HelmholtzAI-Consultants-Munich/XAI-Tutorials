# Introduction to eXplainable AI 

During this course participants will get an introduction to the topic of Explainable AI (XAI). The goal of the course is to help participants understand how XAI methods can help uncover biases in the data or provide interesting insights. After a general introduction to XAI, the course goes deeper into state-of-the-art model agnostic interpretation techniques as well as a practical session covering these techniques. Finally, we will focus on two model specific post-hoc interpretation methods, with hands-on training covering interpretation of random forests and neural networks with imaging data to learn about strengths and weaknesses of these standard methods used in the field.

## Venue

The course will be fully online: [GatherTown link](https://app.gather.town/app/nkxyTbuI84smfiQk/HMC-Workshop-Lounge?spawnToken=u2qffvtMRHqJ9LhU6okM)  
Password: HMC-2024  
Room: 1  

## Schedule

| Time        | Session |
| ----------- | ------- |
|09:00 - 09:30|Introduction|30 min|

**Track: XAI for Random Forests**
| Time        | Session |
| ----------- | ------- |
|09:30 - 10:00	|Permutation Feature Importance|30 min|
|10:00 - 10:30	|LIME|30 min|
|10:30 - 10:40	|Break|10 min|
|10:40 - 11:40 	|SHAP|60 min|
|11:40 - 11:50	|Break|10 min|
|11:50 - 12:20	|FGC|30 min|
|12:20 - 12:50	|Comparison of XAI methods|30 min|
|12:50 - 13:00	|Conclusions|10 min|

**Track: XAI for CNNs**
| Time        | Session |
| ----------- | ------- |
|09:30 - 10:00	|Introduction to CNNs|30 min|
|10:00 - 10:10	|Break|10 min|
|10:10 - 11:10 	|Grad-CAM for Images|60 min|
|11:10 - 11:20	|Break|10 min|
|11:20 - 11:50	|LIME for Images|30 min|
|11:50 - 12:20	|SHAP for Images|30 min|
|12:20 - 12:50	|Comparison of XAI methods|30 min|
|12:50 - 13:00	|Conclusions|10 min|

## Mentors

- [Dr. Lisa Borros de Andrade e Sousa](mailto:lisa.barros@helmholtz-munich.de), Helmholtz Munich
- Dr. Elisabeth Georgii, Helmholtz Munich
- Isra Mekki, Helmholtz Munich
- Francesco Campi, Helmholtz Munich
- Sabrina Benassou, JSC

## Requirements and Setup

This course assumes you have minimal experience running Python and Machine Learning Frameworks like PyTorch and sklearn.

It is possible to either create an environment and install all the necessary packages locally (using the requirements.txt file) or to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This second option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your device, create a virtual environment using the requirements.txt file:
```
conda create -n xai python=3.10
conda activate xai
pip install -r requirements.txt
```

Once your environment is created, clone `2024-HelmholtzSummerSchool` brach branch of the repo using the following command:

```
git clone --branch 2024-HelmholtzSummerSchool https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
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
