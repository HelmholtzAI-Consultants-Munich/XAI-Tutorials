![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![test](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml)
[![stars](https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/XAI-Tutorials?logo=GitHub&color=yellow)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/stargazers)

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials)

# Tutorials for eXplainable Artificial Intelligence (XAI) methods

This repository contains a collection of self-explanatory tutorials for different model-agnostic and model-specific XAI methods.
Each tutorial comes in a Jupyter Notebook, containing a short video lecture and practical exercises.
The material has already been used in the context of two courses: the Helmholtz Summer Academy 2022 and 2023 (fully online) and ml4hearth (hybrid setting), MALTAomics Summer School (hybrid setting).
The course material can be adjusted according to the available time frame and the schedule.
The material is self-explanatory and can also be consumed offline.

The learning objectives are:

- understand the importance of interpretability
- discover the existing model-agnostic and model-specific XAI methods
- learn how to interpret the outputs and graphs of those methods with hands-on exercises
- learn to chose which method is suitable for a specific task

## Venue
The course will be fully online:
[Zoom link](https://fz-juelich-de.zoom.us/j/63066788688?pwd=bGpLZm9oN3lJSFJlbXRkQjdZMHRRQT09)

Meeting ID: 630 6678 8688, Passcode: 969908

Link to the share notes: https://notes.desy.de/HAmuRdemQgK8VW9mqjBBPQ?both

## Schedule at a glance

#### Day 1 - XAI for Random Forest
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

Homework 1: Comparison notebook - [Tutorial_XAI_for_RandomForest](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2024/xai-for-tabular-data/Tutorial_XAI_for_RandomForests.ipynb)

Homework 2: SHAP exercise - [Compute Shapley values by hand](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2024/SHAP_exercise.pdf)


#### Day 2 - XAI for CNNs
|  Time | Session  | Duration  |
|---|---|---|
|9:00 - 9:15 | Welcome |15 min|
|9:15 - 9:30 |	Homework Discussion| 15 min|
|9:30 - 10:00 | Intro CNNs|	30 min|
|10:00 - 10:15 | Break | 15 min| 
|10:15 - 11:05 | Grad-CAM for Images| 50 min|
|11:05- 11:45 |	 Grad-CAM for Signals | 40 min|
|11:45 - 12:00 | Break | 15 min| 
|12:00 - 12:30 | LIME for Images | 30 min|
|12:30 - 12:55| SHAP for Images | 25 min |
|12:55 - 13:00 | Conclusions |5 min|

Homework 1: Comparison notebook - [Tutorial_XAI_for_ImageAnalysis](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/Juelich-2024/xai-for-image-data/Tutorial_XAI_for_ImageAnalysis.ipynb)


#### Day 3 - XAI for Transformers
|  Time | Session  | Duration  |
|---|---|---|
|9:00 - 9:15 |Welcome |15 min|
|9:15 - 9:30 |	Homework Discussion| 15 min|
|9:30 - 10:15 | Intro to trasformers | 45 min|
|10:15 - 10:30 | Break | 15 min| 
|10:30 - 11:00 | Attention for text | 45 min|
|11:00 - 11:30 | Intro to Vision Transformers | 30 min|
|11:30 - 11:45 | Break | 15 min| 
|11:45 - 12:45 | Attention map for image transformers | 60 min|
|12:45 - 13:00 | Conclusions & Survey | 15 min|


## Requirements and Setup - *Check this section*

It is possible to either create an environment and install all the necessary packages locally (using the requirements.txt file) or to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This second option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your device, create a virtual environment using the requirements.txt file:
```
conda create -n XAI-Course-2024 python=3.9
conda activate XAI-Course-2024
pip install -r requirements.txt
```

Once your environment is created, clone `Juelich-2024` brach branch of the repo using the following command:

```
git clone --branch Juelich-2024 https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
```

## Contributions

Comments and input are very welcome! If you have a suggestion or think something should be changed, please open an issue or submit a pull request. 
