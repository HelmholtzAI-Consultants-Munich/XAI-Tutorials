[![test](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml)
[![stars](https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/XAI-Tutorials?logo=GitHub&color=yellow)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/stargazers)


# Tutorials for eXplainable Artificial Intelligence (XAI) methods

This repository contains a collection of self-explanatory tutorials for different model-agnostic and model-specific XAI methods for Random Forests, CNNs and Transformers. Each tutorial comes in a Jupyter Notebook containing a short video lecture and practical exercises.

The learning objectives are:

- understand the importance of interpretability
- discover the existing model-agnostic and model-specific XAI methods
- learn how to interpret the outputs and graphs of those methods with hands-on exercises
- learn to choose which method is suitable for a specific task

For using the content of this repository for an online or offline course, please open a new GitHub branch in this repository with the name of the course and then choose the content you would like to use for your course. The folders `docs`, `test` and `.github` can be removed in the course branch as they are only needed in the main branch.

**List of Tutorials for Model-Agnostic Methods:**

- Permutation Feature Importance
- SHapley Additive exPlanations (SHAP)
- Local Interpretable Model-Agnostic Explanations (LIME)

**List of Tutorials for Model-Specific Methods:**

- Forest-Guided Clustering
- Grad-CAM
- Attention Maps

## Requirements and Setup

It is possible to either create an environment and install all the necessary packages locally (using the requirements.txt file) or to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This second option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your device, create a virtual environment using the requirements.txt file:
```
conda create -n xai python=3.12
conda activate xai
pip install -r requirements.txt
```

Once your environment is created, clone the repo using the following command:

```
git clone https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
```

## Contributions

Comments and input are very welcome! If you have a suggestion or think something should be changed, please open an issue or submit a pull request. 

All content is publicly available under the Creative Commons Attribution License: https://creativecommons.org/licenses/by/4.0/
