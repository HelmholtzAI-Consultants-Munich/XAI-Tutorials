![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![test](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml)
[![stars](https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/XAI-Tutorials?logo=GitHub&color=yellow)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/stargazers)

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials)

# Tutorials for eXplainable Artificial Intelligence (XAI) methods

This repository contains a collection of self-explanatory tutorials for different model-agnostic and model-specific XAI methods.
Each tutorial comes in a Jupyter Notebook which contains a short video lecture and practical exercises.
The material is self-explanatory and can be also be consumed offline.

The learning objectives are:

- understand the importance of interpretability
- discover the existing model-agnostic and model-specific XAI methods
- learn how to interpret the outputs and graphs of those methods with hands-on exercises
- learn to choose which method is suitable for a specific task
The event page can be found here: https://events.hifis.net/event/858/timetable/

## Venue
The course will be fully online:
- [GatherTown link](https://app.gather.town/app/nkxyTbuI84smfiQk/HMC-Workshop-Lounge)
- Password: ISA_Gather
- Room: 2

## Schedule at a glance

| Time          | Content |
| ------------- | -------- |
| 13.30 - 13.50 | Introduction to XAI |
| 13.50 - 15.50 | Tutorial on XAI model-agnostic methods 
| 15.50 - 16.00 | Break |
| 16.00 - 17.30 | Tutorials on “XAI in deep learning-based image analysis” and “XAI for Random Forest” |
| 17.30 - 17.35 | Wrap-up and conclusions |

## Tutorials:
- First session
  - [Tutorial_PermutationFeatureImportance](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-agnostic/Tutorial_PermutationFeatureImportance.ipynb)
  - [Tutorial_SHAP_intro](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-agnostic/Tutorial_SHAP.ipynb)
- Second session
  - XAI for deep learning image analysis:
    - [Model-CNN-Feature Visualization](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/data_and_models/Model-CNN-FeatureVisualization.ipynb)
    - [Tutorial_GradCAM](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-specific/Tutorial_Grad-CAM.ipynb)
    - [Tutorial_SHAP_images](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-agnostic/Tutorial_SHAP_Images.ipynb)
    - [Tutorial_XAI_for_ImageAnalysis](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-specific/Tutorial_XAI_for_ImageAnalysis.ipynb)
  - XAI for Random Forest:
    - [Tutorial_FGC](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-specific/Tutorial_FGC.ipynb)
    - [Tutorial_XAI_for_RAnfomForest](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/SummerAcademy-2023/xai-model-specific/Tutorial_XAI_for_RandomForests.ipynb)


## Requirements and Setup

It is possible to execute the notebooks on the browser, by clicking the 'Open in Colab' button. This option doesn't require any further installation, but the user must have access to a Google account.

If you prefer to run the notebooks on your own device, create a virtual environment using the requirements.txt file:
```
conda create -n XAI-Course-2023 python=3.9
conda activate XAI-Course-2023
pip install -r requirements.txt
```

Once your environment is created, clone `SummerAcademy-2023` brach branch of the repo using the following command:

```
git clone --branch SummerAcademy-2023 https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
```

## Contributions

Comments and input are very welcome! Please, if you have a suggestion or you think something should be changed, open an issue or submit a pull request. 
