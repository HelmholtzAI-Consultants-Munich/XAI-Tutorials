[![test](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/actions/workflows/test_notebooks.yml)
[![stars](https://img.shields.io/github/stars/HelmholtzAI-Consultants-Munich/XAI-Tutorials?logo=GitHub&color=yellow)](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/stargazers)

# Tutorials for eXplainable Artificial Intelligence (XAI)

<!-- LINK INTRODUCTION START -->

 *"The problem is that a single metric, such as classification accuracy, is an incomplete description of most real-world tasks." — (Doshi-Velez et al., 2017)*

This repository contains a collection of interactive tutorials for understanding and applying modern eXplainable Artificial Intelligence (XAI) methods to machine learning and deep learning models, including Random Forests, CNNs, and Transformers. Each tutorial is provided as a Jupyter Notebook combining short video lectures with practical hands-on exercises.

The tutorials cover both model-agnostic and model-specific XAI methods, including SHAP, LIME, Permutation Feature Importance, Grad-CAM, Attention Maps, and Forest-Guided Clustering.

The learning objectives are:

* understand the importance of interpretability and transparency in AI
* learn how different XAI methods work and when to use them
* interpret explanation outputs and visualizations for different model types
* gain hands-on experience applying XAI methods to real-world examples

<!-- LINK INTRODUCTION END -->

## 📚 Included Tutorials

The repository includes tutorials for both **model-agnostic** and **model-specific** XAI methods across tabular, image, and transformer-based models.

### Model-Agnostic Methods

Methods that can be applied independently of the underlying machine learning model:

* Permutation Feature Importance
* SHapley Additive exPlanations (SHAP)
* Local Interpretable Model-Agnostic Explanations (LIME)

### Model-Specific Methods

Methods designed for interpreting specific model architectures such as Random Forests, CNNs, and Transformers:

* Forest-Guided Clustering
* Grad-CAM
* Attention Maps

## 🚀 Requirements and Setup

The notebooks can either be executed locally or directly in the browser using the **Open in Colab** button. Running the notebooks in Colab does not require any installation, but a Google account is needed.

To run the notebooks locally, create a virtual environment and install the required packages:

```bash id="m4c6p2"
conda create -n xai python=3.12
conda activate xai

pip install -r requirements_xai-for-cnn.txt
pip install -r requirements_xai-for-random-forest.txt
pip install -r requirements_xai-for-transformer.txt
```

Clone the repository:

```bash id="k9x2tw"
git clone https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials.git
```

## 🎓 Using This Repository for Courses

This repository can easily be adapted for online or in-person teaching.
For course-specific material, we recommend creating a dedicated GitHub branch:

```bash id="v8k3pm"
git checkout -b <course-name>
```

You can then customize the notebooks and select only the material relevant for your course.

The folders `.github` and `docs` are only required for maintaining the main repository and can optionally be removed in course branches.

To automatically update notebook links and branch references, first set the `NEW_BRANCH` variable inside `update_branch_links.py` to the name of your course branch, then run:

```bash id="q1m7dx"
python update_branch_links.py
```

## 🤝 Contributions

<!-- LINK CONTRIBUTION START -->

Comments, suggestions, and contributions are very welcome!
If you have ideas for improvements or want to report an issue, feel free to open an issue or submit a pull request.

<!-- LINK CONTRIBUTION END -->

## 🛡️ License

<!-- LINK LICENSE START -->

All content is publicly available under the Creative Commons Attribution 4.0 License:

[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

<!-- LINK LICENSE END -->