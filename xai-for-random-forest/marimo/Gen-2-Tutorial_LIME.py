import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![logo](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/main/docs/source/_figures/Helmholtz-AI.png?raw=true)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model-Agnostic Interpretation with LIME

    In this Notebook, we will demonstrate how to use the Local Interpretable Model-Agnostic Explanations (LIME) method ([Rubiero et. al., 2016](https://doi.org/10.1145/2939672.2939778)) and interpret its results.

    --------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports

    Let's start with importing all required Python packages.
    """)
    return


@app.cell
def _():
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import altair as alt

    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.linear_model import LinearRegression, Ridge  # used as surrogate model for LIME

    import warnings

    warnings.filterwarnings("ignore")
    return (
        LimeTabularExplainer,
        LinearRegression,
        Path,
        Ridge,
        alt,
        np,
        pd,
        pickle,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we fix the random seeds to ensure reproducible results, as we work with (pseudo) random numbers.
    """)
    return


@app.cell
def _(np):
    # assert reproducible random number generation
    seed = 1
    np.random.seed(seed)
    return (seed,)


@app.cell
def _(alt, np, pd):
    def coef_bar_chart(df, title):
        """Bar chart of the local feature contributions of a LIME explanation."""
        data = df.copy()
        data["direction"] = np.where(
            data["weight"] >= 0, "increases prediction", "decreases prediction"
        )
        return (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("weight:Q", title="model coefficient"),
                y=alt.Y(
                    "feature:N",
                    sort=alt.EncodingSortField(field="weight", op="sum", order="descending"),
                    title="feature",
                ),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["increases prediction", "decreases prediction"],
                        range=["#2c7fb8", "#e34a33"],
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="feature"),
                    alt.Tooltip("weight:Q", title="coefficient", format="+.4f"),
                ],
            )
            .properties(title=title, width=560, height=320)
        )


    def lime_to_df(explanation, label=None):
        """Turn a LIME explanation into a DataFrame with 'feature' and 'weight' columns."""
        pairs = explanation.as_list(label=label) if label is not None else explanation.as_list()
        return pd.DataFrame(pairs, columns=["feature", "weight"])

    return coef_bar_chart, lime_to_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    --------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and Model Loading: The California Housing Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we will work with the **California Housing dataset**, containing 20,640 median house values for California districts (expressed in $100,000), which are described by 8 numeric feature. Each row in the dataset represents a block of houses, not a single household. The data pertains to the house prices found in a given California district and some summary statistics about them based on the 1990 census data. Our goal is to **predict price** of house blocks and find the most predictive features.

    <center><img src="https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/main/docs/source/_figures/dataset_california_housing.jpg?raw=true" width="900" /></center>

    <font size=1> Source: [Link](https://www.kaggle.com/datasets/harrywang/housing)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the notebook [*Gen-0-Tutorial_RandomForest_Model_Housing_Wine_Penguins.ipynb*](./Gen-0-Tutorial_RandomForest_Model_Housing_Wine_Penguins.ipynb), we explain how to do the exploratory data analysis, preprocess the data and train a Random Forest model with the given data. The focus of this notebook is the interpretation of the previously trained model.
    """)
    return


@app.cell
def _(Path, mo):
    def _find_models_dir():
        # Locate the folder with the trained-model pickles, anchored on this
        # notebook's own location so it works regardless of where the repo is cloned.
        here = mo.notebook_dir() or Path.cwd()
        candidates = [
            here / "models",                  # models bundled next to the notebook
            here / ".." / ".." / "models",    # <repo>/models (notebook in xai-for-random-forest/marimo/)
            here / ".." / "models",
            here / ".." / "XAI-Tutorials" / "models",
            Path("../../models"),             # fallbacks relative to the launch directory
            Path("../models"),
            Path("models"),
        ]
        for c in candidates:
            if (c / "model_rf_housing.pickle").exists():
                return c.resolve()
        raise FileNotFoundError(
            "Could not find model_rf_housing.pickle. Put a 'models' folder next to "
            "this notebook, or set MODELS_DIR to its location."
        )


    MODELS_DIR = _find_models_dir()
    MODELS_DIR
    return (MODELS_DIR,)


@app.cell
def _(MODELS_DIR, pickle):
    # Load and unpack the data
    with open(MODELS_DIR / "model_rf_housing.pickle", "rb") as _fh:
        _data_and_model = pickle.load(_fh)

    X_train_h = _data_and_model[0]
    X_test_h = _data_and_model[1]
    y_train_h = _data_and_model[2]
    y_test_h = _data_and_model[3]
    model_housing = _data_and_model[4]

    print(f"Model Performance on training data: {round(model_housing.score(X_train_h, y_train_h), 2)} R^2.")
    print(f"Model Performance on test data: {round(model_housing.score(X_test_h, y_test_h), 2)} R^2.")
    return X_train_h, model_housing, y_train_h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see by the numbers above, our model performs well on the training set and also generalizes well to the independent test set.

    **You should keep in mind that interpreting a low-performing model can lead to wrong conclusions.**

    *Note: The $R^2$ is the coefficient of determination, and the closer this value is to 1, the better our model explains the data. A constant model that always predicts the average target value disregarding the input features would get an $R^2$ score of 0. However, the $R^2$ score can also be negative because the model can be arbitrarily worse.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now, what does my model think is important in the data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Local Interpretable Model-Agnostic Explanations (LIME)

    **We prepared a small [Introduction to LIME](https://xai-tutorials.readthedocs.io/en/latest/_model_agnostic_xai/lime.html) for you, to help you understand how this method works.**

    *Note: we provide all references [here](https://xai-tutorials.readthedocs.io/en/latest/_model_agnostic_xai/lime.html#references).*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Question 1: What is a surrogate model?

    > **Your answer:** It's an interpretable model, such as a linear or tree model, that approximates the predictions of a more complex model.

    #### Question 2: How does LIME use surrogate models to explain a model prediction?

    > **Your answer:** It randomly samples data points in the neighbourhood of the sample point and lets the complex model make predictions for those points. This data set is then used to fit a surrogate model, which can then be analysed to identify important features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's use LIME to get some insights into the Random Forest Regression model we loaded above. We first have to specify an important parameter for LIME: the *kernel_width*, which, in principle, determines how large the neighborhood around our sample will be. The optimal choice of this parameter is difficult and currently still an open research question and one of the main disadvantages of the method. Feel free to play around with different values and observe how the generated explanations can change.

    *Note: this method is a local method, which means that it only provides explanations for individual samples but not for a full dataset.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the controls below to choose the instance to explain, the kernel width, the
    number of neighborhood samples, and the surrogate model.
    """)
    return


@app.cell
def _(X_train_h, mo):
    housing_idx = mo.ui.slider(
        start=0, stop=X_train_h.shape[0] - 1, step=1, value=0,
        label="instance index", show_value=True,
    )
    housing_kw = mo.ui.slider(
        start=0.25, stop=5.0, step=0.25, value=1.0,
        label="kernel width", show_value=True,
    )
    housing_ns = mo.ui.slider(
        start=1000, stop=8000, step=1000, value=5000,
        label="num samples", show_value=True,
    )
    housing_surrogate = mo.ui.dropdown(
        options=["Linear regression", "Ridge (alpha=1)"],
        value="Linear regression",
        label="surrogate model",
    )
    mo.vstack([housing_idx, housing_kw, housing_ns, housing_surrogate])
    return housing_idx, housing_kw, housing_ns, housing_surrogate


@app.cell
def _(LimeTabularExplainer, X_train_h, housing_kw, seed, y_train_h):
    explainer_h = LimeTabularExplainer(
        training_data=X_train_h,  # the data was standardized beforehand during training of the RF model
        mode="regression",
        training_labels=y_train_h,
        feature_names=X_train_h.columns,
        feature_selection="none",  # before applying the surrogate model, one could also select features
        random_state=seed,
        sample_around_instance=True,  # default: mean value of the training data; however, can be set to the instance itself to generate samples similar to our instance with high probability
        kernel_width=housing_kw.value,
        discretize_continuous=False,
    )
    return (explainer_h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we have defined the setup for LIME, we have to choose an instance of interest for which we want to get explanations, the neighborhood size, i.e. number of points we want to sample in the neighborhood of our instance of interest to build the surrogate model, and the type of surrogate model. The size of the neighborhood should be chosen large enough because the surrogate model needs a sufficient number of data points for training. As a surrogate model, we choose the Linear Regression model because the coefficients of a linear model allow an intrinsic interpretation of the model.

    *Note: when you are using a linear model as a surrogate model, you have to ensure that the input data is properly standardized beforehand. Our data was standardized before training the Random Forest model.*
    """)
    return


@app.cell
def _(X_train_h, y_train_h):
    # NOTE: LIME expects as input a numpy array. Hence, we convert all data frames to numpy arrays
    X_train_h_arr = X_train_h.to_numpy()
    y_train_h_arr = y_train_h.to_numpy()
    return X_train_h_arr, y_train_h_arr


@app.cell
def _(
    LinearRegression,
    Ridge,
    X_train_h_arr,
    explainer_h,
    housing_idx,
    housing_ns,
    housing_surrogate,
    model_housing,
    seed,
    y_train_h_arr,
):
    # choose an instance that you want to explain
    print(f"Instance {housing_idx.value} of training data will be explained.")

    _surrogate = (
        LinearRegression()
        if housing_surrogate.value == "Linear regression"
        else Ridge(alpha=1, random_state=seed)
    )
    _instance = X_train_h_arr[housing_idx.value]
    _instance_label = y_train_h_arr[housing_idx.value]

    explanation_h = explainer_h.explain_instance(
        data_row=_instance,
        predict_fn=model_housing.predict,  # prediction method of model that I want to explain
        labels=_instance_label,
        model_regressor=_surrogate,  # surrogate model
        num_samples=housing_ns.value,  # size of the neighborhood
    )
    return (explanation_h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All that LIME did was to fit a linear regression model to approximate the complex model's predictions, i.e., the predictions of the Random Forest model. The dataset for creating the fit consists of the neighborhood samples that were randomly created around our selected instance.

    Linear regression models estimate a single parameter (coefficient) for each feature. Those coefficients describe the mathematical relationship between each feature (independent variable) and the target (dependent variable). The sign of a linear regression coefficient tells you whether there is a positive or negative correlation between the feature and the target. The coefficient value signifies how much the mean of the target changes given a one-unit shift in the feature while holding other features in the model constant. Hence, we can plot the coefficients of the Linear Regression Surrogate model to understand which features are most predictive for our instance of interest.

    *Note: LIME is not restricted to linear regression models; other easy-to-interpret surrogate models could be used like decision trees.*
    """)
    return


@app.cell
def _(coef_bar_chart, explanation_h, housing_idx, lime_to_df, mo):
    housing_chart = mo.ui.altair_chart(
        coef_bar_chart(
            lime_to_df(explanation_h),
            title=f"LIME - local model coefficients (instance {housing_idx.value})",
        )
    )
    housing_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two of the most informative features for the model are the *median income* (MedInc) and *average occupancy* (AveOccup), which are the features with the highest absolute model coefficients. The positive sign of the *median income* coefficient indicates a positive relation between this feature and the target variable (*price*), i.e., a higher income leads to higher price predictions. On the other hand, the negative sign of the *average occupancy* coefficient suggests that as the occupancy (i.e., more people living in the house block) increases, the price tends to decrease.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To measure how well the simpler surrogate model is able to approximate the predictions of the more complex model, we can use any metric that summarizes the quality of the predictions that the surrogate model makes. This chosen metric can serve as a "fidelity measure", which indicates how reliable the interpretable model is. Even though the choice of the fidelity measure is up to you, it is important to assess the predictive ability of the surrogate model when LIME explanations are to be used!

    How much would you trust explanations delivered from a surrogate model that can not reasonably approximate the complex model's predictions?

    In our case, we used a linear regression model as a surrogate model. A commonly used metric to quantify the goodness of fit for the linear regression model is the $R^2$ score. It shows how well the linear model is able to approximate the predictions of the more complex model on the neighborhood samples. $R^2$ scores closer to 1 indicate better approximations. Our surrogate model achieves an $R^2$ score of ~ 0.8, indicating that the explanations given by that model can be trusted.
    """)
    return


@app.cell
def _(explanation_h):
    explanation_h.score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Question 3: What are the main differences to Permutation Feature Importance?

    > **Your answer:** LIME is purely local and can only explain one instance at a time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    --------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task: Apply LIME to the Wine Dataset

    In this exercise, you will apply **LIME (Local Interpretable Model-Agnostic Explanations)** to the trained Random Forest classifier in order to understand how different chemical properties influence individual wine classification predictions.

    Your task is to:

    1. Create an appropriate LIME explainer for the wine dataset.
    2. Generate LIME explanations for individual wine samples from different wine classes.
    3. Visualize and interpret the local feature contributions for the selected samples.
    4. Compare the explanations between different wine classes and individual observations.
    5. Investigate how the local feature importance changes for correctly and incorrectly classified samples.
    6. Experiment with different LIME parameters, such as the number of generated neighborhood samples (`num_samples`), the kernel width (`kernel_width`), or the choice of the surrogate model, and analyze how these settings influence the stability and interpretability of the explanations.

    > **Hint:** In contrast to global explanation methods such as Permutation Feature Importance, LIME provides local explanations for individual predictions. The resulting feature importance scores therefore depend on the selected wine sample.

    > **Hint:** The standard `LimeTabularExplainer` implementation is designed around linear surrogate models and therefore expects interpretable models with coefficients (e.g. linear or ridge regression). While nonlinear surrogate models such as decision trees could theoretically be used for local approximation, they are not directly supported by the default tabular LIME implementation.

    > **Hint:** Compare the local LIME explanations with the global feature importance patterns observed previously. Features that are globally important are not necessarily equally important for every individual wine sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The Wine dataset**

    Let's use the wine quality dataset to see how XAI can be used to explain multi-class classification models. The **Wine Recognition dataset** contains 178 wine samples from three different cultivators of wine in the same region in Italy. The wine was chemically analyzed and 13 different chemical attributes like *alcohol*, *malic acid*, *flavanoids* etc were measured. Our goal is to **classify wines** and find the most predictive features.

    <center><img src="https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/blob/main/docs/source/_figures/dataset_red_wine.jpg?raw=true" width="900" /></center>

    <font size=1> Source:
    [Link](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the notebook [*Gen-0-Tutorial_RandomForest_Model_Housing_Wine_Penguins.ipynb*](./Gen-0-Tutorial_RandomForest_Model_Housing_Wine_Penguins.ipynb), we explain how to do the exploratory data analysis, preprocess the data and train a Random Forest model with the given data. The focus of this notebook lies on the interpretation of the previously trained model.
    """)
    return


@app.cell
def _(MODELS_DIR, pickle):
    # Load and unpack the data
    with open(MODELS_DIR / "model_rf_wine.pickle", "rb") as _fh:
        _data_and_model = pickle.load(_fh)

    X_train_w = _data_and_model[0]
    X_test_w = _data_and_model[1]
    y_train_w = _data_and_model[2]
    y_test_w = _data_and_model[3]
    model_wine = _data_and_model[4]

    print(f"Model Performance on training data: {round(model_wine.score(X_train_w, y_train_w) * 100, 2)} % accuracy.")
    print(f"Model Performance on test data: {round(model_wine.score(X_test_w, y_test_w) * 100, 2)} % accuracy.")
    return X_train_w, model_wine, y_train_w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As shown by the metrics above, the model achieves perfect performance on the training set while also generalizing very well to the independent test set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Understanding Local Feature Contributions in the Wine Dataset**

    We can now apply LIME to better understand how the trained model uses the different chemical properties to make predictions for individual wine samples and distinguish between the three wine classes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the controls below to choose the wine sample, the kernel width, the number of
    samples, and the surrogate model.

    > Tip: set the sample index to **4**, **1**, and **101** to look at the three wine
    > classes (type_1 / type_2 / type_3) discussed in the conclusion.
    """)
    return


@app.cell
def _(X_train_w, mo):
    wine_idx = mo.ui.slider(
        start=0, stop=X_train_w.shape[0] - 1, step=1, value=4,
        label="instance index", show_value=True,
    )
    wine_kw = mo.ui.slider(
        start=0.5, stop=6.0, step=0.25, value=2.75,
        label="kernel width", show_value=True,
    )
    wine_ns = mo.ui.slider(
        start=1000, stop=8000, step=1000, value=5000,
        label="num samples", show_value=True,
    )
    wine_surrogate = mo.ui.dropdown(
        options=["Ridge (alpha=3)", "Linear regression"],
        value="Ridge (alpha=3)",
        label="surrogate model",
    )
    mo.vstack([wine_idx, wine_kw, wine_ns, wine_surrogate])
    return wine_idx, wine_kw, wine_ns, wine_surrogate


@app.cell
def _(X_train_w, y_train_w):
    X_train_w_arr = X_train_w.to_numpy()
    y_train_w_arr = y_train_w.to_numpy()
    return X_train_w_arr, y_train_w_arr


@app.cell
def _(LimeTabularExplainer, X_train_w, seed, wine_kw, y_train_w):
    explainer_w = LimeTabularExplainer(
        training_data=X_train_w,  # the data was standardized beforehand during training of the RF model
        mode="classification",
        training_labels=y_train_w,
        feature_names=X_train_w.columns,
        feature_selection="none",
        random_state=seed,
        sample_around_instance=True,
        kernel_width=wine_kw.value,
        discretize_continuous=False,
    )
    return (explainer_w,)


@app.cell
def _(
    LinearRegression,
    Ridge,
    X_train_w_arr,
    coef_bar_chart,
    explainer_w,
    lime_to_df,
    mo,
    model_wine,
    seed,
    wine_idx,
    wine_ns,
    wine_surrogate,
    y_train_w_arr,
):
    # choose an instance that you want to explain
    _surrogate = (
        Ridge(alpha=3, random_state=seed)
        if wine_surrogate.value == "Ridge (alpha=3)"
        else LinearRegression()
    )
    _instance = X_train_w_arr[wine_idx.value]
    _instance_label = y_train_w_arr[wine_idx.value]

    # generate LIME explanation
    explanation_w = explainer_w.explain_instance(
        data_row=_instance,  # instance to explain
        predict_fn=model_wine.predict_proba,  # probability predictions
        top_labels=1,  # explain highest predicted class
        model_regressor=_surrogate,  # local surrogate model
        num_samples=wine_ns.value,  # perturbed neighborhood samples
    )

    # get the actually explained class
    explained_label = explanation_w.top_labels[0]

    print(
        f"Instance {wine_idx.value} with true class {_instance_label} "
        f"is explained for predicted class {explained_label}. "
        f"The surrogate model score is {explanation_w.score:.2f}."
    )

    wine_chart = mo.ui.altair_chart(
        coef_bar_chart(
            lime_to_df(explanation_w, label=explained_label),
            title=f"LIME - wine instance {wine_idx.value} (predicted class {explained_label})",
        )
    )
    wine_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Conclusion**

    The LIME explanations provide local insights into how the Random Forest model distinguishes between the three wine classes based on the chemical properties of individual wine samples. Although the globally important features already highlighted the relevance of variables such as *flavanoids*, *proline*, *alcohol*, and *color intensity*, the local explanations reveal that the importance and direction of these features vary substantially between individual predictions.

    For the sample belonging to **class type_1**, features such as *flavanoids*, *proline*, and *total phenols* strongly support the prediction, while *alcalinity_of_ash* slightly decreases the model confidence. In contrast, the explanation for **class type_2** shows that high values of *proline*, *color intensity*, and *alcohol* negatively contribute to this class prediction, whereas features such as *hue* provide positive evidence. For the **class type_3** sample, the model mainly relies on negative contributions from *flavanoids*, *hue*, and *od280/od315_of_diluted_wines*, while features such as *alcohol* and *alcalinity_of_ash* increase the prediction confidence.

    The surrogate model scores indicate how well the local linear approximation captures the behavior of the underlying Random Forest model around the selected sample. While the explanation for class type_3 achieves a reasonably good surrogate score of approximately 0.75, the scores for class type_1 (0.55) and class type_2 (0.68) are comparatively low. This suggests that the local linear surrogate only approximates the true model behavior moderately well in these regions of the feature space. Consequently, the corresponding LIME explanations should be interpreted with caution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    --------
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extra Material: LIME computation step by step

    To get a better understanding of LIME, we will now guide you step by step through the algorithm. Even though LIME offers an easy-to-use API, it can be beneficial to have a quick look behind the scenes to fully understand what is going on. LIME is especially suitable here since the basic algorithm can be programmed in just a few steps.

    [Cristian Arteaga](https://nbviewer.org/urls/arteagac.github.io/blog/lime.ipynb) has also prepared a nice step-by-step explanation of LIME for a 2D toy problem and we recommend taking a look at his notebook which contains nice visualizations.
    Here, we are focussing on tabular data, but Christian also provides a notebook that demonstrates how LIME can work on other modalities like images as well, which is one of its big strengths!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1
    First, we select an instance for which we want to explain the prediction.
    We generate many normally distributed random samples that will serve as our sample neighborhood.
    The samples' expected value coincides with our instance to ensure sufficient similarity between the neighborhood and our instance, while the standard deviation is estimated from the training data.

    Also note that we make our instance itself part of the neighborhood.
    """)
    return


@app.cell
def _(X_train_h_arr, np, seed):
    # select instance of interest
    inst_idx = 0
    num_samples = 5000
    kernel_width = 1
    x = X_train_h_arr[inst_idx]

    # generate random perturbations around our selected instance
    # with given mean and standard deviation
    std = X_train_h_arr.std(axis=0)

    # NOTE: there are two options on setting the mean of the samples.
    # The default in LIME is to set it to the mean value of the training data.
    # However, it may be a better idea to set the mean to the instance itself
    # (in LIME this is done by sample_around_instance=True) in order
    # to generate samples similar to our instance with high probability.

    # mu = X_train_h.mean(axis=0)
    mu = x

    np.random.seed(seed)
    neighbors_of_x = np.random.normal(mu, std, size=(num_samples, X_train_h_arr.shape[1]))

    # sneak in the instance itself as part of the neighbors
    neighbors_of_x[0] = x
    return kernel_width, neighbors_of_x, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2
    We let our original model predict the outcomes of the neighborhood samples.
    """)
    return


@app.cell
def _(model_housing, neighbors_of_x):
    # use the trained rf model to generate labels
    neighbors_of_x_pred = model_housing.predict(neighbors_of_x)
    return (neighbors_of_x_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3
    We compute the distance of each neighbor to our instance and transform it to a weight which we will later use to fit our local surrogate model.
    Note that distances are computed on standardized data in order to avoid numerical instabilities.
    """)
    return


@app.cell
def _(kernel_width, neighbors_of_x, np, x):
    # compute euclidean distance of each neighbor to x
    # NOTE: distances are computed based on standardized data (including the instance of interest);
    # the data we loaded was already standardized beforehand before training the RF model.
    distance_to_x = np.sum((neighbors_of_x - x) ** 2, axis=1)
    weights = np.sqrt(np.exp(-1.0 * (distance_to_x / kernel_width) ** 2))
    return (weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4
    A surrogate model is fit to approximate the complex model's prediction on the neighborhood samples. We will use a linear regression model since that offers fairly straightforward explanations by looking at the estimated model coefficients. The fit will be performed again on the standardized samples!

    The weights computed above will serve to indicate their importance to the fit. Models with large weights are closer to our instance and should get predicted more accurately than neighbors further apart from our instance.
    """)
    return


@app.cell
def _(LinearRegression, neighbors_of_x, neighbors_of_x_pred, weights):
    # fit an explainable model on the scaled data to approximate the predictions of the
    # complex model on the neighborhood samples
    explainable_model = LinearRegression()
    explainable_model.fit(neighbors_of_x, neighbors_of_x_pred, sample_weight=weights)

    score = explainable_model.score(neighbors_of_x, neighbors_of_x_pred, sample_weight=weights)
    print("\nModel performance", score)
    return (explainable_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 5
    We visualize the model coefficients to obtain a similar explanation as the LIME API offers us.
    """)
    return


@app.cell
def _(X_train_h, coef_bar_chart, explainable_model, mo, pd):
    # get model coefficients
    coef_df = pd.DataFrame(
        {"feature": list(X_train_h.columns), "weight": explainable_model.coef_}
    )

    step_chart = mo.ui.altair_chart(
        coef_bar_chart(coef_df, title="LIME - local model coefficients")
    )
    step_chart
    return


if __name__ == "__main__":
    app.run()
