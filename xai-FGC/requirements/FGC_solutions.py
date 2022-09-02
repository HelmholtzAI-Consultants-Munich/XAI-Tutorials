#!/usr/bin/env python
# coding: utf-8

# ![logo](https://github.com/donatellacea/DL_tutorials/blob/main/notebooks/figures/1128-191-max.png?raw=true)

# # Explainability of Random Forests
# 
# In this Notebook we will show you different methods that can be used for interpreting Random Forest models. We will demonstrate you how to apply those methods and how to interpret the results.
# 
# --------

# ### Setup Colab environment
# 
# If you installed the packages and requirments on your own machine, you can skip this section and start from the import section.
# Otherwise you can follow and execute the tutorial on your browser. In order to start working on the notebook, click on the following button, this will open this page in the Colab environment and you will be able to execute the code on your own.
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HelmholtzAI-Consultants-Munich/Zero2Hero---Introduction-to-XAI/blob/master/xai-FGC/part2.ipynb)

# Now that you are visualizing the notebook in Colab, run the next cell to install the packages we will use:

# In[ ]:


get_ipython().system('pip install fgclustering')
get_ipython().system('pip install matplotlib==3.4.3')
get_ipython().system('pip install palmerpenguins')


# By running the next cell you are going to create a folder in your Google Drive. All the files for this tutorial will be uploaded to this folder. After the first execution you might receive some warning and notification, please follow these instructions:
# 1. Warning: This notebook was not authored by Google. *Click* on 'Run anyway'.
# 2. Permit this notebook to access your Google Drive files? *Click* on 'Yes', and select your account.
# 3. Google Drive for desktop wants to access your Google Account. *Click* on 'Allow'.
# 
# At this point, a folder has been created and you can navigate it through the lefthand panel in Colab, you might also have received an email that informs you about the access on your Google Drive. 

# In[ ]:


# Create a folder in your Google Drive
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_line_magic('cd', 'drive/MyDrive')


# In[ ]:


get_ipython().system('git clone https://github.com/HelmholtzAI-Consultants-Munich/Zero2Hero---Introduction-to-XAI.git')
# or !git clone git@github.com:HelmholtzAI-Consultants-Munich/Zero2Hero---Introduction-to-XAI.git   


# In[ ]:


get_ipython().run_line_magic('cd', 'Zero2Hero---Introduction-to-XAI/xai-FGC')


# ### Import

# In[ ]:


# Load the required packages

import joblib
import numpy as np
import pandas as pd

from palmerpenguins import load_penguins

from fgclustering import FgClustering
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import * 

from IPython.display import VimeoVideo

import warnings
warnings.filterwarnings('ignore')


# ## Data Pre-Processing and Model Training

# In this course, we will work with the **Palmer penguins dataset**, containing the information on 3 different species of penguins - Adelie, Chinstrap, and Gentoo - which were observed in the Palmer Archipelago near Palmer Station, Antarctica. The dataset consist of a total of 344 penguings, together with their size measurements, clutch observations, and blood isotope ratios. Our goal is to predict the species of Palmer penguins and find out the major differences among them.
# 
# <center><img src="./figures/penguins.png" width="400" /></center>
# 
# <font size=1> Source:\
# https://pypi.org/project/palmerpenguins/#description \
# https://allisonhorst.github.io/palmerpenguins/

# In[ ]:


# Load the data
penguins = load_penguins()

# Inspect the data
penguins.head()


# The focus of this notebook is on the interpretation of Random Forest models and not on the data pre-processing or model training part. If you want to learn more about data exploration and each step in the data pre-processing and model trainng pipeline, you can have a look the supplemental notebook [*FGC_supplement.ipynb*](./FGC_supplement.ipynb), which gives a detailed description for each of those steps. Here we briefly list the steps that are done, prepare the data and train the Random Forest model on the Palmer pinguins dataset. 
# 

# Before we start training the model, we need to do some preprocessing of our dataset. First, we need to take care of the missing values. In this example, we will apply the most common approach and simply omit those cases with the missing data and analyse the remaining data. In addition, categorical features need to be encoded, i.e. turned into numerical data. Here, we will use a simple Label encoding for the categorical features and for the target variable, which will transform the categorical feature values into unique integer values. 

# In[ ]:


# Copy data and save original data in penguins variable
data_penguins = pd.DataFrame(penguins.copy())

# Remove rows with missing values
data_penguins.dropna(inplace=True)

# Transform the target variable (Species) and the two categorical features (Sex, Island) with LabelEncoder
le1 = preprocessing.LabelEncoder()
data_penguins.species = le1.fit_transform(data_penguins.species)

le2 = preprocessing.LabelEncoder()
data_penguins.sex = le2.fit_transform(data_penguins.sex)

le3 = preprocessing.LabelEncoder()
data_penguins.island = le3.fit_transform(data_penguins.island)


# Now we are ready to train our Random Forest model! First, we define a small grid of hyperparameters that is used for model optimization. For demonstration pupose, we define a rather small grid of hyperparameters. Then, we define an instance of the RandomForestClassifier and run the GridSearchCV with the 5-fold cross validation to tune the model on the pre-defined set of hyperparameters. The model with the best hyperparameters is saved as the _best_estimator__ in the GridSearchCV instance. 

# In[ ]:


# Grid of hyperparameters 
hyper_grid_classifier = {'n_estimators': [100, 1000], 
            'max_depth': [2, 5, 10], 
            'max_samples': [0.8],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt','log2']
}


# In[ ]:


# A Random Forest instance from sklearn requires a separate input of feature matrix and target values. 
# Hence, we will first separate the target and feature columns. 
X_penguins = data_penguins.loc[:, data_penguins.columns != 'species']
y_penguins = data_penguins.species

# Define a classifier. We set the oob_score = True, as OOB is a good approximation of the test set score
classifier = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=1)

# Define a grid search with 5-fold CV and fit 
gridsearch_classifier = GridSearchCV(classifier, hyper_grid_classifier, cv=5, verbose=1)
gridsearch_classifier.fit(X_penguins, y_penguins)

# Take the best estimator
rf = gridsearch_classifier.best_estimator_


# If you have trouble pre-processing the data or training the Random Forest model, you can load the pre-processed dataset and pre-trained model that we prepared for you by uncommenting and running the following cell:

# In[ ]:


# Load the data
#data_penguins = pd.read_csv('./data/data_penguins_processed.csv', index_col=0)

# Separate the target and feature columns
#X_penguins = data_penguins.loc[:, data_penguins.columns != 'species']
#y_penguins = data_penguins.species

# Load the model
#rf = joblib.load(open('./models/random_forest_penguins.joblib', 'rb'))


# In[ ]:


# Check the results
print('OOB accuracy of prediction model:')
print(rf.oob_score_)


# Great, now you trained your Random Forest model! And it scored with the high OOB accuracy of 98%! 
# 
# But - is that all? Don't we want to know more? What about the explainability and deriving some knowledge out of it? Let us dive into the interpretation :)

# ## Interpreting Random Forest models with Feature Importance

# ### Permutation Feature Importance

# In the previous courses you were introduced to Permutation Feature Importance. Recall, the Permutation Feature Importance is defined to be the decrease in a model score when a single feature value is randomly shuffled. This procedure breaks the relationship between the feature and the target, thus the drop in the model score is indicative of how much the model depends on the feature. Now it is time to see how it works on the penguins dataset. 
# 

# In[ ]:


result = permutation_importance(rf, X_penguins, y_penguins, n_repeats=50, max_samples = 0.8, random_state=42)
plot_permutation_feature_importance(result=result, data=X_penguins, title="Permutation Feature Importance")


# <font color='green'>
# 
# #### Question 1: How big is the influence of the most important feature on the model performance?
# 
# <font color='grey'>
# 
# #### Your Answer: 
# 
# Permutation of the feature 'bill_length_mm' drops the accuracy by at most 0.3 (right plot), and on average 0.25 (left plot).
# 

# ### Random Forest Feature Importance
# 

# An alternative for Permutation Feature Importance is the Random Forest specific Feature Importance method based on the mean decrease in impurity. The mean decrease in impurity is defined as the total decrease in node impurity (weighted by the probability of reaching that node, approximated by the proportion of samples reaching that node) averaged over all trees of the ensemble. This Feature Importances is directly provided by the fitted attribute _feature_importances__ .
# 
# Lets plot the feature importance based on mean decrease in impurity:

# In[ ]:


plot_impurity_feature_importance(rf.feature_importances_, names=X_penguins.columns, title="Random Forest Feature Importance")


# <font color='green'>
# 
# #### Question 2: Inspect the differences between the results of the two feature importance plots. What do you notice? 
# _Hint:_ Take a look at the correlation plot below (run the cell to see it)
# 
# <font color='grey'>
# 
# #### Your Answer: 
# 
# 1. Random Forest Feature Importance identifies more important features than the Permutation Feature Importance.
# 2. It seems that the feature importance of the correlated features flipper_length and body_mass are artificially lower due to the high correlation. Random Forest Feature Importance does not seem to be affected by this correlation effect. This shows that Permutation Feature Importance results should be interpreted with great care in the presence of correlated features.

# In[ ]:


plot_correlation_matrix(X_penguins)


# Even though the Random Forest Feature Importance does overcome some disadvantages of Permutation Feature Importance, it does not give us more information about the class-specific differences and further insights into the decision paths of the Random Forest model. Therefore, we developed a Random Forest specific interpretability method called Forest-Guided Clustering (FGC) that leverages the tree structure of Random Forest models to get insights into the decision making process of the model. Let us dive into the method... 

# ## Interpreting Random Forest models with Forest-Guided Clustering
# 
# We prepared a small video lecture for you as an Introduction to Forest-Guided Clustering.
# For additional information you can have a look at the documentation for the FGC: https://forest-guided-clustering.readthedocs.io/en/latest/

# In[ ]:


from IPython.display import VimeoVideo
VimeoVideo("745319036/a86f126018")


# We will use FGC to gain more insights into the decision making process of the Random Forest model we trained previously. Afterwards, we will compare the feature importance results obtained by the previous methods and with FGC.

# In[ ]:


# create an FGC instance
fgc = FgClustering(model=rf, data=X_penguins, target_column=y_penguins)


# FGC is based on the K-Medoids clustering algorithm, which requires a predefined number of clusters as input. FGC is able to optimize the number of clusters based on a scoring function, which is minimizing the model bias while restricting the model complexity. The argument _number_of_clusters_ is used to either pass the predefined number of clusters or should be left empty if optimization is desired. 
# 
# For the sake of example and since the optimization part takes some time, we will set the number of cluster equal to the number of species present in the penguins dataset.

# In[ ]:


# Run the fgc instance:
fgc.run(number_of_clusters=3)


# FGC provides couple of ways to visualise the results and help interpret them:
# 
# - visualise global and local feature importance: features that show different and concise value distributions across clusters are defined to be globally or locally important
# - reveal the decision rules of RF model by visualizing feature patterns per cluster

# ### Global and Local Feature importance provided by FGC

# **Global feature importance** is represetned as the significance of the difference between cluster-wise feature distributions as a measure of global feature importance (ANOVA for continuous features and chi square for categorical features). **Features, which have significantly different distributions across clusters, have a high feature importance**, while features, which have a similar feature distribution across clusters have a low feature importance.
# 
# In addition to the global feature importance, we also provide a **local feature importance**, which gives the **importance of each feature for each cluster**. For the local feature importance we pre-filter the features based on the global feature importance (_thr_pvalue_ is used for the filtering step, just as in the plots before). Here, a feature is considered important if its distribution in a particlular cluster is clearly different from the feature distribution in the whole dataset.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Plot global feature importance
fgc.plot_global_feature_importance()
# Plot local feature importance
fgc.plot_local_feature_importance(thr_pvalue=1) # Set thr_pvalue=1 to show all the features


# <font color='green'>
# 
# #### Question 3: What do you observe when comparing the Random Forest Feature Importance and the FGC Feature Importance?
# 
# 
# <font color='grey'>
# 
# #### Your Answer: 
# 
# The global feature importance gives us the same results as the Random Forest Feature Importance. The local feature importance reveals more information. For example, the feature island is important for the cluster 1 and 2, but not for the cluster 0.
# 
# 

# ### Visualizing the decision paths of the Random Forest model

# Forest-Guided Clustering provides the special option to visualize the decision path of a Random Forest model, reflecting the decision making process of that model, in a heatmap summary plot and a feature-wise distribution plot. The heatmap provides a general overview on the target value attribution and feature enrichment / depletion per cluster.  We can see which classes/target values fall into which cluster and samples that fall into the "wrong" cluster can be inspected further as they might be extreme outliers or wrongly labelled samples / measurement errors. The distribution plots contain the same information as the heatmap just presented in a different way. Here the features are not standardized and we can see the actual scale of each feature on the y axis. Furthermore, we get an idea of the distribution of feature values within each cluster, e.g. having a small or high within-cluster-variation. 
# 
# We can choose which features we want to plot by specifying the _p_-value threshold applied to the _p_-values of the features from the global feature importance calculation. The default threshold _thr_pvalue_ is set to 0.01. By selecting a lower p-value threshold, we only plot features that show high differences between cluster-wise feature distributions. 
# 
# Remember, we were transforming our target variable into integeres with the LabelEncoder instance. The resulting mapping from that process looks like this:
# 
# - Adelie = 0
# - Chinstrap = 1
# - Gentoo = 2
# 

# In[ ]:


fgc.plot_decision_paths(thr_pvalue=0.01) # feel free to try different p-values thresholds


# <font color='green'>
# 
# #### Question 4: Why are the features 'sex' and 'year' not shown on the plots above?
# 
# 
# <font color='grey'>
# 
# #### Your Answer: 
# We only show features with p-values < thr_pvalue = 0.01. This means that these two features don't show significant difference between clusters. Hence, they don't seem to play a role in the decision making process of this random forest model

# <font color='green'>
# 
# #### Question 5: Try to describe species by observing the plots (Adelie = 0, Chinstrap = 1, Gentoo = 2). Use the following examples to guide you:
# 
# - What makes Gentoo different from the other two species? 
# - What makes Chinstrap different from Adelie?
# - ...
# 
# <center><img src="./figures/bill_length.png" width="200" /></center>
# 
# 
# <font color='grey'>
# 
# #### Your Answer: 
# - Gento has a larger body mass and smaller bill depth
# - Adelie has smaller bill lenght
# 

# 
