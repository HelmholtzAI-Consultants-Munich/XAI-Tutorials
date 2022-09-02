
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_permutation_feature_importance(result, data, title):

    perm_sorted_idx = result.importances_mean.argsort()
    perm_indices = np.arange(0, len(result.importances_mean)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    ax1.barh(perm_indices, result.importances_mean[perm_sorted_idx], height=0.7, color = 'lightblue')
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result.importances_mean)))
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=data.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()


def plot_impurity_feature_importance(importance,names,title):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(5,4))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], color = 'lightblue')
    #Add chart labels
    plt.title(title)
    plt.xlabel('feature importance')
    plt.ylabel('feature names')


def plot_correlation_matrix(data):

    f, ax = plt.subplots(figsize=(5, 5))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
    square=True, ax=ax, annot=True)
