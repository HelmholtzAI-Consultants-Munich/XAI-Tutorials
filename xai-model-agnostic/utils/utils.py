

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_permutation_feature_importance(result, data, title):

    perm_sorted_idx = result.importances_mean.argsort()
    perm_indices = np.arange(0, len(result.importances_mean)) + 0.5

    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4))
    fig.suptitle(title)
    ax1.barh(perm_indices, result.importances_mean[perm_sorted_idx], height=0.7, color = 'cornflowerblue')
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result.importances_mean)))
    ax1.axvline(x=0, color='.5')
    fig.tight_layout()
    plt.show()


def plot_permutation_feature_importance_with_variance(result, data, title):

    perm_sorted_idx = result.importances_mean.argsort()
    perm_indices = np.arange(0, len(result.importances_mean)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)
    ax1.barh(perm_indices, result.importances_mean[perm_sorted_idx], height=0.7, color = 'cornflowerblue')
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result.importances_mean)))
    ax1.axvline(x=0, color='.5')
    ax2.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=data.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()


def plot_explanation(explanation):
    explanation_df = pd.DataFrame({k: v for k, v in explanation.items() if k != "importances"}).sort_values(
        by="importances_mean", ascending=True)

    f, ax = plt.subplots(1, 1, figsize=(9, 7))
    explanation_df.plot(kind='barh', ax=ax)
    plt.title('Permutation importances')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)

    if "feature" in explanation_df:
        _ = ax.set_yticklabels(explanation_df["feature"])


def plot_correlation_matrix(data):

    f, ax = plt.subplots(figsize=(5, 5))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
    square=True, ax=ax, annot=True)

