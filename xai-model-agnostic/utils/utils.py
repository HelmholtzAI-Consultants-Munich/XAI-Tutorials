

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

