############################################################
##### Imports
############################################################

import matplotlib.pyplot as plt
import seaborn as sns

############################################################
##### Utility Functions
############################################################


def plot_distributions(dataset, features_categorical, ncols, nrows):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 4, nrows * 4))
    fig.suptitle("Distribution Dataset", fontsize=16)
    fig_col = 0
    fig_row = 0

    for col in dataset.columns:
        if col not in features_categorical:
            sns.histplot(data=dataset, x=col, bins=30, ax=axs[fig_row, fig_col])
        else:
            n_colors = dataset[col].nunique()
            sns.countplot(
                data=dataset,
                x=col,
                hue=col,
                palette=sns.color_palette(palette="Set2", n_colors=n_colors),
                ax=axs[fig_row, fig_col],
            )

        fig_col += 1
        if fig_col == 5:
            fig_col = 0
            fig_row += 1

    plt.subplots_adjust(top=0.95)
