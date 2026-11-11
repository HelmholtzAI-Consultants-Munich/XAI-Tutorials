############################################################
##### Imports
############################################################

import os
import math
import shap
import umap
import kmedoids
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy

############################################################
##### Utility Functions
############################################################


def plot_distributions(dataset, ncols):
    """
    Plots the distributions of all features in a dataset as histograms or countplots.

    If a feature is categorical or has fewer than 5 unique values, a countplot is used.
    Otherwise, a histogram is used. Subplots are arranged according to the number of columns specified.

    :param dataset: The dataset containing the features to plot.
    :type dataset: pandas.DataFrame
    :param ncols: Number of columns in the subplot grid layout.
    :type ncols: int
    """
    nrows = int(np.ceil(len(dataset.columns) / ncols))

    plt.figure(figsize=(ncols * 4.5, nrows * 4.5))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle("Distribution of features")

    for n, feature in enumerate(dataset.columns):
        # add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, n + 1)
        if dataset[feature].nunique() < 5 or isinstance(dataset[feature].dtype, pd.CategoricalDtype):
            sns.countplot(
                data=dataset,
                x=feature,
                hue=feature,
                palette="Blues_r",
                ax=ax,
            )
            # ax.legend(bbox_to_anchor=(1, 1), loc=2)
        else:
            sns.histplot(
                data=dataset,
                x=feature,
                bins=30,
                ax=ax,
                color="#3470a3",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_permutation_feature_importance(result, data, title, top_n=None, figsize=(7, 5)):
    """
    Plot permutation feature importances as a boxplot.

    :param result: Result of permutation importance containing importances.
    :type result: sklearn.inspection._permutation_importance.PermutationImportanceResult
    :param data: Dataset used for feature names.
    :type data: pandas.DataFrame
    :param title: Title of the plot.
    :type title: str
    :param top_n: Number of top features to display (optional).
    :type top_n: int
    :param figsize: Size of the figure.
    :type figsize: tuple
    """
    # Sort the features by importance mean
    perm_sorted_idx = result.importances_mean.argsort()[::-1]

    # If top_n is provided, limit the selection to top_n features
    if top_n:
        perm_sorted_idx = perm_sorted_idx[:top_n]

    # Prepare the data for Seaborn's boxplot (convert to long format)
    feature_importances = result.importances[perm_sorted_idx].T
    df = pd.DataFrame(feature_importances, columns=data.columns[perm_sorted_idx])
    df_long = df.melt(var_name="Feature", value_name="Importance")

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df_long, x="Importance", y="Feature", ax=ax, flierprops=dict(marker=".", alpha=0.5, markersize=2)
    )

    # Set title and layout
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_permutation_feature_importance_train_vs_test(
    result_train, data_train, result_test, data_test, title, figsize=(10, 4)
):
    """
    Compare permutation feature importances between train and test datasets.

    :param result_train: Permutation importance result for the training data.
    :type result_train: sklearn.inspection._permutation_importance.PermutationImportanceResult
    :param data_train: Training dataset.
    :type data_train: pandas.DataFrame
    :param result_test: Permutation importance result for the test data.
    :type result_test: sklearn.inspection._permutation_importance.PermutationImportanceResult
    :param data_test: Test dataset.
    :type data_test: pandas.DataFrame
    :param title: Title of the plot.
    :type title: str
    :param figsize: Size of the figure.
    :type figsize: tuple
    """
    perm_sorted_idx = result_train.importances_mean.argsort()
    perm_indices = np.arange(0, len(result_train.importances_mean)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)
    ax1.barh(
        perm_indices,
        result_train.importances_mean[perm_sorted_idx],
        height=0.7,
        color="#3470a3",  # color = 'cornflowerblue'
    )
    ax1.set_yticks(perm_indices)
    ax1.set_yticklabels(data_train.columns[perm_sorted_idx])
    ax1.set_ylim((0, len(result_train.importances_mean)))

    perm_sorted_idx = result_test.importances_mean.argsort()
    perm_indices = np.arange(0, len(result_test.importances_mean)) + 0.5

    ax2.barh(
        perm_indices,
        result_test.importances_mean[perm_sorted_idx],
        height=0.7,
        color="#3470a3",  # color = 'cornflowerblue'
    )
    ax2.set_yticks(perm_indices)
    ax2.set_yticklabels(data_test.columns[perm_sorted_idx])
    ax2.set_ylim((0, len(result_test.importances_mean)))

    fig.tight_layout()
    plt.show()


def plot_impurity_feature_importance(importance, names, title, top_n=None, figsize=(5, 4)):
    """
    Plot feature importance based on impurity decrease.

    :param importance: List of feature importances.
    :type importance: list
    :param names: List of feature names.
    :type names: list
    :param title: Title of the plot.
    :type title: str
    :param top_n: Number of top features to display (optional).
    :type top_n: int
    :param figsize: Size of the figure.
    :type figsize: tuple
    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order of decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    if top_n:
        fi_df = fi_df.iloc[:top_n]

    # Define size of bar plot
    fig, ax = plt.subplots(figsize=figsize)
    # Plot Seaborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"], color="#3470a3")
    # Add chart labels
    plt.title(title)
    plt.xlabel("Feature Importance (mean decrease in impurity)")
    plt.ylabel("Feature Names")


def plot_explanation(explanation):
    """
    Plot permutation explanation as a barplot.

    :param explanation: Dictionary containing explanation results.
    :type explanation: dict
    """
    explanation_df = pd.DataFrame({k: v for k, v in explanation.items() if k != "importances"}).sort_values(
        by="importances_mean", ascending=True
    )

    f, ax = plt.subplots(1, 1, figsize=(9, 7))
    explanation_df.plot(kind="barh", ax=ax)
    plt.title("Permutation importances")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)

    if "feature" in explanation_df:
        _ = ax.set_yticklabels(explanation_df["feature"])


def plot_correlation_matrix(data, figsize=(5, 5), annot=True, labelsize=10, shrink=1):
    """
    Plot the correlation matrix of the dataset.

    :param data: Dataset to calculate the correlation matrix.
    :type data: pandas.DataFrame
    :param figsize: Size of the figure.
    :type figsize: tuple
    :param annot: Whether to annotate the heatmap.
    :type annot: bool
    :param labelsize: Font size of labels.
    :type labelsize: int
    :param shrink: Shrink factor for the colorbar.
    :type shrink: float
    """
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)

    f, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    sns.heatmap(
        round(corr, 2),
        mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        cbar_kws={"shrink": shrink},
        square=True,
        ax=ax,
        annot=annot,
    )


def plot_dendrogram(linked, feature_names, figsize=(5, 5), leaf_font_size=10):
    """
    Plot a dendrogram based on hierarchical clustering.

    :param linked: The linkage matrix.
    :type linked: numpy.ndarray
    :param feature_names: List of feature names corresponding to leaves.
    :type feature_names: list
    :param figsize: Size of the figure.
    :type figsize: tuple
    :param leaf_font_size: Font size for the leaf labels.
    :type leaf_font_size: int
    """
    # Create a figure
    plt.figure(figsize=figsize)

    # Plot the dendrogram
    dendrogram = hierarchy.dendrogram(
        linked,
        orientation="top",  # 'top', 'bottom', 'left', or 'right'
        distance_sort="descending",  # 'ascending' or 'descending'
        show_leaf_counts=True,  # Show the number of observations in each cluster
        leaf_rotation=90,  # Rotation of leaf labels
        leaf_font_size=leaf_font_size,  # Font size of leaf labels
        show_contracted=True,  # Show contracted leaves
        labels=feature_names,
    )

    # Add title and labels
    plt.title("Dendrogram")
    plt.xlabel("Feature")
    plt.ylabel("Distance")

    # Show the plot
    plt.show()


###############################################################
##### Utility functions for AML Case Study
###############################################################

palette_dict = {
    "Condition": ["lightgrey", "grey"],
    "Tissue": ["darkseagreen", "darkgreen"],
    "GSE": sns.color_palette("tab20") + sns.color_palette("tab10"),
    "Disease": sns.color_palette("tab20b") + sns.color_palette("tab20c"),
    "Set": ["cornflowerblue", "darkblue"],
}

plt.rcParams.update({"font.size": 12})

dir_output = "results"


def plot_pie_charts(dataset, columns_to_plot=["Condition", "Tissue", "GSE", "Disease"], name=""):

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    for i, col in enumerate(columns_to_plot):
        sorted_labels = sorted(dataset[col].dropna().unique())  # sort for consistency
        counts = {label: (dataset[col] == label).sum() for label in sorted_labels}
        values = list(counts.values())
        labels = list(counts.keys())

        color_list = palette_dict[col][: len(labels)]

        if len(labels) <= 10:
            wedges, texts, autotext = axes[i].pie(
                values,
                labels=[None] * len(labels),
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops=dict(edgecolor="white"),
                colors=color_list,
            )
        else:
            wedges, texts = axes[i].pie(
                values,
                labels=[None] * len(labels),
                autopct=None,
                startangle=140,
                wedgeprops=dict(edgecolor="white"),
                colors=color_list,
            )

        axes[i].set_title(f"Distribution of {col}")
        axes[i].axis("equal")
        axes[i].legend(
            handles=wedges,
            labels=labels,
            title=col,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2 if len(labels) > 10 else 1,
            fontsize="small",
        )

    # plt.tight_layout()
    # plt.savefig(os.path.join(dir_output, f"{name}_pie_chart.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def plot_pca(dataset, columns_to_plot=["Condition", "Tissue", "GSE", "Disease"], name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    expression = dataset.drop(columns=metadata_cols, errors="ignore")
    metadata = dataset[columns_to_plot]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False)

    for i, col in enumerate(columns_to_plot):
        unique_classes = sorted(dataset[col].unique())  # sort for consistency
        palette = {cls: palette_dict[col][j] for j, cls in enumerate(unique_classes)}

        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=col,
            alpha=0.7,
            edgecolor=None,
            palette=palette,
            ax=axes[0, i],
        )
        axes[0, i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        axes[0, i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        axes[0, i].set_title(f"PCA colored by {col}")
        axes[0, i].get_legend().remove()
        # ax.legend(markerscale=1.5, fontsize="x-small", loc="best", frameon=True)

    plt.tight_layout()
    # plt.savefig(os.path.join(dir_output, f"{name}_pca.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def plot_umap(dataset, columns_to_plot=["Condition", "Tissue", "GSE", "Disease"], name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = dataset.drop(columns=metadata_cols)

    # Normalize the gene expression data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gene_expression)

    # Run UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    for i, col in enumerate(columns_to_plot):
        unique_classes = sorted(dataset[col].unique())  # sort for consistency
        palette = {cls: palette_dict[col][j] for j, cls in enumerate(unique_classes)}

        for cls in unique_classes:
            indices = dataset[col] == cls
            axes[i].scatter(
                embedding[indices, 0],
                embedding[indices, 1],
                label=str(cls),
                color=palette[cls],
                alpha=0.7,
                s=20,
            )

        axes[i].set_title(f"UMAP colored by {col}")
        axes[i].set_xlabel("UMAP-1")
        axes[i].set_ylabel("UMAP-2")
        # axes[i].legend(markerscale=1.5, fontsize="x-small", loc="best", frameon=True)

    plt.tight_layout()
    # plt.savefig(os.path.join(dir_output, f"{name}_umap.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def hierarchical_clustering_heatmap(dataset, name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = dataset.drop(columns=metadata_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gene_expression)
    X_scaled_df = pd.DataFrame(X_scaled, index=dataset.index, columns=gene_expression.columns)

    condition_values = dataset["Condition"].astype(str)
    disease_values = dataset["Disease"].astype(str)
    study_values = dataset["GSE"].astype(str)

    sorted_condition = sorted(condition_values.unique())
    sorted_disease = sorted(disease_values.unique())
    sorted_study = sorted(study_values.unique())

    condition_palette = palette_dict["Condition"][: len(sorted_condition)]
    disease_palette = palette_dict["Disease"][: len(sorted_disease)]
    study_palette = palette_dict["GSE"][: len(sorted_study)]

    condition_lut = dict(zip(sorted_condition, condition_palette))
    disease_lut = dict(zip(sorted_disease, disease_palette))
    study_lut = dict(zip(sorted_study, study_palette))

    row_colors = pd.DataFrame(
        {
            "Condition": condition_values.map(condition_lut),
            "Disease": disease_values.map(disease_lut),
            "GSE": study_values.map(study_lut),
        },
        index=dataset.index,
    )

    g = sns.clustermap(
        X_scaled_df,
        method="average",
        metric="euclidean",
        row_colors=row_colors,
        figsize=(16, 10),
        row_cluster=False,
        col_cluster=False,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=False,
    )

    # condition_patches = [Patch(color=condition_lut[c], label=c) for c in sorted_condition]
    # disease_patches = [Patch(color=disease_lut[d], label=d) for d in sorted_disease]

    # g.ax_heatmap.legend(
    #    handles=condition_patches + disease_patches,
    #    bbox_to_anchor=(1.05, 1),
    #    loc="upper left",
    #    fontsize="small",
    #    title="Annotations",
    # )

    plt.title("Hierarchical clustering on full dataset", y=1.05)
    # plt.savefig(
    #     os.path.join(dir_output, f"{name}_hierarchical_clustering.png"),
    #     format="png",
    #     bbox_inches="tight",
    #     dpi=800,
    # )
    plt.show()


def kmedoids_clustering_heatmap(dataset, k=2, name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = dataset.drop(columns=metadata_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(gene_expression)
    X_scaled_df = pd.DataFrame(X_scaled, index=dataset.index, columns=gene_expression.columns)

    clustering = kmedoids.KMedoids(n_clusters=k, random_state=42, method="pam", metric="euclidean")
    cluster_labels = clustering.fit_predict(X_scaled_df.to_numpy())
    dataset["cluster"] = cluster_labels

    sorted_idx = dataset.sort_values("cluster").index
    X_sorted = X_scaled_df.loc[sorted_idx]

    condition_values = dataset.loc[sorted_idx, "Condition"].astype(str)
    disease_values = dataset.loc[sorted_idx, "Disease"].astype(str)
    study_values = dataset.loc[sorted_idx, "GSE"].astype(str)

    sorted_condition = sorted(condition_values.unique())
    sorted_disease = sorted(disease_values.unique())
    sorted_study = sorted(study_values.unique())

    condition_palette = palette_dict["Condition"][: len(sorted_condition)]
    disease_palette = palette_dict["Disease"][: len(sorted_disease)]
    study_palette = palette_dict["GSE"][: len(sorted_study)]

    condition_lut = dict(zip(sorted_condition, condition_palette))
    disease_lut = dict(zip(sorted_disease, disease_palette))
    study_lut = dict(zip(sorted_study, study_palette))

    row_colors = pd.DataFrame(
        {
            "Condition": condition_values.map(condition_lut),
            "Disease": disease_values.map(disease_lut),
            "GSE": study_values.map(study_lut),
        },
        index=sorted_idx,
    )

    g = sns.clustermap(
        X_sorted,
        row_colors=row_colors,
        col_cluster=False,
        row_cluster=False,
        cmap="coolwarm",
        figsize=(16, 10),
        xticklabels=False,
        yticklabels=False,
    )

    condition_patches = [Patch(color=condition_lut[c], label=c) for c in sorted_condition]
    disease_patches = [Patch(color=disease_lut[d], label=d) for d in sorted_disease]
    g.ax_heatmap.legend(
        handles=condition_patches + disease_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
        title="Annotations",
    )

    cluster_sorted = dataset.loc[sorted_idx, "cluster"].values
    cluster_change_indices = [
        i for i in range(1, len(cluster_sorted)) if cluster_sorted[i] != cluster_sorted[i - 1]
    ]

    row_color_axes = g.ax_row_colors
    if not isinstance(row_color_axes, (list, np.ndarray)):
        row_color_axes = [row_color_axes]

    for i in cluster_change_indices:
        g.ax_heatmap.axhline(i, color="white", linewidth=2)
        for ax in row_color_axes:
            ax.axhline(i, color="white", linewidth=2)

    plt.title(f"K-Medoids clustering (k={k}) on full dataset", y=1.05)
    # plt.savefig(
    #     os.path.join(dir_output, f"{name}_kmedoids_clustering.png"),
    #     format="png",
    #     bbox_inches="tight",
    #     dpi=800,
    # )
    plt.show()

    return dataset


def plot_stacked_bar_chart(feature_importance, columns_to_plot=["Tissue", "GSE", "Disease"], name=""):

    data_clustering = feature_importance.data_clustering.copy()

    fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(28, 6))
    for i, feature in enumerate(columns_to_plot):
        sorted_categories = sorted(data_clustering[feature].dropna().unique())
        data_clustering[feature] = pd.Categorical(
            data_clustering[feature], categories=sorted_categories, ordered=True
        )

        counts = data_clustering.groupby(["cluster", feature], observed=True).size().unstack(fill_value=0)
        counts = counts[sorted_categories]  # Ensure consistent order
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100

        num_categories = len(sorted_categories)
        colors = palette_dict[feature][:num_categories]
        cmap = ListedColormap(colors)

        percentages.plot(kind="bar", stacked=True, ax=axes[i], width=0.6, colormap=cmap)

        axes[i].set_ylabel("Percentage")
        axes[i].set_title(f"Stacked Bar Chart by Cluster for {feature}")
        axes[i].get_legend().remove()
        axes[i].legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    # plt.savefig(os.path.join(dir_output, f"{name}_stacked_bar_chart.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def plot_correlation_distribution(data, name=""):
    metadata_cols = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue"]
    gene_expression = data.drop(columns=metadata_cols)

    corr_matrix = gene_expression.corr().values

    upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    plt.figure(figsize=(6, 4))
    plt.hist(upper, bins=50)
    plt.title("Distribution of Gene-Gene Correlations")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
    plt.show()


def plot_shap_dependence(explanation, num_cols=6):
    # 1. Compute mean absolute SHAP values
    mean_shap = np.abs(explanation.values).mean(axis=0)

    # 2. Sort features by descending mean SHAP
    sorted_indices = np.argsort(-mean_shap)
    sorted_features = [explanation.feature_names[i] for i in sorted_indices]

    # 3. Set up grid layout
    n_features = len(sorted_features)
    rows = math.ceil(n_features / num_cols)

    fig, axes = plt.subplots(rows, num_cols, figsize=(num_cols * 6, rows * 4))
    axes = axes.flatten()

    # 4. Generate SHAP dependence plots in the grid
    for i, feature in enumerate(sorted_features):
        shap.dependence_plot(
            ind=feature,
            shap_values=explanation.values,
            features=explanation.data,
            feature_names=explanation.feature_names,
            ax=axes[i],
            show=False,
        )
        axes[i].set_title(f"Dependence: {feature}", fontsize=10)

    # 5. Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()