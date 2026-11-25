############################################################
##### Imports
############################################################

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

############################################################
##### Setups
############################################################

BLUE, ORANGE = "#0072B2", "#E69F00"
PALETTE = [BLUE, ORANGE, "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#666666"]
CMAP = LinearSegmentedColormap.from_list("blue_orange", [BLUE, "#F7F7F7", "#D55E00"])
METADATA = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue", "Set", "cluster"]
DEFAULT_COLUMNS = ["Condition", "Tissue", "GSE", "Disease"]
palette_dict = {
    "Condition": ["lightgrey", "grey"],
    "Tissue": ["darkseagreen", "darkgreen"],
    "GSE": sns.color_palette("tab20") + sns.color_palette("tab10"),
    "Disease": sns.color_palette("tab20b") + sns.color_palette("tab20c"),
    "Set": ["cornflowerblue", "darkblue"],
}
dir_output = "results"  # Retained for existing notebooks; no automatic file saving.


def set_plot_style():
    """Set consistent notebook-wide fonts, colors, and export quality."""
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        font="sans-serif",
        rc={
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "normal",
            "figure.titlesize": 15,
            "figure.titleweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )


set_plot_style()

############################################################
##### Helper Functions
############################################################


def _finish(fig, axes, show, return_fig):
    if show:
        plt.show()
    if return_fig:
        return fig, axes


def _categories(series, column=None):
    labels = series.astype("string").fillna("(missing)")
    categories = sorted(labels.unique())
    palette = palette_dict.get(column, PALETTE)
    if len(categories) > len(palette):
        palette = sns.color_palette("husl", len(categories))
    return labels, dict(zip(categories, palette))


def _expression(dataset, extra_metadata=()):
    return dataset.drop(columns=METADATA + list(extra_metadata), errors="ignore")


def _subplot_grid(count, ncols, figsize):
    ncols = min(ncols, count)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (min(10, 4.5 * ncols), min(6.5, max(4, 3 * nrows))),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, axes


############################################################
##### Utility Functions
############################################################


def plot_distributions(dataset, ncols=7, *, show=True, return_fig=False, figsize=None):
    ncols = min(ncols, dataset.shape[1])
    nrows = int(np.ceil(dataset.shape[1] / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (3 * ncols, 2.7 * nrows), squeeze=False, layout="constrained"
    )
    fig.suptitle("Distribution of features")
    for ax, feature in zip(axes.flat, dataset.columns):
        values = dataset[feature]
        if not pd.api.types.is_numeric_dtype(values) or values.nunique() < 5:
            labels, _ = _categories(values)
            counts = labels.value_counts(sort=False)
            ax.bar(counts.index, counts, color=BLUE, alpha=0.8)
        else:
            ax.hist(
                values.replace([np.inf, -np.inf], np.nan).dropna(),
                bins=30,
                color=BLUE,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set(xlabel=str(feature), ylabel="Count")
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[dataset.shape[1] :]:
        ax.set_visible(False)
    return _finish(fig, axes, show, return_fig)


def plot_permutation_feature_importance(
    result,
    data,
    title,
    top_n=None,
    figsize=None,
    *,
    xlabel="Permutation importance (decrease in score)",
    show=True,
    return_fig=False,
):
    values = np.asarray(result["importances"])
    order = np.argsort(-values.mean(axis=1), kind="stable")[:top_n]
    fig, ax = plt.subplots(
        figsize=figsize or (8, min(6.5, max(4, len(order) * 0.3 + 1.5))), layout="constrained"
    )
    frame = pd.DataFrame(values[order].T, columns=data.columns[order])
    sns.boxplot(data=frame, orient="h", color=BLUE, width=0.6, fliersize=2, ax=ax)
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel="")
    ax.grid(axis="y", visible=False)
    return _finish(fig, ax, show, return_fig)


def plot_permutation_feature_importance_train_vs_test(
    result_train,
    data_train,
    result_test,
    data_test,
    title,
    figsize=None,
    *,
    top_n=None,
    xlabel="Permutation importance (decrease in score)",
    show=True,
    return_fig=False,
):
    train = pd.DataFrame(result_train["importances"], index=data_train.columns)
    test = pd.DataFrame(result_test["importances"], index=data_test.columns)
    if set(train.index) != set(test.index):
        raise ValueError("Train and test results must contain the same features.")
    names = train.mean(axis=1).sort_values(ascending=False, kind="stable").index[:top_n]
    fig, axes = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
        layout="constrained",
        figsize=figsize or (10, min(6.5, max(4, len(names) * 0.3 + 1.5))),
    )
    fig.suptitle(title)
    for ax, frame, label, color in zip(axes, [train, test], ["Training data", "Test data"], [BLUE, ORANGE]):
        values = frame.loc[names].to_numpy()
        ax.barh(
            np.arange(len(names)),
            values.mean(axis=1),
            xerr=values.std(axis=1),
            color=color,
            height=0.65,
            error_kw={"ecolor": "0.25", "capsize": 2},
        )
        ax.set_yticks(np.arange(len(names)), labels=names)
        ax.axvline(0, color="0.4", ls="--", lw=1)
        ax.set(title=label, xlabel=xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    return _finish(fig, axes, show, return_fig)


def plot_impurity_feature_importance(
    importance, names, title, top_n=None, figsize=None, *, show=True, return_fig=False
):
    values = pd.Series(importance, index=names).sort_values(ascending=False, kind="stable").iloc[:top_n]
    fig, ax = plt.subplots(
        figsize=figsize or (8, min(6.5, max(4, len(values) * 0.3 + 1.5))), layout="constrained"
    )
    ax.barh(np.arange(len(values)), values, color=BLUE, height=0.65)
    ax.set_yticks(np.arange(len(values)), labels=values.index)
    ax.invert_yaxis()
    ax.set(title=title, xlabel="Mean decrease in impurity", ylabel="")
    ax.grid(axis="y", visible=False)
    return _finish(fig, ax, show, return_fig)


def plot_explanation(explanation, *, figsize=(8, 5), show=True, return_fig=False):
    means = np.asarray(explanation["importances_mean"])
    std = np.asarray(explanation.get("importances_std", np.zeros_like(means)))
    names = np.asarray(explanation.get("feature", [f"Feature {i}" for i in range(len(means))]))
    order = np.argsort(-means, kind="stable")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.barh(
        np.arange(len(order)),
        means[order],
        xerr=std[order],
        color=BLUE,
        height=0.65,
        error_kw={"ecolor": "0.25", "capsize": 2},
    )
    ax.set_yticks(np.arange(len(order)), labels=names[order])
    ax.invert_yaxis()
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set(title="Permutation importances", xlabel="Permutation importance (decrease in score)")
    ax.grid(axis="y", visible=False)
    return _finish(fig, ax, show, return_fig)


def plot_correlation_matrix(
    data, figsize=None, annot=True, labelsize=10, shrink=0.75, *, show=True, return_fig=False
):
    corr = data.select_dtypes(include="number").corr()
    size = min(9.5, max(7, 0.4 * len(corr) + 4))
    labels = [str(col).replace("_", " ") for col in corr.columns]
    fig, ax = plt.subplots(figsize=figsize or (size, min(6.5, size - 1)), layout="constrained")
    ax.grid(False)
    sns.heatmap(
        corr,
        mask=np.triu(np.ones_like(corr, dtype=bool), k=1),
        cmap=CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annot,
        fmt=".2f",
        annot_kws={"fontsize": 9},
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        cbar_kws={
            "shrink": shrink,
            "aspect": 25,
            "pad": 0.03,
            "ticks": [-1, -0.5, 0, 0.5, 1],
            "label": "Pearson correlation",
        },
        ax=ax,
    )
    ax.tick_params(labelsize=labelsize, length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_title("Feature correlations", pad=16)
    return _finish(fig, ax, show, return_fig)


def plot_dendrogram(
    linked, feature_names, figsize=(8, 4.5), leaf_font_size=10, *, show=True, return_fig=False, cutoff=None
):

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    hierarchy.dendrogram(
        linked,
        labels=list(feature_names),
        distance_sort="descending",
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        color_threshold=cutoff,
        ax=ax,
    )
    ax.set(title="Dendrogram", xlabel="Feature", ylabel="Distance")
    if cutoff is not None:
        ax.axhline(cutoff, color=ORANGE, ls="--", lw=1.5, label=f"Cutoff = {cutoff:g}")
        ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, show, return_fig)


def plot_pie_charts(dataset, columns_to_plot=None, name="", *, show=True, return_fig=False, figsize=None):
    columns = DEFAULT_COLUMNS if columns_to_plot is None else list(columns_to_plot)
    legend_rows = max(
        int(
            np.ceil(
                dataset[col].nunique(dropna=False) / (2 if dataset[col].nunique(dropna=False) > 10 else 1)
            )
        )
        for col in columns
    )
    legend_height = 0.23 * legend_rows + 1.2
    fig, axes = plt.subplots(
        1, len(columns), figsize=figsize or (8 * len(columns), 5.5 + legend_height), squeeze=False
    )
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.94, bottom=legend_height / fig.get_figheight(), wspace=0.2
    )
    for ax, col in zip(axes.flat, columns):
        labels, colors = _categories(dataset[col], col)
        counts = labels.value_counts().reindex(colors)
        wedges, *_ = ax.pie(
            counts,
            colors=list(colors.values()),
            startangle=140,
            autopct="%1.1f%%" if len(counts) <= 10 else None,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        ax.set_title(f"Distribution of {col}")
        ax.legend(
            wedges,
            list(colors),
            title=col,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=2 if len(counts) > 10 else 1,
            frameon=True,
        )
    return _finish(fig, axes, show, return_fig)


def _plot_embedding(embedding, dataset, columns, method, labels, name, show, return_fig, figsize):
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=figsize or (7 * len(columns), 6),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, col in zip(axes.flat, columns):
        categories, colors = _categories(dataset[col], col)
        for label, color in colors.items():
            selected = categories.eq(label).to_numpy(dtype=bool)
            ax.scatter(
                embedding[selected, 0],
                embedding[selected, 1],
                color=color,
                s=20,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
            )
        ax.set(title=f"{method} colored by {col}", xlabel=labels[0], ylabel=labels[1])
        ax.tick_params(labelleft=True)
        ax.grid(False)
    # As in the original overview, colors are explained by the pie-chart legends.
    return _finish(fig, axes, show, return_fig)


def plot_pca(dataset, columns_to_plot=None, name="", *, show=True, return_fig=False, figsize=None):
    columns = DEFAULT_COLUMNS if columns_to_plot is None else list(columns_to_plot)
    scaled = StandardScaler().fit_transform(_expression(dataset, columns))
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(scaled)
    labels = [f"PC{i + 1} ({ratio:.1%})" for i, ratio in enumerate(pca.explained_variance_ratio_)]
    return _plot_embedding(embedding, dataset, columns, "PCA", labels, name, show, return_fig, figsize)


def plot_umap(
    dataset, columns_to_plot=None, name="", *, show=True, return_fig=False, random_state=42, figsize=None
):
    import umap

    columns = DEFAULT_COLUMNS if columns_to_plot is None else list(columns_to_plot)
    scaled = StandardScaler().fit_transform(_expression(dataset, columns))
    embedding = umap.UMAP(
        n_neighbors=min(15, len(dataset) - 1), min_dist=0.1, metric="euclidean", random_state=random_state
    ).fit_transform(scaled)
    return _plot_embedding(
        embedding, dataset, columns, "UMAP", ["UMAP 1", "UMAP 2"], name, show, return_fig, figsize
    )


def _expression_heatmap(values, dataset, title, row_cluster=False, col_cluster=False, figsize=(16, 10)):
    row_colors = {}
    for col in ["Condition", "Disease", "GSE"]:
        if col in dataset:
            labels, colors = _categories(dataset[col], col)
            row_colors[col] = labels.map(colors)
    g = sns.clustermap(
        values,
        row_colors=pd.DataFrame(row_colors) if row_colors else None,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        method="average",
        cmap="coolwarm",
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Standardized expression"},
    )
    # Preserve the original data-driven color limits: no clipping at +/-3.
    g.fig.suptitle(title, y=0.99)
    g.ax_heatmap.set(xlabel="", ylabel="")
    return g


def hierarchical_clustering_heatmap(
    dataset, name="", *, row_cluster=False, col_cluster=False, show=True, return_fig=False, figsize=(16, 10)
):
    expression = _expression(dataset)
    values = pd.DataFrame(
        StandardScaler().fit_transform(expression), index=dataset.index, columns=expression.columns
    )
    title = "Hierarchically clustered expression" if row_cluster or col_cluster else "Standardized expression"
    g = _expression_heatmap(
        values, dataset, f"{title}: {name}" if name else title, row_cluster, col_cluster, figsize
    )
    if show:
        plt.show()
    if return_fig:
        return g


def kmedoids_clustering_heatmap(dataset, k=2, name="", *, show=True, return_fig=False, figsize=(16, 10)):
    import kmedoids

    expression = _expression(dataset)
    scaled = StandardScaler().fit_transform(expression)
    model = kmedoids.KMedoids(n_clusters=k, random_state=42, method="pam", metric="euclidean")
    result = dataset.copy()
    result["cluster"] = model.fit_predict(scaled)
    order = np.argsort(result["cluster"].to_numpy(), kind="stable")
    ordered = result.iloc[order].reset_index(drop=True)
    values = pd.DataFrame(scaled[order], columns=expression.columns)
    title = f"K-medoids clustering (k={k})"
    g = _expression_heatmap(values, ordered, f"{title}: {name}" if name else title, figsize=figsize)
    for boundary in np.flatnonzero(np.diff(ordered["cluster"])) + 1:
        g.ax_heatmap.axhline(boundary, color="white", lw=2)
        if g.ax_row_colors is not None:
            g.ax_row_colors.axhline(boundary, color="white", lw=2)
    if show:
        plt.show()
    return (result, g) if return_fig else result


def plot_stacked_bar_chart(
    feature_importance, columns_to_plot=None, name="", *, show=True, return_fig=False, figsize=None
):
    """One-row cluster-composition overview with category legends underneath."""
    dataset = (
        feature_importance
        if isinstance(feature_importance, pd.DataFrame)
        else feature_importance.data_clustering
    )
    columns = ["Tissue", "GSE", "Disease"] if columns_to_plot is None else list(columns_to_plot)
    legend_rows = max(
        int(
            np.ceil(
                dataset[col].nunique(dropna=False) / (2 if dataset[col].nunique(dropna=False) > 10 else 1)
            )
        )
        for col in columns
    )
    legend_height = 0.23 * legend_rows + 1.2
    fig, axes = plt.subplots(
        1, len(columns), figsize=figsize or (8 * len(columns), 5 + legend_height), squeeze=False
    )
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.92, bottom=legend_height / fig.get_figheight(), wspace=0.3
    )
    for ax, col in zip(axes.flat, columns):
        labels, colors = _categories(dataset[col], col)
        counts = pd.crosstab(dataset["cluster"], labels).reindex(columns=colors, fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        percentages.plot.bar(stacked=True, color=list(colors.values()), width=0.65, ax=ax, legend=False)
        ax.set(title=f"Cluster composition by {col}", xlabel="Cluster", ylabel="Samples (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(
            title=col,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2 if len(colors) > 10 else 1,
            frameon=True,
        )
    return _finish(fig, axes, show, return_fig)


def plot_correlation_distribution(data, name="", *, show=True, return_fig=False, figsize=(7, 4)):
    corr = _expression(data).corr().to_numpy()
    values = corr[np.triu_indices_from(corr, k=1)]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.hist(
        values[np.isfinite(values)], bins=np.linspace(-1, 1, 51), color=BLUE, edgecolor="white", linewidth=0.4
    )
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set(
        title=f"Feature-pair correlations: {name}" if name else "Feature-pair correlations",
        xlabel="Pearson correlation",
        ylabel="Number of feature pairs",
        xlim=(-1, 1),
    )
    return _finish(fig, ax, show, return_fig)


def plot_shap_dependence(
    explanation, num_cols=5, *, show=True, return_fig=False, figsize=None, interaction_index="auto"
):
    import shap

    values, features = np.asarray(explanation.values), np.asarray(explanation.data)
    if values.ndim != 2 or values.shape != features.shape:
        raise ValueError("Select one output: SHAP values and feature data must be matching 2D arrays.")
    names = explanation.feature_names
    if names is None:
        names = [f"Feature {i}" for i in range(values.shape[1])]
    order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")
    columns = min(num_cols, len(order))
    rows = int(np.ceil(len(order) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=figsize or (6 * columns, 4 * rows), squeeze=False, layout="constrained"
    )
    for ax, i in zip(axes.flat, order):
        shap.dependence_plot(
            int(i),
            values,
            features,
            feature_names=list(names),
            interaction_index=interaction_index,
            ax=ax,
            show=False,
        )
        ax.set_title(f"Dependence: {names[i]}", fontsize=13, fontweight="normal")
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.tick_params(labelsize=10)
        ax.grid(False)
    for ax in axes.flat[len(order) :]:
        ax.set_visible(False)
    return _finish(fig, axes, show, return_fig)


def plot_lime_coefficients(coef_df, *, figsize=None, show=True, return_fig=False):

    if isinstance(coef_df, pd.DataFrame):
        if coef_df.shape[1] != 1:
            raise ValueError("coef_df must be a Series or a one-column DataFrame.")
        coefficients = coef_df.iloc[:, 0]
    elif isinstance(coef_df, pd.Series):
        coefficients = coef_df
    else:
        raise ValueError("coef_df must be a Series or a one-column DataFrame.")
    if not pd.api.types.is_numeric_dtype(coefficients):
        raise TypeError("LIME coefficients must be numeric.")
    coefficients = coefficients.sort_values(key=lambda values: values.abs(), ascending=False, kind="stable")

    fig, ax = plt.subplots(
        figsize=figsize or (8, min(6.5, max(2.6, len(coefficients) * 0.28 + 1.3))),
        layout="constrained",
    )
    y = np.arange(len(coefficients))
    colors = [BLUE if value >= 0 else ORANGE for value in coefficients]
    ax.barh(y, coefficients.to_numpy(), color=colors, height=0.82, zorder=3)
    ax.set_yticks(y, labels=coefficients.index)
    ax.invert_yaxis()
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set(title="LIME: local model coefficients", xlabel="Model coefficient", ylabel="")
    ax.grid(axis="y", visible=False)
    return _finish(fig, ax, show, return_fig)
