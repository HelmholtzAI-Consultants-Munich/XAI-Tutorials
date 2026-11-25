"""Plotting helpers for the XAI notebooks.

All existing function names and positional arguments are retained.
Figures are returned only with return_fig=True. Use show=False to suppress display.
K-medoids still returns its labeled DataFrame by default.
Plot defaults are set once on import; edit set_plot_style() to change the look.
Default canvases stay within 1000 x 650 pixels for notebook screen sharing.
For very large feature sets, use top_n or select a subset to keep labels readable.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BLUE, ORANGE = "#0072B2", "#E69F00"
PALETTE = [BLUE, ORANGE, "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#666666"]
CMAP = LinearSegmentedColormap.from_list("blue_orange", [BLUE, "#F7F7F7", "#D55E00"])
METADATA = ["sample_id", "Dataset", "GSE", "Condition", "Disease", "Tissue", "Set", "cluster"]
DEFAULT_COLUMNS = ["Condition", "Tissue", "GSE", "Disease"]
palette_dict = {column: PALETTE for column in DEFAULT_COLUMNS + ["Set"]}
dir_output = "results"  # Retained for existing notebooks; no automatic file saving.


def set_plot_style():
    """Set consistent notebook-wide fonts, colors, and export quality."""
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        font="DejaVu Sans",
        rc={
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "figure.titlesize": 15,
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


def _finish(fig, axes, show, return_fig):
    if show:
        plt.show()
    if return_fig:
        return fig, axes


def _categories(series):
    labels = series.astype("string").fillna("(missing)")
    categories = sorted(labels.unique())
    colors = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(categories)}
    return labels, colors


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


def plot_distributions(dataset, ncols, *, show=True, return_fig=False, figsize=None):
    """Plot numeric distributions and categorical counts."""
    fig, axes = _subplot_grid(dataset.shape[1], min(ncols, 3), figsize)
    fig.suptitle("Distribution of features")
    for ax, feature in zip(axes.flat, dataset.columns):
        values = dataset[feature]
        if not pd.api.types.is_numeric_dtype(values) or values.nunique() < 5:
            labels, _ = _categories(values)
            labels.value_counts().sort_index().plot.barh(ax=ax, color=BLUE)
            ax.set(xlabel="Count", ylabel="")
            ax.invert_yaxis()
        else:
            ax.hist(
                values.replace([np.inf, -np.inf], np.nan).dropna(),
                bins=30,
                color=BLUE,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.set(xlabel=str(feature), ylabel="Count")
        ax.set_title(str(feature))
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
    """Boxplots of permutation repetitions, ranked by mean importance."""
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
    """Compare means ± shuffle SD, with matching feature order and scales.

    Both results must use the same scoring metric. Error bars are not
    confidence intervals.
    """
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
    """Plot impurity-based importance, highest first."""
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


def plot_lime_coefficients(coef_df, *, figsize=None, show=True, return_fig=False):
    """Plot signed local-model coefficients from a LIME explanation.

    ``coef_df`` may be a Series or a one-column DataFrame indexed by feature.
    Features are ordered by absolute coefficient size, largest first.
    """
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


def plot_explanation(explanation, *, figsize=(8, 5), show=True, return_fig=False):
    """Plot permutation means with shuffle SD as error bars."""
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
    """Pearson correlations on a fixed -1 to +1 scale."""
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
    linked, feature_names, figsize=(8, 4.5), leaf_font_size=10, *, show=True, return_fig=False
):
    """Plot a feature linkage matrix."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    hierarchy.dendrogram(
        linked,
        labels=list(feature_names),
        distance_sort="descending",
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        link_color_func=lambda _: BLUE,
        ax=ax,
    )
    ax.set(title="Feature clustering", xlabel="Feature", ylabel="Linkage distance")
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, show, return_fig)


def plot_pie_charts(dataset, columns_to_plot=None, name="", *, show=True, return_fig=False, figsize=None):
    """Plot category proportions with counts in the legends."""
    columns = DEFAULT_COLUMNS if columns_to_plot is None else list(columns_to_plot)
    fig, axes = _subplot_grid(len(columns), 2, figsize)
    for ax, col in zip(axes.flat, columns):
        labels, colors = _categories(dataset[col])
        counts = labels.value_counts().reindex(colors)
        wedges, *_ = ax.pie(
            counts,
            colors=list(colors.values()),
            startangle=90,
            autopct="%1.0f%%" if len(counts) <= 6 else None,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title(f"Distribution of {col}")
        ax.legend(
            wedges,
            [f"{label} (n={count})" for label, count in counts.items()],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2 if len(counts) > 6 else 1,
            frameon=False,
        )
    return _finish(fig, axes, show, return_fig)


def _plot_embedding(embedding, dataset, columns, method, labels, name, show, return_fig, figsize):
    fig, axes = _subplot_grid(len(columns), 2, figsize)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for ax, col in zip(axes.flat, columns):
        categories, colors = _categories(dataset[col])
        for i, (label, color) in enumerate(colors.items()):
            selected = categories.eq(label).to_numpy(dtype=bool)
            ax.scatter(
                embedding[selected, 0],
                embedding[selected, 1],
                label=label,
                color=color,
                marker=markers[i % len(markers)],
                s=25,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.2,
                rasterized=True,
            )
        ax.set(title=f"{method} by {col}", xlabel=labels[0], ylabel=labels[1])
        ax.legend(title=col, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    if name:
        fig.suptitle(name)
    return _finish(fig, axes, show, return_fig)


def plot_pca(dataset, columns_to_plot=None, name="", *, show=True, return_fig=False, figsize=None):
    """Standardize expression features and plot the first two principal components."""
    columns = DEFAULT_COLUMNS if columns_to_plot is None else list(columns_to_plot)
    scaled = StandardScaler().fit_transform(_expression(dataset, columns))
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(scaled)
    labels = [f"PC{i + 1} ({ratio:.1%})" for i, ratio in enumerate(pca.explained_variance_ratio_)]
    return _plot_embedding(embedding, dataset, columns, "PCA", labels, name, show, return_fig, figsize)


def plot_umap(
    dataset, columns_to_plot=None, name="", *, show=True, return_fig=False, random_state=42, figsize=None
):
    """Plot a UMAP embedding; requires umap-learn."""
    import umap

    columns = DEFAULT_COLUMNS if columns_to_plot is None else list(columns_to_plot)
    scaled = StandardScaler().fit_transform(_expression(dataset, columns))
    embedding = umap.UMAP(
        n_neighbors=min(15, len(dataset) - 1), min_dist=0.1, metric="euclidean", random_state=random_state
    ).fit_transform(scaled)
    return _plot_embedding(
        embedding, dataset, columns, "UMAP", ["UMAP 1", "UMAP 2"], name, show, return_fig, figsize
    )


def _expression_heatmap(values, dataset, title, row_cluster=False, col_cluster=False, figsize=(10, 6.5)):
    row_colors, handles = {}, []
    for col in ["Condition", "Disease", "GSE"]:
        if col in dataset:
            labels, colors = _categories(dataset[col])
            row_colors[col] = labels.map(colors)
            handles.extend(Patch(color=color, label=f"{col}: {label}") for label, color in colors.items())
    g = sns.clustermap(
        values,
        row_colors=pd.DataFrame(row_colors) if row_colors else None,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        method="average",
        cmap=CMAP,
        center=0,
        vmin=-3,
        vmax=3,
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.16 if row_cluster else 0.02, 0.16 if col_cluster else 0.02),
        cbar_kws={"label": "Expression (z-score)", "orientation": "horizontal", "extend": "both"},
    )
    g.fig.suptitle(title, y=0.99)
    g.fig.subplots_adjust(right=0.70, top=0.93)
    g.cax.set_position([0.76, 0.86, 0.16, 0.018])
    g.ax_heatmap.set(xlabel="Expression feature", ylabel="Sample")
    if handles:
        g.fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.74, 0.70),
            frameon=False,
            ncol=max(1, int(np.ceil(len(handles) / 22))),
        )
    return g


def hierarchical_clustering_heatmap(
    dataset, name="", *, row_cluster=False, col_cluster=False, show=True, return_fig=False, figsize=(10, 6.5)
):
    """Expression heatmap; return its ClusterGrid only when return_fig=True."""
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


def kmedoids_clustering_heatmap(dataset, k=2, name="", *, show=True, return_fig=False, figsize=(10, 6.5)):
    """Return labeled data; with return_fig=True return (data, ClusterGrid)."""
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
    """Plot metadata proportions within clusters, with category legends."""
    dataset = (
        feature_importance
        if isinstance(feature_importance, pd.DataFrame)
        else feature_importance.data_clustering
    )
    columns = ["Tissue", "GSE", "Disease"] if columns_to_plot is None else list(columns_to_plot)
    fig, axes = _subplot_grid(len(columns), 2, figsize)
    for ax, col in zip(axes.flat, columns):
        labels, colors = _categories(dataset[col])
        counts = pd.crosstab(dataset["cluster"], labels).reindex(columns=colors, fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        percentages.plot.bar(stacked=True, color=list(colors.values()), width=0.65, ax=ax)
        ax.set(title=f"Cluster composition by {col}", xlabel="Cluster", ylabel="Samples (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title=col, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    if name:
        fig.suptitle(name)
    return _finish(fig, axes, show, return_fig)


def plot_correlation_distribution(data, name="", *, show=True, return_fig=False, figsize=(7, 4)):
    """Plot unique feature-pair correlations, excluding the diagonal."""
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


def plot_shap_dependence(explanation, num_cols=3, *, show=True, return_fig=False, figsize=None):
    """Plot feature values against SHAP values; select one output for multiclass."""
    values, features = np.asarray(explanation.values), np.asarray(explanation.data)
    if values.ndim != 2 or values.shape != features.shape:
        raise ValueError("Select one output: SHAP values and feature data must be matching 2D arrays.")
    names = explanation.feature_names
    if names is None:
        names = [f"Feature {i}" for i in range(values.shape[1])]
    order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")
    fig, axes = _subplot_grid(len(order), min(num_cols, 3), figsize)
    for ax, i in zip(axes.flat, order):
        x = pd.Series(features[:, i])
        try:
            x = pd.to_numeric(x, errors="raise")
        except (ValueError, TypeError):
            labels, colors = _categories(x)
            x = pd.Categorical(labels, categories=list(colors)).codes
            ax.set_xticks(range(len(colors)), labels=list(colors), rotation=45, ha="right")
        ax.scatter(x, values[:, i], color=BLUE, s=18, alpha=0.6, edgecolors="none", rasterized=True)
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.set(title=str(names[i]), xlabel=str(names[i]), ylabel="SHAP value")
    fig.suptitle("SHAP dependence (ranked by mean absolute SHAP value)")
    return _finish(fig, axes, show, return_fig)
