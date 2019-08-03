from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

sns.set(color_codes=True)
MATPLOT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def create_cdict(meta_labels):
    labels = np.unique(meta_labels)
    return dict(zip(labels, MATPLOT_COLORS[: len(labels)]))


def plot_embedding2D(embedding, meta_labels, alpha=0.5):
    cdict = create_cdict(meta_labels)
    fig, ax = plt.subplots()
    for lab in labels:
        idx = np.where(meta_labels == lab)
        ax.scatter(embedding[idx, 0], embedding[idx, 1], label=lab, alpha=alpha)
    ax.legend()
    plt.show()


def plot_embedding3D(embedding, meta_labels, alpha=0.5):
    cdict = create_cdict(meta_labels)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for lab in labels:
        idx = np.where(meta_labels == lab)
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            embedding[idx, 2],
            label=lab,
            alpha=alpha,
        )
    ax.legend()
    plt.show()


def plot_scatter_matrix(df, color_field, x_y_prefix, tooltip_fields, size):
    repeated_facets = df.columns[df.columns.str.match(x_y_prefix)]
    scatter_matrix = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            alt.X(alt.repeat("column"), type="quantitative"),
            alt.Y(alt.repeat("row"), type="quantitative"),
            color=f"{color_field}",
            tooltip=tooltip_fields,
        )
        .properties(width=size, height=size)
        .repeat(row=list(repeated_facets), column=list(repeated_facets[::-1]))
    )
    return scatter_matrix


def sample_and_melt(file_name, cell_id, N):
    n_hidden, rep = get_file_info(file_name)
    df = pd.read_csv(file_name, index_col="inst_id").iloc[:, -(n_hidden + 1) :]
    subset = df[df.cell_id == cell_id].sample(N).reset_index()
    subset["inst_id"] = subset.cell_id + "_" + subset.index.astype(str)
    return pd.melt(
        subset,
        id_vars=["inst_id", "cell_id"],
        var_name="hidden_unit",
        value_name="activation",
    )


def plot_heatmap(file_name, N=5, width=None):
    data = pd.concat(
        sample_and_melt(file_name, cell_id, N) for cell_id in ["PC3", "MCF7", "VCAP"]
    )
    heatmap = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("hidden_unit:N", sort=list(range(16))),
            y=alt.Y("inst_id:O", sort=alt.EncodingSortField(field="cell_id")),
            color=alt.Color("activation:Q", scale=alt.Scale(scheme="inferno")),
        )
    )

    return heatmap


def plot_embedding(hidden_output_df, meta_data_df, color="cell_id", type_="pca"):
    activations = hidden_output_df.values
    if type_ == "pca":
        embedding = PCA(n_components=2).fit_transform(activations)

    elif type_ == "umap":
        embedding = UMAP().fit_transform(activations)

    elif type_ == "tsne":
        embedding = TSNE().fit_transform(activations)

    df = pd.DataFrame(
        data=embedding,
        columns=[f"{type_}_1", f"{type_}_2"],
        index=hidden_output_df.index,
    )

    df = df.join(meta_data_df)
    return (
        alt.Chart(df.reset_index())
        .mark_circle()
        .encode(x=f"{type_}_1:Q", y=f"{type_}_2:Q", color=color, tooltip=["inst_id"])
    )


def plot_clustermap(data, meta_data, meta_colname):
    meta_col = meta_data[meta_colname]
    cdict = create_cdict(meta_col)
    row_colors = meta_col.map(cdict)
    sns.clustermap(data, row_colors=row_colors, z_score=0)


def boxplot(df, x, y, extent):
    plot = alt.Chart(df).mark_boxplot(extent=extent).encode(y=y)
    if x is not None:
        plot = plot.encode(x=x)
    return plot


def scatter(df, x_field, y_field, color_field=None, tooltip_fields=None):
    scatter = alt.Chart(df).mark_circle().encode(x=x_field, y=y_field)
    if color_field:
        scatter = scatter.encode(color=color_field)
    if tooltip_fields:
        scatter = scatter.encode(tooltip=tooltip_fields)
    return scatter


def barplot(df, x, y, sort=None):
    return alt.Chart(df).mark_bar().encode(x=alt.X(x, sort=sort), y=y)


def heatmap(df, x_field, y_field, value_field, color_scheme, has_text):
    base = alt.Chart(df).encode(x=f"{x_field}:N", y=f"{y_field}:N")

    base += base.mark_rect().encode(value_field, scale=alt.Scale(scheme=color_scheme))
    if has_text:
        base += base.mark_text().encode(text=alt.Text(f"{value_field}:Q", format=".2"))
    return base


def get_file_info(file_path):
    return [int(s) for s in file_path.split("/")[-1].split(".")[0].split("_")]
