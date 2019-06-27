from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .utils import get_file_info

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


def create_cdict(metadata_labels):
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


def plot_scatter_matrix(file_path, size=75):
    n_units, rep = [
        int(val) for val in file_path.split("/")[-1].split(".")[0].split("_")
    ]
    data = pd.read_csv(file_path)
    scatter_matrix = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            alt.X(alt.repeat("column"), type="quantitative"),
            alt.Y(alt.repeat("row"), type="quantitative"),
            color="cell_id:N",
            tooltip=["inst_id", "pert_id", "pert_iname"],
        )
        .properties(width=size, height=size)
        .repeat(
            row=list(data.columns.values[-n_units::]),
            column=list(data.columns.values[-n_units::][::-1]),
        )
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


def plot_embedding(file_path, type_="pca"):
    n_hidden, rep = get_file_info(file_path)
    data = pd.read_csv(file_path, na_values="-666", index_col="inst_id")
    activations = data.iloc[:, -n_hidden:].values

    if type_ == "pca":
        embedding = PCA(n_components=2).fit_transform(activations)

    elif type_ == "umap":
        embedding = UMAP().fit_transform(activations)

    elif type_ == "tsne":
        embedding = TSNE().fit_transform(activations)

    df = pd.DataFrame(
        data=embedding, columns=[f"{type_}_1", f"{type_}_2"], index=data.index
    )
    df["cell_id"] = data.cell_id.values

    return (
        alt.Chart(df.reset_index())
        .mark_circle()
        .encode(x=f"{type_}_1", y=f"{type_}_2", color="cell_id", tooltip=["inst_id"])
        .properties(title=f"hidden units: {n_hidden}, rep: {rep}")
    )


def plot_clustermap(file_path):
    n_hidden, rep = get_file_info(file_path)
    data = pd.read_csv(file_path, na_values="-666", index_col="inst_id")
    units = data.iloc[:, -n_hidden:]

    cell_ids = data.pop("cell_id")
    lut = dict(zip(cell_ids.unique(), "rbg"))
    row_colors = cell_ids.map(lut)
    sns.clustermap(units, row_colors=row_colors)


def get_file_info(file_path):
    return [int(s) for s in file_path.split("/")[-1].split(".")[0].split("_")]
