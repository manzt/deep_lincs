from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py

# L1000 Data
N_LANDMARK_GENES = 978
N_SAMPLES = 1_319_138

# HDF5 nodes
RID_NODE = "/0/META/ROW/id"
CID_NODE = "/0/META/COL/id"
DATA_NODE = "/0/DATA/0/matrix"
ROW_META_GROUP_NODE = "/0/META/ROW"
COL_META_GROUP_NODE = "/0/META/COL"


def load_data(data_path, col_meta_path, pert_types, cell_ids=None, only_landmark=True):
    ridx_max = N_LANDMARK_GENES if only_landmark else None  # only select landmark genes
    subset_metadata = subset_samples(col_meta_path, pert_types, cell_ids)
    with h5py.File(data_path, "r") as gctx_file:
        # read in metadata columns
        all_sample_ids = gctx_file[CID_NODE][:].astype(str)
        gene_labels = gctx_file[RID_NODE][:ridx_max].astype(
            str
        )  # first 978 are landmark

        # read in dataset
        data_dset = gctx_file[DATA_NODE]
        data = da.from_array(data_dset)

        # create mask for desired ids
        mask = np.isin(all_sample_ids, subset_metadata.inst_id.values.astype(str))
        data = dd.from_dask_array(data[mask, :ridx_max], columns=gene_labels).compute()
        data["inst_id"] = all_sample_ids[mask]
        data = data.set_index("inst_id")

    col_meta = pd.DataFrame({"inst_id": all_sample_ids[mask]}).join(
        subset_metadata.set_index("inst_id"), on="inst_id"
    )
    return col_meta, gene_labels, data


def subset_samples(col_meta_path, pert_types, cell_ids):
    metadata = pd.read_csv(col_meta_path, sep="\t", low_memory=False)
    # filter metadata by pert types and cell ids

    if cell_ids is None:
        filtered_metadata = metadata[metadata.pert_type.isin(pert_types)]
    else:
        filtered_metadata = metadata[
            metadata.pert_type.isin(pert_types) & metadata.cell_id.isin(cell_ids)
        ]
    return filtered_metadata


def split_data(X, y, p1=0.2, p2=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=p2)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_tf_dataset(X, y, shuffle=True, repeated=True, batch_size=32):
    """Creates a tensorflow Dataset to be ingested by Keras."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if repeated:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def compute_summary_stats(data_path):
    with h5py.File(data_path) as f:
        data_dset = f[DATA_NODE]
        arr = da.from_array(data_dset)[:, :N_LANDMARK_GENES]
        ddf = dd.from_dask_array(arr)

        landmark_gene_labels = f[RID_NODE][:N_LANDMARK_GENES].astype(str)
        ddf.columns = landmark_gene_labels
        df = ddf.describe().compute()

    return df


def clip_and_normalize(data, n_sigma=2.0):
    data_mean = data.mean()
    data_std = data.std()

    upper = data_mean + n_sigma * data_std
    lower = data_mean - n_sigma * data_std

    clipped = np.where(data > upper, upper, data)
    clipped = np.where(data < lower, lower, clipped)

    return clipped


def plot_embedding2D(embedding, meta_labels, alpha=0.5):
    colors = [
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
    labels = np.unique(meta_labels)
    cdict = dict(zip(labels, colors[: len(labels)]))

    fig, ax = plt.subplots()
    for lab in labels:
        idx = np.where(meta_labels == lab)
        ax.scatter(embedding[idx, 0], embedding[idx, 1], label=lab, alpha=alpha)
    ax.legend()
    plt.show()


def plot_embedding3D(embedding, meta_labels, alpha=0.5):
    colors = [
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
    labels = np.unique(meta_labels)
    cdict = dict(zip(labels, colors[: len(labels)]))

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
