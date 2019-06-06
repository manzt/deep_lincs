from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
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
        data = data[mask, :ridx_max].compute()

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


def compute_summary_stats(data_path):
    with h5py.File(data_path) as f:
        data_dset = f[DATA_NODE]
        arr = da.from_array(data_dset)[:, :N_LANDMARK_GENES]
        ddf = dd.from_dask_array(arr)

        landmark_gene_labels = f[RID_NODE][:N_LANDMARK_GENES].astype(str)
        ddf.columns = landmark_gene_labels
        df = ddf.describe().compute()

    return df


def clip_and_normalize(data, n_sigma = 2.0):
    data_mean = data.mean()
    data_std  = data.std()
    
    upper = data_mean + n_sigma * data_std
    lower = data_mean - n_sigma * data_std

    clipped = np.where(data > upper, upper, data)
    clipped = np.where(data < lower, lower, clipped)

    return clipped 
