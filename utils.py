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


def load_data(data_path, col_meta_path, pert_types, only_landmark=True):
    ridx_max = N_LANDMARK_GENES if only_landmark else None  # only select landmark genes
    desired_sample_ids = get_desired_ids(col_meta_path, pert_types)
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
        mask = np.isin(all_sample_ids, desired_sample_ids)
        data = data[mask, :ridx_max].compute()

    return all_sample_ids[mask], gene_labels, data


def get_desired_ids(col_meta_path, pert_types):
    metadata = pd.read_csv(col_meta_path, sep="\t", low_memory=False)
    desired_sample_ids = metadata.inst_id[metadata.pert_type.isin(pert_types)]
    return desired_sample_ids.values.astype(str)


def split_data(X, y, p=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=p)
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
