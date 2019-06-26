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


def load_data(
    data_path, col_meta_path, pert_types=None, cell_ids=None, only_landmark=True
):
    """Loads Level3 or Level4 data (gtcx) and subsets by cell_id and pert_type.
    
     GTCX (HFD5):                             Dataframe:
           all genes              
         -------------                        landmark genes
        |             |                          -------- 
        |             |                         |        |
        |             | all samples    --->     |        |  selected samples
        |             |                         |        |
        |             |                          --------
         -------------                    
    
    Inputs:
        - data_path (str): full path to gctx file you want to parse.
        - col_meta_path (str): full path to csv file with sample metadata for experiment.
        - pert_types (list of strings): list of perturbagen types. Default=None.
        - cell_ids (list of strings): list of cell types. Default=None.
        - only_landmark (bool): whether to only subset landmark genes. Default=True.
        
    Output: 
        - data (dataframe): L1000 expression dataframe (samples x genes).
        - sample_metadata (dataframe): (samples x metadata).
        - gene_ids (ndarray): array with entrez ids for each gene (same as colnames in data).
    """
    ridx_max = N_LANDMARK_GENES if only_landmark else None  # only select landmark genes
    sample_metadata = subset_samples(col_meta_path, pert_types, cell_ids)
    with h5py.File(data_path, "r") as gctx_file:
        # Extract sample-ids (col_meta) and gene_ids (row_meta)
        all_sample_ids = pd.Index(gctx_file[CID_NODE][:].astype(str), name="inst_id")
        gene_ids = gctx_file[RID_NODE][:ridx_max].astype(str)
        sample_mask = all_sample_ids.isin(sample_metadata.index)

        # Allow data to be read in chunks in parallel (dask)
        data_dset = gctx_file[DATA_NODE]
        data = da.from_array(data_dset)  # dask array
        data = dd.from_dask_array(
            data[sample_mask, :ridx_max], columns=gene_ids
        ).compute()  # compute in parallel
        data = data.set_index(all_sample_ids[sample_mask])

    sample_metadata = sample_metadata.reindex(data.index)
    return data, sample_metadata, gene_ids


def subset_samples(sample_meta_path, pert_types, cell_ids):
    """Filters metadata by cell_id and pert_type.
    
    Input:
        - sample_meta_path (str): full path to csv file with sample metadata for experiment.
        - pert_types (list of strings): list of perturbagen types.
        - cell_ids (list of strings): list of cell types.
    
    Output:
        - metadata (dataframe): metadata for filtered samples.
    """
    metadata = pd.read_csv(
        sample_meta_path,
        sep="\t",
        na_values="-666",
        index_col="inst_id",
        low_memory=False,
    )

    if pert_types:
        metadata = metadata[metadata.pert_type.isin(pert_types)]

    if cell_ids:
        metadata = metadata[metadata.cell_id.isin(cell_ids)]

    return metadata


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

def get_file_info(file_path):
    return [int(s) for s in file_path.split("/")[-1].split(".")[0].split("_")]

