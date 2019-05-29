import dask.array as da
import dask.dataframe as dd
import pandas as pd
import h5py
import os

# L1000 Data
N_LANDMARK_GENES = 978
N_SAMPLES = 1_319_138

# HDF5 nodes
RID_NODE = "/0/META/ROW/id"
CID_NODE = "/0/META/COL/id"
DATA_NODE = "/0/DATA/0/matrix"
ROW_META_GROUP_NODE = "/0/META/ROW"
COL_META_GROUP_NODE = "/0/META/COL"

# FILE name and path
file_name = "GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx"
data_dir = os.environ['DATA_DIR']
file_path = os.path.join(data_dir, file_name)


def compute_summary_stats(file_path):
    with h5py.File(file_path) as f:
        data_dset = f[DATA_NODE]
        arr = da.from_array(data_dset)[:, :N_LANDMARK_GENES]
        ddf = dd.from_dask_array(arr)

        landmark_gene_labels = f[RID_NODE][:N_LANDMARK_GENES].astype(str)
        ddf.columns = landmark_gene_labels
        df = ddf.describe().compute()

    return df


if __name__ == "__main__":
    stats = compute_summary_stats(file_path)
    stats.to_csv("summary_stats.tsv", sep="\t")
