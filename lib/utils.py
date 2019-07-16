import pandas as pd
import dask.array as da
import dask.dataframe as dd
import h5py

from .lincs_dataset import LINCSDataset

# L1000 Data
N_LANDMARK_GENES = 978

# HDF5 nodes
RID_NODE = "/0/META/ROW/id"
CID_NODE = "/0/META/COL/id"
DATA_NODE = "/0/DATA/0/matrix"
ROW_META_GROUP_NODE = "/0/META/ROW"
COL_META_GROUP_NODE = "/0/META/COL"


def load_data(
    data_path,
    inst_meta_path,
    cell_meta_path,
    gene_meta_path,
    pert_types=None,
    cell_ids=None,
    only_landmark=True,
):
    """Loads Level3 or Level4 data (gtcx) and subsets by cell_id and pert_type.
    
     GTCX (HFD5):                             LINCS DATASET:
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
        - inst_meta_path (str): full path to tsv file with sample metadata.
        - cell_meta_path (str): full path to tsv file with cell metadata.
        - gene_meta_path (str): full path to tsv file with gene metadata.
        - pert_types (list of strings): list of perturbagen types. Default=None.
        - cell_ids (list of strings): list of cell types. Default=None.
        - only_landmark (bool): whether to only subset landmark genes. Default=True.
        
    Output: 
        - data (dataframe): L1000 expression dataframe (samples x genes).
        - sample_metadata (dataframe): (samples x metadata).
        - gene_ids (ndarray): array with entrez ids for each gene (same as colnames in data).
    """
    ridx_max = N_LANDMARK_GENES if only_landmark else None  # only select landmark genes
    sample_metadata = subset_samples(inst_meta_path, pert_types, cell_ids)

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

    cell_info = pd.read_csv(cell_meta_path, sep="\t", na_values="-666")
    sample_metadata = merge_cell_metadata(
        cell_meta_path, sample_metadata.reindex(data.index)
    )
    gene_metadata = load_gene_metadata(gene_meta_path, gene_ids)
    return LINCSDataset(data, sample_metadata, gene_metadata)


def load_gene_metadata(gene_meta_path, gene_ids):
    gene_info = pd.read_csv(gene_meta_path, sep="\t")
    gene_info.columns = gene_info.columns.str.slice(
        3
    )  # remove leading "pr_" from columns
    gene_info = gene_info.set_index("gene_id")
    return gene_info[gene_info.index.isin(gene_ids)]


def merge_cell_metadata(cell_meta_path, sample_meta_df):
    cell_info = pd.read_csv(cell_meta_path, sep="\t", na_values="-666")
    return sample_meta_df.reset_index().merge(cell_info).set_index("inst_id")


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


def compute_summary_stats(data_path):
    with h5py.File(data_path) as f:
        data_dset = f[DATA_NODE]
        arr = da.from_array(data_dset)[:, :N_LANDMARK_GENES]
        ddf = dd.from_dask_array(arr)

        landmark_gene_labels = f[RID_NODE][:N_LANDMARK_GENES].astype(str)
        ddf.columns = landmark_gene_labels
        df = ddf.describe().compute()

    return df
