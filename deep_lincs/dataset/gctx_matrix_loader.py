import os
import dask.array as da
import dask.dataframe as dd
import h5py
import pandas as pd

# HDF5 nodes
RID_NODE = "/0/META/ROW/id"
CID_NODE = "/0/META/COL/id"
DATA_NODE = "/0/DATA/0/matrix"
ROW_META_GROUP_NODE = "/0/META/ROW"
COL_META_GROUP_NODE = "/0/META/COL"


class GctxMatrixLoader:
    def __init__(self, path, sample_index_name, gene_index_name):
        self.path = path
        self.sample_index_name = sample_index_name
        self.gene_index_name = gene_index_name
        self.n_rows, self.n_cols = self.get_shape()

    @classmethod
    def from_node(cls, node, data_dir):
        path = os.path.join(data_dir, node.pop("file"))
        return cls(path, **node)

    def get_shape(self):
        with h5py.File(self.path, "r") as gctx_file:
            shape = gctx_file[DATA_NODE].shape
        return shape

    def read(self, sample_ids, max_genes):
        """Returns an expression dataframe subsetted by held out samples (row_index)"""
        with h5py.File(self.path, "r") as gctx_file:
            # Extract sample-ids (col_meta) and gene_ids (row_meta)
            all_sample_ids = pd.Index(
                gctx_file[CID_NODE][:].astype(str), name=self.sample_index_name
            )
            gene_ids = gctx_file[RID_NODE][:max_genes].astype(str)
            sample_mask = all_sample_ids.isin(sample_ids)

            # Allow data to be read in chunks in parallel (dask)
            data_dset = gctx_file[DATA_NODE]
            data = da.from_array(data_dset)  # dask array
            data = dd.from_dask_array(
                data[sample_mask, :max_genes], columns=gene_ids
            ).compute()  # compute in parallel
            data = data.set_index(all_sample_ids[sample_mask])
        return data

    def __repr__(self):
        return (
            f"< GctxMatrixLoader ({self.sample_index_name}: {self.n_rows:,}"
            f" , {self.gene_index_name}: {self.n_cols:,}) >"
        )
