import os
import yaml
import pandas as pd
from functools import partial
import numpy as np

from .gctx_matrix_loader import GctxMatrixLoader
from .csv_metadata_loader import CsvMetaDataLoader


def yaml_to_dataframes(yaml_file, only_landmark=True, **filter_kwargs):
    max_genes = 978 if only_landmark else None  # only select landmark genes
    data_loader, sample_meta_loader, gene_meta_loader = parse_settings(yaml_file)
    # Filter metadata by categorical column values
    sample_metadata = load_df_and_filter(sample_meta_loader, **filter_kwargs)
    # Read only a subset of the data matrix
    data = data_loader.read(sample_ids=sample_metadata.index, max_genes=max_genes)
    gene_metadata = gene_meta_loader.read(ids=data.columns.tolist())
    return data, sample_metadata.reindex(data.index), gene_metadata


def load_df_and_filter(loader, **kwargs):
    df = loader.read()
    for colname, values in kwargs.items():
        values = [values] if type(values) == str else values
        df = df[df[colname].isin(values)]
    return df


def parse_settings(yaml_file):
    with open(yaml_file) as f:
        d = yaml.safe_load(f)
    data_dir = d["data_dir"]
    data_loader = GctxMatrixLoader.from_node(d["data"], data_dir)
    sample_meta_loader = CsvMetaDataLoader.from_node(d["sample_metadata"], data_dir)
    gene_meta_loader = CsvMetaDataLoader.from_node(d["gene_metadata"], data_dir)
    return data_loader, sample_meta_loader, gene_meta_loader
