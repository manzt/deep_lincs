import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import altair as alt
import os

from .normalizers import get_norm_method
from .tf_dataset_pipeline import prepare_tf_dataset
from .load_yaml import yaml_to_dataframes
from ..plotting.plots import boxplot, barplot


class Dataset:
    def __init__(self, data, gene_meta, n_genes=978):
        self._data = data
        self.gene_meta = gene_meta
        self.n_genes = n_genes

    @property
    def data(self):
        return self._data.iloc[:, : self.n_genes]

    @property
    def sample_meta(self):
        return self._data.iloc[:, self.n_genes :]

    @classmethod
    def from_dataframes(cls, data_df, sample_meta_df, gene_meta_df):
        data = data_df.join(sample_meta_df)
        return cls(data, gene_meta_df, len(gene_meta_df))

    @classmethod
    def from_yaml(cls, yaml_file, sample_ids=None, only_landmark=True, **filter_kwargs):
        data_df, sample_meta_df, gene_meta_df = yaml_to_dataframes(
            yaml_file, sample_ids, only_landmark, **filter_kwargs
        )
        data = data_df.join(sample_meta_df)
        return cls(data, gene_meta_df, len(gene_meta_df))

    def _copy(self, data):
        return Dataset(data, self.gene_meta.copy(), self.n_genes)

    def sample_rows(self, size=100, replace=False, meta_groups=None):
        sampled = (
            self._data.sample(size, replace=replace)
            if meta_groups is None
            else self._data.sample(frac=1, replace=replace)
            .groupby(meta_groups)
            .head(size)
        )
        return self._copy(sampled)

    def filter_rows(self, **kwargs):
        filtered = self._data.copy()
        for colname, values in kwargs.items():
            values = [values] if type(values) == str else values
            filtered = filtered[filtered[colname].isin(values)]
        return self._copy(filtered)

    def select_meta(self, meta_fields):
        selected = self._data[[*self.data.columns.values, *meta_fields]]
        return self._copy(selected)

    def lookup_samples(self, sample_ids):
        """Returns new dataset with subset of fields."""
        mask = self._data.isin(sample_ids)
        return self._copy(self._data[mask])

    def split(self, **kwargs):
        if len(kwargs.keys()) != 1:
            raise ValueError(
                "One keyword argument is required: Key must be a meta_data field."
            )
        data = self._data.copy()
        for colname, values in kwargs.items():
            mask = data[colname].isin(values)
            return self._copy(data[mask]), self._copy(data[~mask])

    def merge(self, lincs_dataset, r_prefix="ctl_"):
        data = self._merge_right(self.data, lincs_dataset.data, r_prefix)
        sample_meta = self._merge_right(
            self.sample_meta, lincs_dataset.sample_meta, r_prefix
        )
        merged = data.join(sample_meta.drop("inst_id", axis=1)).set_index("inst_id")
        return Dataset(merged, self.gene_meta.copy(), data.shape[1] - 1)

    def _merge_right(self, df1, df2, r_prefix):
        df2 = df2.add_prefix(r_prefix).reset_index().drop("inst_id", axis=1)
        merged = df1.reset_index().join(df2)
        return merged

    def set_categorical(self, colname):
        self._data[colname] = pd.Categorical(self._data[colname])

    def dropna(self, subset, inplace=False):
        if type(subset) is str:
            subset = [subset]
        if not inplace:
            filtered = self._data.dropna(subset=subset)
            return self._copy(filtered)
        else:
            self._data.dropna(subset=subset, inplace=True)

    def normalize_by_gene(self, normalizer):
        normalizer = get_norm_method(normalizer)
        self._data.iloc[:, : self.n_genes] = normalizer(self.data)

    def train_val_test_split(self, p1=0.2, p2=0.2):
        X_train, X_test = train_test_split(self._data, test_size=p1)
        X_train, X_val = train_test_split(X_train, test_size=p2)
        train = KerasDataset(X_train, self.gene_meta.copy(), self.n_genes, "train")
        val = KerasDataset(X_val, self.gene_meta.copy(), self.n_genes, "validation")
        test = KerasDataset(X_test, self.gene_meta.copy(), self.n_genes, "test")
        return train, val, test

    def to_tsv(self, out_dir, prefix=None):
        # create dirs if non-existent
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"{prefix}_" if prefix else ""
        fpaths = [
            os.path.join(out_dir, f"{prefix}{suf}.tsv")
            for suf in ["data", "sample_meta"]
        ]
        self.data.to_csv(fpaths[0], sep="\t")
        self.sample_meta.to_csv(fpaths[1], sep="\t")

    def one_hot_encode(self, meta_field):
        one_hot_df = pd.get_dummies(self.sample_meta[meta_field])
        return one_hot_df.values

    def gene_boxplot(
        self, gene_id=None, gene_symbol=None, meta_field=None, extent="min-max"
    ):
        if gene_id is None and gene_symbol is None:
            raise ValueError("Must provide either gene_id or gene_symbol.")
        if gene_id is not None and gene_symbol is not None:
            raise ValueError(
                "Cannot provide both gene_id and gene_symbol. Please use one."
            )
        if gene_id:
            gene_mask = self.gene_meta.index == int(gene_id)
        elif gene_symbol:
            gene_mask = self.gene_meta.gene_symbol == str(gene_symbol)
        gene_info = self.gene_meta[gene_mask]
        if gene_info.shape[0] is 0:
            raise ValueError(
                "Gene not found please make sure you have the correct id or symbol."
            )
        gene_id = gene_info.index[0].astype(str)
        gene_symbol = gene_info.gene_symbol.values[0]
        df = self._data[[gene_id, *self.sample_meta.columns]]
        df = df.rename(columns={gene_id: gene_symbol})
        return boxplot(df=df, x_field=meta_field, y_field=gene_symbol, extent=extent)

    def plot_meta_counts(self, meta_field, normalize=True, sort_values=True):
        counts = self.sample_meta[meta_field].value_counts(normalize=normalize)
        colname = "counts" if normalize is False else "frequency"
        df = pd.DataFrame({meta_field: counts.index.values, colname: counts.values})
        return barplot(df=df, x_field=meta_field, y_field=colname)

    def copy(self):
        return Dataset(self._data.copy(), self.gene_meta.copy(), self.n_genes)

    def __len__(self):
        return self._data.shape[0]

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return f"<L1000 Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"


class KerasDataset(Dataset):
    _valid_names = ["train", "validation", "test"]

    def __init__(self, data, gene_meta, n_genes, name, **kwargs):
        super(KerasDataset, self).__init__(data, gene_meta, n_genes, **kwargs)
        self.name = name
        if name not in self._valid_names:
            raise ValueError(
                f"LINCSKerasDataset 'name' must be one of {self._valid_names}, not '{name}'."
            )
        self.shuffle, self.repeat = (True, True) if name is "train" else (False, False)

    def __call__(self, target, batch_size=64, batch_normalize=None):
        """Converts dataset to tf.data.Dataset to be ingested by Keras."""
        X = tf.data.Dataset.from_tensor_slices(self.data.values)
        y = self._get_target_as_tf_dataset(target)
        tf_dataset = prepare_tf_dataset(
            dataset=tf.data.Dataset.zip((X, y)),
            batch_size=batch_size,
            shuffle=self.shuffle,
            repeat=self.repeat,
            batch_normalize=batch_normalize,
            shuffle_buffer_size=self.data.shape[0],
        )
        return tf_dataset

    @classmethod
    def from_lincs_dataset(cls, lincs_dataset, name):
        return cls(
            lincs_dataset._data, lincs_dataset.gene_meta, lincs_dataset.n_genes, name
        )

    def _get_target_as_tf_dataset(self, target):
        if target == "self":
            y = self.data.values
        elif type(target) == str:
            y = self.one_hot_encode(target)
        elif type(target) == list:
            y = tuple(self.one_hot_encode(t) for t in target)
        y_tf_dataset = tf.data.Dataset.from_tensor_slices(y)
        return y_tf_dataset

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return f"< {self.name} Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"
