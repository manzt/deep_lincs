import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import altair as alt
import os

from .tf_dataset_pipeline import prepare_tf_dataset


class LINCSDataset:
    def __init__(self, data, gene_meta):
        self._data = data
        self.gene_meta = gene_meta
        self._split_index = len(gene_meta.index.values)

    @property
    def data(self):
        return self._data.iloc[:, : self._split_index]

    @property
    def sample_meta(self):
        return self._data.iloc[:, self._split_index :]

    @classmethod
    def from_dataframes(cls, data_df, sample_meta_df, gene_meta_df):
        data = data_df.join(sample_meta_df)
        return cls(data, gene_meta_df)

    def sample_rows(self, size=100, meta_groups=None):
        subset = (
            self._data.sample(size)
            if meta_groups is None
            else self._data.sample(frac=1).groupby(meta_groups).head(size)
        )
        return LINCSDataset(subset, self.gene_meta.copy())

    def filter_rows(self, **kwargs):
        filtered = self._data.copy()
        for colname, values in kwargs.items():
            values = [values] if type(values) == str else values
            filtered = filtered[filtered[colname].isin(values)]
        return LINCSDataset(filtered, self.gene_meta.copy())

    def lookup_samples(self, sample_ids):
        mask = self._data.isin(sample_ids)
        return LINCSDataset(self._data[mask], self.gene_meta.copy())

    def dropna(self, subset, inplace=False):
        if type(subset) is str:
            subset = [subset]
        if not inplace:
            filtered = self._data.dropna(subset=subset)
            return LINCSDataset(filtered, self.gene_meta.copy())
        else:
            self._data.dropna(subset=subset, inplace=True)

    def train_val_test_split(self, p1=0.2, p2=0.2):
        X_train, X_test = train_test_split(self._data, test_size=p1)
        X_train, X_val = train_test_split(X_train, test_size=p2)
        train = LINCSKerasDataset(X_train, self.gene_meta.copy(), "train")
        val = LINCSKerasDataset(X_val, self.gene_meta.copy(), "validation")
        test = LINCSKerasDataset(X_test, self.gene_meta.copy(), "test")
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

    def gene_boxplot(self, gene_id=None, gene_symbol=None, size=5000):
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
        if gene_info.shape[0] == 0:
            raise ValueError(
                "Gene not found please make sure you have the correct id or symbol."
            )

        df = (
            self.data.loc[:, gene_info.index.astype(str)]
            .join(self.sample_meta)
            .sample(size)
        )

        return (
            alt.Chart(df)
            .mark_boxplot(extent="min-max")
            .encode(
                x="cell_id:N",
                y=alt.Y(
                    f"{gene_info.index[0]}:Q",
                    title=f"{gene_info.pr_gene_symbol.values[0]}",
                ),
            )
        )

    def plot_meta_counts(self, meta_field, normalize=True, sort_values=True):
        count_values = self.sample_meta[meta_field].value_counts(normalize=normalize)
        labels = count_values.index.values
        freq = count_values.values
        df = pd.DataFrame({meta_field: labels, "freq": freq})

        return (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X(meta_field, sort=None), y=alt.Y("freq", title=None))
        )

    def copy(self):
        return LINCSDataset(self._data.copy(), self.gene_meta.copy())

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return f"<LINCS Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"


class LINCSKerasDataset(LINCSDataset):
    _valid_names = ["train", "validation", "test"]

    def __init__(self, data, gene_meta, name, **kwargs):
        super(LINCSKerasDataset, self).__init__(data, gene_meta, **kwargs)
        self.name = name
        if name not in self._valid_names:
            raise ValueError(
                f"LINCSKerasDataset 'name' must be one of {self._valid_names}, not '{name}'."
            )
        self.shuffle, self.repeat = (True, True) if name is "train" else (False, False)

    def __call__(
        self, target, batch_size=64, norm_method="z_score", batch_normalize=False
    ):
        """Converts dataset to tf.data.Dataset to be ingested by Keras."""
        X = tf.data.Dataset.from_tensor_slices(self.data.values)
        y = self._get_target_as_tf_dataset(target)
        tf_dataset = prepare_tf_dataset(
            dataset=tf.data.Dataset.zip((X, y)),
            batch_size=batch_size,
            shuffle=self.shuffle,
            repeat=self.repeat,
            norm_method=norm_method,
            batch_normalize=batch_normalize,
            shuffle_buffer_size=self.data.shape[0],
        )
        return tf_dataset

    def _get_target_as_tf_dataset(self, target):
        if target == "self":
            y = self.data.values
        elif type(target) == str:
            y = pd.get_dummies(self.sample_meta[target]).values
        elif type(target) == list:
            y = tuple(pd.get_dummies(self.sample_meta[t]).values for t in target)
        y_tf_dataset = tf.data.Dataset.from_tensor_slices(y)
        return y_tf_dataset

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return (
            f"<LINCS {self.name} Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"
        )
