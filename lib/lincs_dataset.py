import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import altair as alt
import os


class LINCSDataset:
    def __init__(self, data_df, sample_meta_df, gene_meta_df):
        self.data = data_df
        self.sample_meta = sample_meta_df
        self.gene_meta = gene_meta_df
        self.pert_types = sample_meta_df.pert_type.unique()
        self.cell_ids = sample_meta_df.cell_id.unique()

    def sample_rows(self, size=100, meta_groups=None):
        sample_meta_subset = (
            self.sample_meta.sample(size)
            if meta_groups is None
            else self.sample_meta.sample(frac=1).groupby(meta_groups).head(size)
        )

        data_subset = self.data[self.data.index.isin(sample_meta_subset.index)]

        return LINCSDataset(data_subset, sample_meta_subset, self.gene_meta.copy())

    def filter_rows(self, **kwargs):
        filtered_meta = self.sample_meta.copy()

        for colname, values in kwargs.items():
            values = [values] if type(values) == str else values
            filtered_meta = filtered_meta[filtered_meta[colname].isin(values)]

        data_subset = self.data[self.data.index.isin(filtered_meta.index)]

        return LINCSDataset(data_subset, filtered_meta, self.gene_meta.copy())

    def lookup_samples(self, sample_ids):
        mask = self.sample_meta.index.isin(sample_ids)
        return LINCSDataset(
            self.data[mask], self.sample_meta[mask], self.gene_meta.copy()
        )
    
    def dropna(self, subset, inplace=False):
        if type(subset) == str:
            subset = [subset]
        if not inplace:
            filtered_meta = self.sample_meta.dropna(subset=subset)
            filtered_data = self.data[self.data.index.isin(filtered_meta.index)]
            return LINCSDataset(filtered_data, filtered_meta, self.gene_meta.copy())
        else:
            self.sample_meta.dropna(subset=subset, inplace=True)
            self.data = self.data[self.data.index.isin(self.sample_meta.index)]

    def normalize_by_gene(self):
        # TODO: requires train_val_test_split to be called after
        # TODO: implement this normalization per batch during training
        self.data = self.data / self.data.max(axis=0)

    def train_val_test_split(self, p1=0.2, p2=0.2):
        # TODO: must be called after normalization.
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.sample_meta, test_size=p1
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=p2
        )
        train = LINCSDataset(X_train, y_train, self.gene_meta.copy())
        val = LINCSDataset(X_val, y_val, self.gene_meta.copy())
        test = LINCSDataset(X_test, y_test, self.gene_meta.copy())
        return train, val, test

    def to_tf_dataset(self, target="self", shuffle=True, repeated=True, batch_size=32):
        """Creates a tensorflow Dataset to be ingested by Keras."""
        data = self.data.copy()
        sample_meta = self.sample_meta.copy()

        y = (
            data.values
            if target == "self"
            else pd.get_dummies(sample_meta[target]).values
        )  # one-hot encode feature-col
        dataset = tf.data.Dataset.from_tensor_slices((data.values, y))
        if repeated:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
        dataset = dataset.batch(batch_size)
        # `prefetch` lets the dataset fetch batches, in the background while the model is training.
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def to_tsv(self, out_dir, name=None):
        if name is not None:
            name = f"{name}_"
        self.data.to_csv(os.path.join(out_dir, f"{name}data.tsv"), sep="\t")
        self.sample_meta.to_csv(
            os.path.join(out_dir, f"{name}sample_meta.tsv"), sep="\t"
        )

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

        return alt.Chart(df).mark_bar().encode(
            x=alt.X(meta_field, sort=None),
            y=alt.Y("freq", title=None)
        )
    
    def copy(self):
        return LINCSDataset(self.data.copy(), self.sample_meta.copy(), self.gene_meta.copy())

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return f"<LINCS Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"
