import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


class LINCSDataset:
    def __init__(self, data_df, sample_meta_df, gene_meta_df):
        self.data = data_df
        self.sample_meta = sample_meta_df
        self.gene_meta = gene_meta_df
        self.pert_types = sample_meta_df.pert_type.unique()
        self.cell_ids = sample_meta_df.cell_id.unique()

    def sample_rows(self, meta_groups=None, size=100):
        data, sample_meta, gene_meta = self._copy_dfs()

        sample_meta_subset = (
            sample_meta.sample(size)
            if meta_groups is None
            else sample_meta.sample(frac=1).groupby(meta_groups).head(size)
        )
        data_subset = data[data.index.isin(sample_meta_subset.index)]

        return LINCSDataset(data_subset, sample_meta_subset, gene_meta)

    def filter_rows(self, **kwargs):
        data, filtered_meta, gene_meta = self._copy_dfs()

        for colname, values in kwargs.items():
            values = [values] if type(values) == str else values
            filtered_meta = filtered_meta[filtered_meta[colname].isin(values)]

        data_subset = data[data.index.isin(filtered_meta.index)]

        return LINCSDataset(data_subset, filtered_meta, gene_meta)

    def lookup_samples(self, sample_ids):
        data, sample_meta, gene_meta = self._copy_dfs()
        mask = sample_meta.index.isin(sample_ids)
        return LINCSDataset(data[mask], sample_meta[mask], gene_meta)

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
        self.train = LINCSDataset(X_train, y_train, self.gene_meta)
        self.val =  LINCSDataset(X_val, y_val, self.gene_meta)
        self.test = LINCSDataset(X_test, y_test, self.gene_meta)
        
    def to_tf_dataset(self, target="self", shuffle=True, repeated=True, batch_size=32):
        """Creates a tensorflow Dataset to be ingested by Keras."""
        y = self.data.values if target == "self" else self.sample_meta[target]
        dataset = tf.data.Dataset.from_tensor_slices((self.data.values, y))
        if repeated:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.data.shape[0])
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

    def _copy_dfs(self):
        return self.data.copy(), self.sample_meta.copy(), self.gene_meta.copy()

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return f"<LINCS Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"