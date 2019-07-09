import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns

from .plotting import create_cdict

sns.set(color_codes=True)


class HiddenEmbedding:
    embedders = {"PCA": PCA(n_components=2), "UMAP": UMAP(), "TSNE": TSNE()}

    def __init__(self, dataset, encoder):
        self._dataset = dataset
        self._encoder = encoder
        h = encoder.predict(dataset.data)
        colnames = [f"unit_{i}" for i in range(h.shape[1])]
        self._h = pd.DataFrame(h, columns=colnames, index=self._dataset.data.index)
        self._embeddings = {}

    def zero_out_genes(self, k=50):
        diff_tensor = self._zero_out_difference()
        nsamples, ngenes, nunits = diff_tensor.shape

        kth_idxs = (-diff_tensor).argsort(1)[:, :k, :]
        kth_genes = self._dataset.data.columns.values[kth_idxs].astype(int)

        # find which units are inactive for each sample across all genes
        total_diff_per_sample = diff_tensor.sum(1)
        total_diff_mask = np.where(total_diff_per_sample == 0, np.nan, 1)

        kth_genes = (
            kth_genes * total_diff_mask[:, np.newaxis]
        )  # set gene ids to nan if total diff is 0

        k_rank = np.tile(np.arange(k), nsamples)[:, np.newaxis]

        unit_names = [f"unit_{i}" for i in range(nunits)]

        df = (
            pd.DataFrame(
                np.concatenate(
                    (kth_genes.reshape(nsamples * k, nunits), k_rank), axis=1
                ),
                columns=unit_names + ["k_rank"],
                index=self._dataset.data.index.repeat(k),
            )
            .reset_index()
            .melt(
                id_vars=["inst_id", "k_rank"],
                value_vars=unit_names,
                var_name="unit",
                value_name="gene_id",
            )
            .set_index("inst_id")
            .dropna()  # remove NaNs
        )

        df["k_rank"] = df["k_rank"].astype(int)
        df["gene_id"] = df["gene_id"].astype(int)
        self._ranked_genes = df

    def count_gene_frequency(self, by, k_max=50):
        meta_groups = [by, "unit", "gene_id"]
        count_df = (
            self._ranked_genes.query(f"k_rank < {k_max}")  # filter for desired top_k
            .join(self._dataset.sample_meta)  # add metadata for each sample
            .groupby(meta_groups)
            .size()  # count the frequency of each gene_id for each unit for each item of "by"
            .reset_index(name="freq")
            .merge(
                self._dataset.gene_meta, left_on="gene_id", right_index=True
            )  # add gene metadata
        )
        return count_df

    def _zero_out_difference(self):
        nsamples, ngenes = self._dataset.data.shape

        # Repeat each sample for all genes and set repeated diagonal to zero
        repeated = np.repeat(self._dataset.data.values, ngenes, axis=0)
        repeated[np.arange(repeated.shape[0]), np.tile(np.arange(ngenes), nsamples)] = 0

        # Get the hidden embedding for all repeated and reshape by size of hidden dimension
        zero_out_h = self._encoder.predict(repeated).reshape(
            nsamples, ngenes, self._h.shape[1]
        )

        return (self._h.values[:, np.newaxis] - zero_out_h) ** 2

    def _embed_hidden_output(self, embedding_type):
        if embedding_type not in self.embedders.keys():
            raise ValueError(f"Embedding type must be one of {self.embedders.keys()}")
        if embedding_type in self._embeddings:
            embedding = self._embeddings[embedding_type]
        else:
            embedding = self.embedders[embedding_type].fit_transform(self._h.values)
            self._embeddings[embedding_type] = embedding
        return embedding

    def plot_clustermap(self, meta_colname="cell_id"):
        meta_col = self._dataset.sample_meta[meta_colname]
        cdict = create_cdict(meta_col)
        row_colors = meta_col.map(cdict)
        sns.clustermap(self._h, row_colors=row_colors, z_score=0)

    def plot_embedding(self, type_="PCA", color="cell_id"):
        TYPE = type_.upper()

        df = (
            pd.DataFrame(
                data=self._embed_hidden_output(TYPE),
                columns=[f"{TYPE}_1", f"{TYPE}_2"],
                index=self._h.index,
            )
            .join(self._dataset.sample_meta)
            .reset_index()
        )

        return (
            alt.Chart(df)
            .mark_circle()
            .encode(x=f"{TYPE}_1", y=f"{TYPE}_2", color=color, tooltip=["inst_id"])
        )
