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

    def get_genes_by_unit(self, by=None, units=None, count_thresh=0, k_max=50):
        counts = self._count_gene_frequency(by, count_thresh, k_max)
        if units is not None:
            counts = counts.query(
                "unit == " + " | unit == ".join([f"'unit_{i}'" for i in units])
            )
        ids = {unit: {} for unit in counts.unit.unique()}
        for unit_name in counts.unit.unique():
            unit_activations = counts.query(f"unit == '{unit_name}'")
            if by is None:
                ids[unit_name] = unit_activations.gene_symbol.to_list()
            else:
                for item in unit_activations[by].unique():
                    filtered = unit_activations.query(
                        f"{by} == '{item}'"
                    )
                    # show freq and gene_names
                    # ids[unit_name][item] = list(zip(filtered.gene_symbol.to_list(), filtered.freq.to_list())) 
                    ids[unit_name][item] = filtered.gene_symbol.to_list() # just gene names
        return ids

    def plot_clustermap(self, meta_colnames="cell_id", only_active_units=True):
        if not isinstance(meta_colnames, (list, tuple)):
            meta_colnames = [meta_colnames]
        row_colors = [self._meta_col_cmap(colname) for colname in meta_colnames]
        embedding = self._h.loc[:,self._h.sum(0) > 0] if only_active_units else self._h
        sns.clustermap(embedding, row_colors=row_colors, standard_scale=1)
    
    def _meta_col_cmap(self, colname):
        meta_column = self._dataset.sample_meta[colname]
        cdict = create_cdict(meta_column)
        row_colors = meta_column.map(cdict)
        return row_colors

    def plot_embedding(self, type_="PCA", color=None, facet_row=None):
        TYPE = type_.upper()
        embedding = self._embed_hidden_output(TYPE)
        df = (
            pd.DataFrame(
                embedding, columns=[f"{TYPE}_1", f"{TYPE}_2"], index=self._h.index
            )
            .join(self._dataset.sample_meta)  # add metadata
            .reset_index()
        )
     
        scatter = (
            alt.Chart(df)
            .mark_circle()
            .encode(x=f"{TYPE}_1", y=f"{TYPE}_2", tooltip=["inst_id"])
        )
        if color:
            scatter = scatter.encode(color=color)
        if facet_row:
            scatter = scatter.encode(row=facet_row)
            
        return scatter

    def plot_unit_counts(
        self, meta_field="cell_id", units=None, count_thresh=0, k_max=50
    ):
        counts = self._count_gene_frequency(
            by=meta_field, count_thresh=count_thresh, k_max=k_max
        )
        if units is not None:
            counts = counts.query(
                "unit == " + " | unit == ".join([f"'unit_{i}'" for i in units])
            )
        if len(counts) == 0: 
            raise ValueError(
                f"There are no genes (in the top-k of {k_max}) which "
                f"exceed the count threshold of {count_thresh}. "
                f"Try lowering this threshold or increasing the k_max."
            )
        barplot = self._create_barplot(counts, meta_field)
        return barplot

    def _k_and_per_unit_threshold(self, count_thresh, k_max):
        return (
            self._ranked_genes.query(f"k_rank < {k_max}")  # filter for desired top_k
            .groupby(["unit", "gene_id"])
            .filter(
                lambda x: len(x) > count_thresh
            )  # set frequency thresh independent of meta-data
        )

    def _count_gene_frequency(self, by, count_thresh, k_max):
        meta_groups = ["unit", "gene_id"]
        threshed = self._k_and_per_unit_threshold(count_thresh, k_max)
        counts_df = (
            threshed.join(self._dataset.sample_meta)  # add metadata for each sample
            .groupby([by] + meta_groups)  # `by` must be a field in metadata
            .size()  # count the frequency of each gene_id for each unit for each item of "by"
            .reset_index(name="freq")
            .merge(
                self._dataset.gene_meta, left_on="gene_id", right_index=True
            )  # add gene metadata
        )
        return counts_df

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
    
    def _create_barplot(self, counts_df, meta_field):
        return (
            alt.Chart(counts_df)
            .transform_calculate(
                url="https://www.ncbi.nlm.nih.gov/gene/" + alt.datum.gene_id
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "gene_symbol:N",
                    sort=alt.EncodingSortField(field="freq", order="descending"),
                ),
                y="freq:Q",
                color=f"{meta_field}:N",
                href="url:N",
                tooltip=["gene_id:N", "gene_symbol:N", "freq:Q", f"{meta_field}:N"],
            )
            .properties(height=50, width=850)
            .facet(row="unit")
        )
        
