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
    """Represents an L1000 Dataset

    Parameters
    ----------
    ``data`` : ``dataframe``, shape (n_samples, (n_genes + n_metadata_fields))
            A sample by gene expression matrix padded to the right with per sample metadata.
            Generally it is easiest to construct a Dataset from a class method, ``Dataset.from_yaml()`` or ``Dataset.from_dataframes()``.
            
    ``gene_meta`` : ``dataframe``, shape (n_genes, n_features)
            Contains the metadata for each of the genes in the data matrix.
            
    ``n_genes`` : ``int``
            Number of genes in expression matrix. This explicitly defines 
            the column index which divides the expression values and metadata.
            
    Attributes
    ----------
    ``data`` : ``dataframe``, shape (n_samples, n_genes)
            Gene expression matrix as a dataframe. Shared indicies with `self.sample_meta` and `self.gene_meta`.
            
    ``sample_meta`` : ``dataframe``, shape (n_samples, n_metadata_features)
            Per profile metadata. Row index same as ``self.data.index``.
            
    ``gene_meta`` : ``dataframe``, shape (n_genes, n_gene_features)
            Gene metadata. Row index same as ``self.data.columns``.
    """
    def __init__(self, data, gene_meta, n_genes):
        self._data = data
        self.gene_meta = gene_meta
        self.n_genes = n_genes

    @property
    def data(self):
        """A dataframe representing the sample x gene expression matrix"""
        return self._data.iloc[:, : self.n_genes]

    @property
    def sample_meta(self):
        """A dataframe representing the per sample metadata"""
        return self._data.iloc[:, self.n_genes :]

    @classmethod
    def from_dataframes(cls, data_df, sample_meta_df, gene_meta_df):
        """Dataset constructor method from multiple dataframes

        Parameters
        ----------
        ``data_df`` : dataframe, shape (n_samples, n_genes)
                Contains the expression data from experiment. Must have 
                shared row index with ``sample_meta_df``.

        ``sample_meta_df`` : ``dataframe``, shape (n_samples, n_meta_features)
                Contains the metadata for each of the samples in experiment.

        ``gene_meta_df`` : dataframe, shape (n_genes, n_gene_features)
                Contains the metadata for each of the genes in experiment.
        """
        data = data_df.join(sample_meta_df)
        return cls(data, gene_meta_df, len(gene_meta_df))

    @classmethod
    def from_yaml(cls, path, sample_ids=None, only_landmark=True, **filter_kwargs):
        """Dataset constructor method from yaml specification

        Parameters
        ----------
        ``path`` : ``str``
                Valid string path to ``.yaml`` or ``.yml`` file.
        
        ``sample_ids`` : ``list`` (optional, default ``None``)
                Unique sample ids to read from data and metadata files.
                
        ``only_landmark`` : ``bool`` (optional, default ``True``)
                Whether to parse all genes or only the landmark.
                
        ``filter_kwargs`` :
                Optional keyword args to subset data by specific features 
                in per sample metadata. Each kwarg must follow the following.
                ``keyword`` - a column in metadata ``arg``     - a list 
                of values to filter from keyword field.
                
        >>> Dataset.from_yaml("settings.yaml", cell_id=["MCF7", "PC3"], pert_id=["trt_cp"]) 
        """
        data_df, sample_meta_df, gene_meta_df = yaml_to_dataframes(
            path, sample_ids, only_landmark, **filter_kwargs
        )
        data = data_df.join(sample_meta_df)
        return cls(data, gene_meta_df, len(gene_meta_df))

    def sample_rows(self, size, replace=False, meta_groups=None):
        """Returns a Dataset of sampled profiles
        
        Parameters
        ----------
        ``size`` : ``int``
                Number of samples to return per meta grouping. Default is 
                to sample from all profiles.
        
        ``replace`` : ``bool`` (optional, default ``False``)
                Sample with or without replacement. 
            
        ``meta_groups`` : ``str`` or ``list`` (optional, default ``None``)
                If provided, equal numbers of profiles are returned for each metadata grouping.
                
        >>> dataset.sample_rows(size=5000, meta_groups="cell_id") 
            // returns 5000 profiles for each cell_id in dataset
        >>> dataset.sample_rows(size=5000, meta_groups=["cell_id", "pert_type"])
            // returns 5000 profiles for all groupings of cell_id and pert_type
        """
        sampled = (
            self._data.sample(size, replace=replace)
            if meta_groups is None
            else self._data.sample(frac=1, replace=replace)
            .groupby(meta_groups)
            .head(size)
        )
        return self._copy(sampled)

    def filter_rows(self, **kwargs):
        """Returns a Dataset of filtered profiles
        
        Parameters
        ----------
        ``kwargs`` :
                Keyword args to subset data by specific features in sample metadata. 
                Each kwarg must follow the following.``keyword``: a column in metadata, 
                ``arg``: a list of values to filter from keyword field.
                
        >>> dataset.filter_rows(cell_id=["VCAP, PC3"]) 
        >>> dataset.filter_rows(cell_id="VCAP", pert_type=["ctl_vehicle", "trt_cp"])
        """
        filtered = self._data.copy()
        for colname, values in kwargs.items():
            values = [values] if type(values) == str else values
            filtered = filtered[filtered[colname].isin(values)]
        return self._copy(filtered)

    def select_meta(self, meta_fields):
        """Returns a Dataset with select metadata fields.
        
        Parameters
        ----------
        ``meta_fields`` : ``list``
                Desired metadata columns.
        
        >>> dataset.select_meta(["cell_id", "pert_id", "moa"])
            // returns dataset with only ["cell_id", "pert_id", "moa"] as metadata fields.
        """
        selected = self._data[[*self.data.columns.values, *meta_fields]]
        return self._copy(selected)

    def select_samples(self, sample_ids):
        """Returns a Dataset with profiles selected by id
        
        Parameters
        ----------
        ``sample_ids`` : ``list``, character ``array``
                Desired sample ids to filter dataset.
        """
        mask = self._data.isin(sample_ids)
        return self._copy(self._data[mask])

    def split(self, **kwargs):
        """Returns a tuple of Datasets, split by inclusion criteria
        
        Parameters
        ----------
        ``kwargs`` :
                Keyword args to subset data by specific features in sample metadata. 
                Each kwarg must follow the following. ``keyword``: a column in metadata,
                ``arg``: a str or list of values to filter from keyword field.
                    
        >>> pc3, not_pc3 = dataset.split(cell_id="PC3")
        >>> vcap_mcf7, not_vcap_mcf7 = dataset.split(cell_id=["VCAP", "MCF7"])
        """
        if len(kwargs.keys()) != 1:
            raise ValueError(
                "One keyword argument is required: Key must be a meta_data field."
            )
        data = self._data.copy()
        for colname, values in kwargs.items():
            values = [values] if type(values) == str else values
            mask = data[colname].isin(values)
            return self._copy(data[mask]), self._copy(data[~mask])

    def dropna(self, subset, inplace=False):
        """Drops profiles for which there is no metadata in subset
        
        Parameters
        ----------
        ``subset`` : ``str`` or ``list``
                Metadata field or fields.
        
        ``inplace``: ``bool`` (optional, default: ``False``)
                If True, do operation inplace and return None. 
        """
        if type(subset) is str:
            subset = [subset]
        if not inplace:
            filtered = self._data.dropna(subset=subset)
            return self._copy(filtered)
        else:
            self._data.dropna(subset=subset, inplace=True)
    
    def set_categorical(self, colname):
        """Sets sample metadata column as categorical
        
        Parameters
        ----------
        ``colname``: ``str``
                Sample metadata column name.
        """
        self._data[colname] = pd.Categorical(self._data[colname])

    def normalize_by_gene(self, normalizer="standard_scale"):
        """Normalize expression by gene
        
        Parameters
        ----------
        ``normalizer`` : ``str`` or ``func`` (optional, default ``'standard_scale'``)
                Method used normalise dataset. Valid str options are ``'standard_scale'`` 
                and ``'z_score'``. If a function is provided, it must take 
                one argument (``array``), and return an array of the same dimensions.
        """
        normalizer = get_norm_method(normalizer)
        self._data.iloc[:, : self.n_genes] = normalizer(self.data)

    def train_val_test_split(self, p1=0.2, p2=0.2):
        """Splits dataset into training, validation, and test datasets
        
        Parameters
        ----------
        ``p1``: ``float`` (optional: default ``0.2``)
            Test size in first train/test split.
        ``p2``: ``float`` (optional: default ``0.2``)
            Validation size in remaining train/val split.
            
        Returns
        -------
        ``train, val, test`` : ``tuple`` of ``Dataset``
        """
        X_train, X_test = train_test_split(self._data, test_size=p1)
        X_train, X_val = train_test_split(X_train, test_size=p2)
        train = KerasDataset(X_train, self.gene_meta.copy(), self.n_genes, "train")
        val = KerasDataset(X_val, self.gene_meta.copy(), self.n_genes, "validation")
        test = KerasDataset(X_test, self.gene_meta.copy(), self.n_genes, "test")
        return train, val, test

    def to_tsv(self, out_dir, sep="\t", prefix=None, **kwargs):
        """Write Dataset object to a tsv file
        
        Parameters
        ----------
        ``out_dir`` : ``str``
            Path to output directory.
            
        ``sep`` : ``str`` (optional)
            String of length 1. Field delimiter for the output file.
            
        ``prefix`` : ``str`` (optional, default ``None``)
            Filename prefix.
        """
        os.makedirs(out_dir, exist_ok=True) # create dirs if non-existent
        prefix = f"{prefix}_" if prefix else ""
        fpaths = [
            os.path.join(out_dir, f"{prefix}{suf}.tsv")
            for suf in ["data", "sample_meta"]
        ]
        self.data.to_csv(fpaths[0], sep="\t", **kwargs)
        self.sample_meta.to_csv(fpaths[1], sep="\t", **kwargs)

    def one_hot_encode(self, meta_field):
        """Return a one-hot vector for a metadata field for all profiles
        
        Parameters
        ----------
        ``meta_field`` : ``str``
            Valid sample metadata column.
        
        Returns
        -------
        ``one_hot`` : ``array``, (n_samples, n_categories)
        """
        one_hot = pd.get_dummies(self.sample_meta[meta_field]).values
        return one_hot

    def plot_gene_boxplot(
        self, identifier, lookup_col=None, meta_field=None, extent=1.5,
    ):
        """Returns a boxplot of gene expression, faceted on metadata field
        
        Parameters
        ----------
        ``identifier`` : ``str``
                String identifier for gene. Default should be one of `self.gene_meta.index`.
        
        ``lookup_col`` : ``str`` (optional, default ``None``)
                Gene metadata column name. Will be used to lookup `identifier` param rather than index.
                
        ``meta_field`` : ``str`` (optional, default ``None``)
                Sample metadata column name. Will make multiple boxplots for each metadata category.
                           
        ``extent`` : ``str`` or ``float`` (optional, default ``1.5``)
                Can be either ``'min-max'``, with whiskers covering entire domain, or an number X where 
                entries outside X stds are shown as individual points.
                
        >>> dataset.plot_gene_boxplot("Gene A", lookup_col="gene_name", meta_field="cell_id")
        >>> dataset.plot_gene_boxplot("5270") // dsitribution for gene_id == '5270'
 
        Returns
        -------
        ``altair.Chart`` object
        """
        if lookup_col:
            gene_mask = self.gene_meta[lookup_col] == str(identifier)
        else:
            gene_mask = self.gene_meta.index.astype(str) == str(identifier)
            
        gene_index = self.gene_meta[gene_mask].index[0]
        df = self._data[[gene_index, *self.sample_meta.columns]]
        df = df.rename(columns={gene_index: identifier})
        return boxplot(df=df, x=meta_field, y=df.columns[0], extent=extent)

    def plot_meta_counts(self, meta_field, normalize=False, sort_values=True):
        """Returns a barplot of a metadata field counts in Dataset
        
        Parameters
        ----------
        ``meta_field`` : ``str``
                Valid sample metadata column.

       ``normalize`` : ``bool`` (optional, default ``False``)
                Whether to show counts or noramlize to frequencies.
                           
        ``sort_values`` : ``bool`` (optional, default ``True``)
                Whether to sort barchart by counts/frequencies.
                
        >>> dataset.plot_meta_counts("cell_id", normalize=True) // barplot of cell_id frequencies
                
        Returns
        -------
        ``altair.Chart`` object
        """
        counts = self.sample_meta[meta_field].value_counts(normalize=normalize)
        colname = "counts" if normalize is False else "frequency"
        df = pd.DataFrame({meta_field: counts.index.values, colname: counts.values})
        return barplot(df=df, x=meta_field, y=colname)
    
    def copy(self):
        """Copies Dataset to a new object"""
        return Dataset(self._data.copy(), self.gene_meta.copy(), self.n_genes)
    
    def _copy(self, data):
        return Dataset(data, self.gene_meta.copy(), self.n_genes)

    def __len__(self):
        return self._data.shape[0]

    def __repr__(self):
        nsamples, ngenes = self.data.shape
        return f"<L1000 Dataset: (samples: {nsamples:,}, genes: {ngenes:,})>"


class KerasDataset(Dataset):
    """ Represents an L1000 Dataset to be injested into Keras pipeline    
        
    Parameters
    ----------
    ``data`` : ``dataframe``, shape (n_samples, (n_genes + n_metadata_fields))
            A sample by gene expression matrix padded to the right with per sample metadata.
            
    ``gene_meta`` : ``dataframe``, shape (n_genes, n_features)
            Contains the metadata for each of the genes in the data matrix.

    ``n_genes`` : ``int``
            Number of genes in expression matrix. This explicitly defines 
            the column index which divides the expression values and metadata.
                
    ``name`` : ``str``
            Identifier for what type of dataset (``'train'``, ``'validation'``, ``'test'``)
        
    Properties
    ----------
    ``data`` : ``dataframe``, shape (n_samples, n_genes)
            Gene expression matrix as a dataframe. Shared indicies with `self.sample_meta` and `self.gene_meta`.
    
    ``sample_meta`` : ``dataframe``, shape (n_samples, n_metadata_features)
            Per profile metadata. Row index same as ``self.data.index``.

    ``gene_meta`` : ``dataframe``, shape (n_genes, n_gene_features)
            Gene metadata. Row index same as ``self.data.columns``.
    """
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
        """Converts Dataset to tf.data.Dataset to be ingested by Keras

        Parameters
        ----------
        ``target`` : ``str``
                Valid sample metadata column or ``'self'``. If ``'self'``, the outputs are designed to
                be the same as the inputs (i.e. an autoencoder).
                
        ``batch_size`` : ``int`` (optional, default ``64``)
                Size of batches during training and testing.

        ``batch_normalize`` : ``str`` (optional, default ``None``)
            Whether to batch normalize. Can be one of ``'standard_scale'`` or ``'z_score'``.

        >>> keras_dataset("cell_id", batch_size=128) ==> TensorFlow prefetch-dataset.

        Returns
        -------
        tf_dataset : ``tensorflow.data.Dataset``
        """
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
        """Constructor from base Dataset class

        Parameters
        ----------
        ``name`` : ``str``
            Type of dataset. Must be one of ``'train'``, ``'validation'``, or ``'test'``.
        """
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
