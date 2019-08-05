Dataset
=======

DeepLincs offers :class:`Dataset` to wrangle L1000 data.

Dataset
-------

.. currentmodule:: deep_lincs.dataset

.. autoclass:: Dataset
   
   .. automethod:: __init__
   
   .. rubric:: Methods

   .. autosummary::
   
      ~Dataset.__init__
      ~Dataset.copy
      ~Dataset.from_yaml
      ~Dataset.from_dataframes
      ~Dataset.sample_rows
      ~Dataset.filter_rows
      ~Dataset.select_meta
      ~Dataset.select_samples
      ~Dataset.split
      ~Dataset.dropna
      ~Dataset.set_categorical
      ~Dataset.normalize_by_gene
      ~Dataset.train_val_test_split
      ~Dataset.to_tsv
      ~Dataset.one_hot_encode
      ~Dataset.plot_gene_boxplot
      ~Dataset.plot_meta_counts
