settings.yaml
=============

The initial barrier in analyzing the LINCS L1000 dataset is the non-standardised 
structure of the data released to GEO. Although all levels of expression data are 
available in `GCTx format <https://www.biorxiv.org/content/10.1101/227041v1>`_, no 
metadata other than gene and profile identifiers are included. Instead, all 
metadata is kept in separate text files.

This decision was likely intended to keep file sizes smaller, avoiding redundancy,
but the current structure requires a significant data assembly step in order to filter
and subset samples. Cell- and perturbation-specific metadata are also kept in 
separate files, which require non-trivial key-value lookup operations to populate 
a per-profile metadata table.

The YAML specification for declares the hierarchical relational structure between the 
data provided by LINCS. The ``data`` field should specify a gene expression matrix, indicating 
the shared index names with ``gene_metadata`` and ``sample_metadata``. Within each of the metadata fields,
a ``main`` file should be specified which has a shared index with ``data``. Additional ``lookup`` data 
can be specified in which a ``lookup_key`` indicates a shared column with the ``main`` data.
This file is parsed via ``Dataset.from_yaml`` which assembles the data automatically in-memory.

Required fields: 
----------------
* ``data_dir``: path to directory of LINCS data
* ``data``: main expression dataset
    - ``gene_index_name``: shared index with gene metadata
    - ``sample_index_name``: shared index with sample metadata
* ``gene_metadata``/ ``sample_metadata``: gene and sample metadata text file(s)
    - ``main``: required metadata with shared indices 
        - ``name``: name of data
        - ``file``: file name
        - ``index_col``: index column shared with ``data``
    - ``lookup``: optional metadata files
        - ``name``: name of data
        - ``file``: file name
        - ``lookup_key``: shared key with ``main`` used to merge data

Additional fields in each file spec are used to specify any necessary keywords arguments 
(i.e. ``use_cols``, ``sep``, ``na_values``) used by ``pandas.read_csv`` to read the metadata. 
An example can be seen below.


.. code-block:: bash

    # settings.yaml

    data_dir: data/

    data:
        file: Level3_INF_mlr12k_n1319138x12328.gctx
        gene_index_name: gene_id
        sample_index_name: inst_id

    gene_metadata:
        main:
            name: gene_info
            file: gene_info.txt
            index_col: gene_id
            use_cols:
                - gene_id
                - gene_symbol 
            na_values: "-666"
            sep: "\t"

    sample_metadata:
        main:
            name: inst_info
            file: inst_info.txt
            index_col: inst_id
            usecols:
                - inst_id
                - cell_id
            na_values: "-666"
            sep: "\t"
        lookup: 
            name: cell_info
            file: cell_info.txt
            lookup_key: cell_id
            usecols: 
                - cell_id 
                - cell_type 
                - precursor_cell_id
                - sample_type 
                - primary_site 
                - subtype
            na_values: "-666"
            sep: "\t"
        lookup:
            name: pert_info
            file: pert_info.txt
            lookup_key: pert_id
            usecols:
                - pert_id
                - pert_type
                - inchi_key_prefix 
                - inchi_key
                - canonical_smile
            na_values: "-666"
            sep: "\t"
