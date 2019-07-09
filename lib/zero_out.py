import pandas as pd
import numpy as np


def repeat_and_zero_diag(data_df):
    nsamples, ngenes = data_df.values.shape
    repeated = np.repeat(data_df.values, ngenes, axis=0)
    # fill diagonal without trailing space
    repeated[np.arange(repeated.shape[0]), np.tile(np.arange(ngenes), nsamples)] = 0
    return repeated


def zero_out_difference(data_df, encoder):
    nsamples, ngenes = data_df.values.shape
    repeated = repeat_and_zero_diag(data_df)

    expected_h = encoder.predict(data_df.values)
    zero_out_h = encoder.predict(repeated)
    units = np.arange(expected_h.shape[1])

    zero_out_h = zero_out_h.reshape(nsamples, ngenes, zero_out_h.shape[1])
    return (expected_h[:, np.newaxis] - zero_out_h) ** 2


def zero_out_method(encoder, dataset, k=5):
    diff_tensor = zero_out_difference(dataset.data, encoder)
    nsamples, ngenes, nunits = diff_tensor.shape

    kth_idxs = (-diff_tensor).argsort(1)[:, :k, :]
    kth_genes = dataset.data.columns.values[kth_idxs].astype(int)

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
            np.concatenate((kth_genes.reshape(nsamples * k, nunits), k_rank), axis=1),
            columns=unit_names + ["k_rank"],
            index=dataset.data.index.repeat(k),
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

    return df
