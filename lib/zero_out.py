import pandas as pd
import numpy as np

def repeat_and_zero_diag(data_df):
    nsamples, ngenes = data_df.values.shape
    repeated = np.repeat(data_df.values, ngenes, axis=0)
    # fill diagonal without trailing space
    repeated[np.arange(repeated.shape[0]), np.tile(np.arange(ngenes), nsamples)] = 0
    return repeated


def zero_out_difference(data_df, encoder, prune_inactive):
    nsamples, ngenes = data_df.values.shape
    repeated = repeat_and_zero_diag(data_df)
    
    expected_h = encoder.predict(data_df.values)
    zero_out_h = encoder.predict(repeated)
    units=np.arange(expected_h.shape[1])
    
    if prune_inactive:
        mask = expected_h.sum(axis=0) >= 0.001
        expected_h = expected_h[:,mask]
        zero_out_h = zero_out_h[:,mask]
        units = units[mask]
    
    zero_out_h = zero_out_h.reshape(nsamples, ngenes, zero_out_h.shape[1])
    unit_names = [f"unit_{i}" for i in units]
    return (expected_h[:, np.newaxis] - zero_out_h) ** 2, unit_names


def zero_out_method(encoder, dataset, k=5, prune_inactive=True):
    diff_tensor, unit_names = zero_out_difference(dataset.data, encoder, prune_inactive)
    nsamples, ngenes, nunits = diff_tensor.shape

    kth_idxs = (-diff_tensor).argsort(1)[:, :k, :].reshape(nsamples * k, nunits)
    k_rank = np.tile(np.arange(k), nsamples)[:, np.newaxis]

    df = (
        pd.DataFrame(
            np.concatenate((dataset.data.columns.values[kth_idxs].astype(int), k_rank), axis=1),
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
    )

    return df