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
    zero_out_h = zero_out_h.reshape(nsamples, ngenes, zero_out_h.shape[1])
    return (expected_h[:, np.newaxis] - zero_out_h) ** 2


def zero_out_method(encoder, dataset, k=5):
    diff_tensor = zero_out_difference(dataset.data, encoder)
    nsamples, ngenes, nunits = diff_tensor.shape

    kth_idxs = (-diff_tensor).argsort(1)[:, :k, :].reshape(nsamples * k, nunits)
    k_rank = np.tile(np.arange(k), nsamples)[:, np.newaxis]

    units = [f"unit_{i}" for i in range(nunits)]

    df = (
        pd.DataFrame(
            np.concatenate((dataset.data.columns.values[kth_idxs].astype(int), k_rank), axis=1),
            columns=units + ["k_rank"],
            index=dataset.data.index.repeat(k),
        )
        .reset_index()
        .melt(
            id_vars=["inst_id", "k_rank"],
            value_vars=units,
            var_name="unit",
            value_name="gene_id",
        )
        .set_index("inst_id")
        .join(dataset.sample_meta["cell_id"])
    )

    return df