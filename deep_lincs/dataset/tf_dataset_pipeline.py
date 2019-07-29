import tensorflow as tf
import pandas as pd


def normalize_X(tf_dataset, method):
    norm_options = {
        "z_score": tf_dataset.map(_zscore_by_gene),
        "standard_scale": tf_dataset.map(_standard_scale_by_gene),
        None: tf_dataset,
    }
    if method not in norm_options.keys():
        raise ValueError(
            f"The norm_method of '{method}' is not valid. "
            f"Must be one of {list(norm_options.keys())}."
        )
    return norm_options[method]


def _standard_scale_by_gene(X, y):
    X = X - tf.math.reduce_min(X, axis=0)
    return X / tf.math.reduce_max(X, axis=0), y


def _zscore_by_gene(X, y):
    X = (X - tf.math.reduce_mean(X, axis=0)) / tf.math.reduce_std(X, axis=0)
    return X, y


def prepare_tf_dataset(
    dataset, batch_size, shuffle, repeat, batch_normalize, shuffle_buffer_size
):
    """Creates a tensorflow Dataset to be ingested by Keras."""
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        dataset = dataset.repeat()
    if batch_size:
        dataset = dataset.batch(batch_size)
    if batch_normalize:
        dataset = normalize_X(dataset, batch_normalize)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
