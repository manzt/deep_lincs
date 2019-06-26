import os
import csv
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from lib.utils import load_data, split_data, create_tf_dataset
from lib.models import create_AE

## Params
pert_types = [
    "trt_cp",  # treated with compound
    "ctl_vehicle",  # control for compound treatment (e.g DMSO)
    "ctl_untrt",  # untreated samples
]
cell_ids = ["VCAP", "MCF7", "PC3"]  # prostate tumor  # breast tumor  # prostate tumor
n_epochs = 10 
batch_size = 64  # for training
recorded_samples = 1000  # number of samples to record to csv
n_replicates = 4
h_sizes = [2 ** n for n in range(1, 8)]  # 2, 4, ..., 128

## Data Path
data_dir = "../data/GSE92742_Broad_LINCS"
data_fname = (
    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
)  # Level 3 data
data_path = os.path.join(data_dir, data_fname)

sample_meta_fname = "GSE92742_Broad_LINCS_inst_info.txt"
sample_meta_path = os.path.join(data_dir, sample_meta_fname)

out_dir = "../data/dark_units"


def h_to_df(hidden_output, meta_data):
    col_names = [f"unit_{unit}" for unit in range(hidden_output.shape[1])]
    df = pd.concat(
        (meta_data.reset_index(), pd.DataFrame(hidden_output, columns=col_names)),
        axis=1,
    )
    return df


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"tensorflow: {tf.__version__}")
    print(f"keras: {tf.keras.__version__}\n")

    print(f"cell ids: {cell_ids}")
    print(f"pert types: {pert_types}\n")

    print(f"Loading data...")
    data, sample_meta, gene_ids = load_data(
        data_path, sample_meta_path, pert_types, cell_ids
    )

    # Normalize expression between 0-1 per gene
    # TODO: implement this normalization per batch during training
    data_normed = data / data.max(0)
    print(f"data size: {data.shape[0]:,}")
    print(f"n_genes: {data.shape[1]}\n")

    print("Splitting data into training, validation, and testing")
    train, val, test = split_data(data_normed, sample_meta, 0.2)
    print(f"training size:   {train[0].shape[0]:,}")
    print(f"validation size: {val[0].shape[0]:,}")
    print(f"testing size:    {test[0].shape[0]:,}\n")

    # Create tensorflow Dataset from training data
    train_dataset = create_tf_dataset(
        train[0].values, train[0].values, batch_size=batch_size
    )
    
    # Tensorboard stuff
    logdir = os.path.join("logs", f"darkunits_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    
    # Train models and evaluate
    rows = []
    for h_size in h_sizes:
        for rep in range(n_replicates):

            model = create_AE([128, h_size, 128])
            print(model.summary())

            model.fit(
                train_dataset,
                epochs=n_epochs,
                shuffle=True,
                steps_per_epoch=train[0].shape[0] // batch_size,
                validation_data=(val[0].values, val[0].values),
                callbacks=[tensorboard_callback]
            )

            encoder = Model(
                inputs=model.layers[0].input, outputs=model.layers[1].output
            )
            h = encoder.predict(test[0])
            df = h_to_df(h, test[1])
            df.sample(recorded_samples).to_csv(
                os.path.join(out_dir, f"{h_size}_{rep}.csv"), index=False
            )
            
            model_eval = model.evaluate(test[0].values, test[0].values)

            row = {
                "n_hidden": h_size,
                "replicate": rep,
                "n_inactive": np.sum(h.sum(axis=0) < 0.01),
                "loss": model_eval[0],
                "cosine_sim": model_eval[1],
                "pearsons": model_eval[2]
            }
            rows.append(row)

    with open("dark_unit_summary.csv", "w") as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=["n_hidden", "replicate", "n_inactive", "loss", "cosine_sim", "pearsons"]
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)
