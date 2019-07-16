import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input, layers
from tensorflow.keras.metrics import Metric
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import altair as alt


class PearsonsR(Metric):
    def __init__(self, name="pearsons_corrcoef", **kwargs):
        super(PearsonsR, self).__init__(name=name, **kwargs)
        self.corrcoef = self.add_weight(name="cc", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mean_true = tf.math.reduce_mean(y_true)
        mean_pred = tf.math.reduce_mean(y_pred)

        diff_true = y_true - mean_true
        diff_pred = y_pred - mean_pred

        numer = tf.math.reduce_sum(diff_true * diff_pred)
        denom = tf.math.sqrt(tf.math.reduce_sum(diff_true ** 2)) * tf.math.sqrt(
            tf.math.reduce_sum(diff_pred ** 2)
        )
        self.corrcoef.assign(numer / denom)

    def result(self):
        return self.corrcoef

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.corrcoef.assign(0.0)


class LincsNN:
    def __init__(self):
        pass

    def compile_model(self):
        pass

    def fit(self, epochs=5, batch_size=32, shuffle=True):
        self.model.fit(
            self.train.to_tf_dataset(target=self.target, batch_size=batch_size),
            epochs=epochs,
            shuffle=shuffle,
            steps_per_epoch=len(self.train) // batch_size,
            validation_data=self.val.to_tf_dataset(
                target=self.target, batch_size=batch_size, shuffle=False, repeated=False
            ),
        )

    def evaluate(self, batch_size=32):
        return self.model.evaluate(
            self.val.to_tf_dataset(
                target=self.target, batch_size=batch_size, shuffle=False, repeated=False
            )
        )

    def summary(self):
        return self.model.summary()


class SingleClassifier(LincsNN):
    def __init__(self, dataset, target, normalize_by_gene=True, p1=0.2, p2=0.2):
        self.target = target
        self.in_size, self.out_size = self._get_in_out_size(dataset, target)
        if normalize_by_gene:
            dataset.normalize_by_gene()
        self.train, self.val, self.test = dataset.train_val_test_split(p1, p2)

    def compile_model(
        self, hidden_layers, dropout_rate=0.0, activation="relu", optimizer="adam"
    ):
        model = Sequential()
        model.add(layers.Dropout(dropout_rate, input_shape=(self.in_size,)))

        for nunits in hidden_layers:
            model.add(layers.Dense(nunits, activation=activation))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.out_size, activation="softmax"))

        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model = model

    def predict(self, as_df=False):
        output = self.model.predict(self.test.data)
        if as_df:
            colnames = [f"unit_{i}" for i in range(self.out_size)]
            output = pd.DataFrame(output, columns=colnames, index=self.test.data.index)
        return output

    def plot_confusion_matrix(
        self, normalize=True, zero_diag=False, size=300, color_scheme="viridis"
    ):
        y_dummies = pd.get_dummies(self.test.sample_meta[self.target])
        classes = y_dummies.columns.values
        y_test = y_dummies.values

        y_pred = self.predict()

        cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
        if zero_diag:
            np.fill_diagonal(cm, 0)
        if normalize:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        df = (
            pd.DataFrame(cm, columns=classes, index=classes)
            .reset_index()
            .melt(id_vars="index")
        )

        base = alt.Chart(df).encode(
            x=alt.X("index", title=None), y=alt.Y("variable", title=None)
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("value", scale=alt.Scale(scheme=color_scheme))
        )

        text = base.mark_text(size=size * 0.05).encode(
            text=alt.Text("value", format=".2")
        )

        return (heatmap + text).properties(width=size, height=size)

    def _get_in_out_size(self, dataset, target):
        unique_targets = dataset.sample_meta[target].unique().tolist()
        if np.nan in unique_targets:
            raise Exception(
                f"Dataset contains np.nan entry in '{target}'. "
                f"You can drop these samples to train the "
                f"classifier with LINCSDataset.drop_na('{target}')."
            )
        in_size = dataset.data.shape[1]
        out_size = len(unique_targets)
        return in_size, out_size

    def __repr__(self):
        return (
            f"<SingleClassifier: "
            f"(target: '{self.target}', "
            f"input_size: {self.in_size}, "
            f"output_size: {self.out_size})>"
        )


class AutoEncoder(LincsNN):
    def __init__(self, dataset, normalize_by_gene=True, p1=0.2, p2=0.2):
        self.target = "self"
        self.in_size = dataset.data.shape[1]
        self.out_size = dataset.data.shape[1]
        if normalize_by_gene:
            dataset.normalize_by_gene()
        self.train, self.val, self.test = dataset.train_val_test_split(p1, p2)

    def compile_model(
        self, hidden_layers, dropout_rate=0.0, activation="relu", optimizer="adam"
    ):
        model = Sequential()
        model.add(layers.Dropout(dropout_rate, input_shape=(self.out_size,)))
        for nunits in hidden_layers:
            model.add(layers.Dense(nunits, activation=activation))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.out_size, activation="relu"))

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[
                tf.keras.metrics.CosineSimilarity(),
                PearsonsR(),  # custom correlation metric
            ],
        )
        self.model = model

    def __repr__(self):
        return (
            f"<AutoEncoder: "
            f"(input_size: {self.in_size}, "
            f"output_size: {self.out_size})>"
        )


class MultiClassifier(SingleClassifier):
    def __init__(self, dataset, targets, normalize_by_gene=True, p1=0.2, p2=0.2):
        self.target = targets
        self.in_size, self.out_size = self._get_in_out_size(dataset, targets)
        if normalize_by_gene:
            dataset.normalize_by_gene()
        self.train, self.val, self.test = dataset.train_val_test_split(p1, p2)

    def compile_model(
        self,
        hidden_layers,
        dropout_rate=0.0,
        activation="relu",
        optimizer="adam",
        final_activation="softmax",
    ):
        model = Sequential()
        inputs = Input(shape=(self.in_size,))

        x = layers.Dropout(dropout_rate)(inputs)
        for nunits in hidden_layers:
            x = layers.Dense(nunits, activation=activation)(x)
            x = layers.Dropout(dropout_rate)(x)

        outputs = [
            layers.Dense(size, activation=final_activation, name=name)(x)
            for name, size in self.target_info.items()
        ]

        model = Model(inputs, outputs)

        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model

    def _get_in_out_size(self, dataset, targets):
        self.target_info = {}
        for target in targets:
            unique_targets = dataset.sample_meta[target].unique().tolist()
            if np.nan in unique_targets:
                raise Exception(
                    f"Dataset contains np.nan entry in '{target}'. "
                    f"You can drop these samples to train the "
                    f"classifier with LINCSDataset.drop_na('{target}')."
                )
            self.target_info[target] = len(unique_targets)
        in_size = dataset.data.shape[1]
        out_size = sum(self.target_info.values())
        return in_size, out_size

    def plot_confusion_matrix(
        self, normalize=True, zero_diag=False, size=300, color_scheme="viridis"
    ):
        y_dummies = [pd.get_dummies(self.test.sample_meta[t]) for t in self.target]
        y_pred = self.predict()

        heatmaps = [
            self._create_heatmap(d, p, normalize, zero_diag, size, color_scheme)
            for d, p in zip(y_dummies, y_pred)
        ]
        return alt.hconcat(*heatmaps)

    def _create_heatmap(
        self, y_dummies, y_pred, normalize, zero_diag, size, color_scheme
    ):
        y_test = y_dummies.values
        classes = y_dummies.columns.values

        cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
        if zero_diag:
            np.fill_diagonal(cm, 0)
        if normalize:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        df = (
            pd.DataFrame(cm, columns=classes, index=classes)
            .reset_index()
            .melt(id_vars="index")
        )

        base = alt.Chart(df).encode(
            x=alt.X("index", title=None), y=alt.Y("variable", title=None)
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("value", scale=alt.Scale(scheme=color_scheme))
        )

        text = base.mark_text(size=size * 0.05).encode(
            text=alt.Text("value", format=".2")
        )

        return (heatmap + text).properties(width=size, height=size)

    def __repr__(self):
        return (
            f"<MultiClassifier: "
            f"(targets: {self.target}, "
            f"input_size: {self.in_size}, "
            f"output_size: {self.out_size})>"
        )
