import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input, layers, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Metric, CosineSimilarity
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import altair as alt

from .hidden_embedding import HiddenEmbedding


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


class BaseNetwork:
    def __init__(self, dataset, target, test_sizes=(0.2, 0.2)):
        self.target = target
        self.train, self.val, self.test = dataset.train_val_test_split(*test_sizes)
        self.model = None
        self._dataset_preprocessed = False

    def prepare_tf_datasets(self, batch_size, batch_normalize=None):
        self._batch_size = batch_size
        self.train_dset, self.val_dset, self.test_dset = [
            lincs_dset(self.target, batch_size, batch_normalize)
            for lincs_dset in [self.train, self.val, self.test]
        ]
        self._dataset_preprocessed = True

    def compile_model(self):
        pass

    def fit(self, epochs=5, shuffle=True, verbose=1, callbacks=None):
        if self._dataset_preprocessed is False:
            raise ValueError(
                f"Data has not been prepared for training. "
                f"Run {self.__class__.__name__}.prepare_tf_datasets()."
            )
        if self.model is None:
            raise ValueError(
                f"Model has not been created. "
                f"Run the {self.__class__.__name__}.compile_model() method before training."
            )
        self.model.fit(
            self.train_dset,
            epochs=epochs,
            shuffle=shuffle,
            steps_per_epoch=len(self.train) // self._batch_size,
            validation_data=self.val_dset,
            verbose=verbose,
            callbacks=callbacks
        )

    def evaluate(self, inputs=None):
        if inputs is None:
            return self.model.evaluate(self.test_dset)
        else:
            return self.model.evaluate(inputs)

    def predict(self, inputs=None):
        if inputs is None:
            return self.model.predict(self.test_dset)
        else:
            return self.model.predict(inputs)
    
    def save(self, file_name):
        self.model.save(file_name)

    def summary(self):
        return self.model.summary()


class SingleClassifier(BaseNetwork):
    def __init__(self, dataset, target, **kwargs):
        dataset._data[target] = pd.Categorical(dataset._data[target])
        super(SingleClassifier, self).__init__(dataset=dataset, target=target, **kwargs)
        self.in_size, self.out_size = self._get_in_out_size(dataset, target)

    def compile_model(
        self, hidden_layers, dropout_rate=0.0, activation="relu", optimizer="adam"
    ):
        model = Sequential()
        model.add(Dropout(dropout_rate, input_shape=(self.in_size,)))

        for nunits in hidden_layers:
            model.add(Dense(nunits, activation=activation))
            model.add(Dropout(dropout_rate))

        model.add(Dense(self.out_size, activation="softmax"))

        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model = model

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
                f"classifier with Dataset.drop_na('{target}')."
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


class MultiClassifier(SingleClassifier):
    def __init__(self, dataset, targets, **kwargs):
        for target in targets:
            dataset._data[target] = pd.Categorical(dataset._data[target])
        super(MultiClassifier, self).__init__(dataset=dataset, target=targets, **kwargs)
        self.in_size, self.out_size = self._get_in_out_size(dataset, targets)

    def compile_model(
        self,
        hidden_layers,
        dropout_rate=0.0,
        activation="relu",
        optimizer="adam",
        final_activation="softmax",
    ):
        inputs = Input(shape=(self.in_size,))

        x = Dropout(dropout_rate)(inputs)
        for nunits in hidden_layers:
            x = Dense(nunits, activation=activation)(x)
            x = Dropout(dropout_rate)(x)

        outputs = [
            Dense(size, activation=final_activation, name=name)(x)
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
                    f"classifier with Dataset.drop_na('{target}')."
                )
            self.target_info[target] = len(unique_targets)
        in_size = dataset.data.shape[1]
        out_size = sum(self.target_info.values())
        return in_size, out_size

    def plot_confusion_matrix(
        self, normalize=True, zero_diag=False, size=300, color_scheme="lightgreyteal"
    ):
        y_dummies = [pd.get_dummies(self.test.sample_meta[t]) for t in self.target]
        y_pred = self.predict()

        heatmaps = [
            self._create_heatmap(d, p, normalize, zero_diag, size, color_scheme, title)
            for d, p, title in zip(y_dummies, y_pred, self.target)
        ]
        return alt.hconcat(*heatmaps)

    def _create_heatmap(
        self, y_dummies, y_pred, normalize, zero_diag, size, color_scheme, title
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
            x=alt.X("index:N", title="Predicted Label"),
            y=alt.Y("variable:N", title="True Label"),
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("value:Q", scale=alt.Scale(scheme=color_scheme))
        )

        text = base.mark_text(size=size * 0.05).encode(
            text=alt.Text("value:Q", format=".2")
        )

        return (heatmap + text).properties(width=size, height=size, title=title)

    def __repr__(self):
        return (
            f"<MultiClassifier: "
            f"(targets: {self.target}, "
            f"input_size: {self.in_size}, "
            f"output_size: {self.out_size})>"
        )


class AutoEncoder(BaseNetwork):
    def __init__(self, dataset, **kwargs):
        super(AutoEncoder, self).__init__(dataset=dataset, target="self", **kwargs)
        self.in_size = dataset.data.shape[1]
        self.out_size = dataset.data.shape[1]
        self._h_name = "hidden_embedding"

    def compile_model(
        self,
        hidden_layers,
        dropout_rate=0.0,
        activation="relu",
        final_activation="relu",
        optimizer="adam",
        l1_reg=None,
    ):
        hsize = AutoEncoder._get_hidden_size(hidden_layers)
        inputs = Input(shape=(self.in_size,))
        x = Dropout(dropout_rate)(inputs)
        for nunits in hidden_layers:
            if nunits is hsize:
                l1_reg = regularizers.l1(l1_reg) if l1_reg else None
                x = Dense(
                    nunits,
                    activation=activation,
                    activity_regularizer=l1_reg,
                    name=self._h_name,
                )(x)
            else:
                x = Dense(nunits, activation=activation)(x)
            x = Dropout(dropout_rate)(x)

        outputs = Dense(self.out_size, activation=final_activation)(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[CosineSimilarity(), PearsonsR()],  # custom correlation metric
        )
        self.model = model

    @staticmethod
    def _get_hidden_size(hidden_layers):
        min_size = min(hidden_layers)
        num_min = len([size for size in hidden_layers if size == min_size])
        if num_min is not 1:
            raise ValueError(
                f"Auto encoder does not contain bottleneck. "
                f"Make sure there is a single minimum in hidden layers: {hidden_layers}."
            )
        return min_size
    
    def get_hidden_embedding(self, lincs_dset=None):
        if lincs_dset is None:
            return HiddenEmbedding(self.test, self.encoder)
        else:
            return HiddenEmbedding(lincs_dset, self.encoder)
        
    @property
    def encoder(self):
        return Model(
            inputs=self.model.layers[0].input,
            outputs=self.model.get_layer(self._h_name).output,
        )

    def __repr__(self):
        return (
            f"<AutoEncoder: "
            f"(input_size: {self.in_size}, "
            f"output_size: {self.out_size})>"
        )
