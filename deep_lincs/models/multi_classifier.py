import pandas as pd
import numpy as np
import altair as alt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

from .base_network import BaseNetwork


class MultiClassifier(BaseNetwork):
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
        classes = y_dummies.columns.tolist()
        y_test = y_dummies.values
        cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
        if zero_diag:
            np.fill_diagonal(cm, 0)
        if normalize:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]

        df = (
            pd.DataFrame(cm.round(2), columns=classes, index=classes)
            .reset_index()
            .melt(id_vars="index")
            .round(2)
        )

        base = alt.Chart(df).encode(
            x=alt.X("index:N", title="Predicted Label"),
            y=alt.Y("variable:N", title="True Label"),
            tooltip=["value"],
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("value:Q", scale=alt.Scale(scheme=color_scheme))
        )
        text = base.mark_text(size=0.5 * (size / len(classes))).encode(
            text=alt.Text("value")
        )

        return (heatmap + text).properties(width=size, height=size, title=title)

    def __repr__(self):
        return (
            f"<MultiClassifier: "
            f"(targets: {self.target}, "
            f"input_size: {self.in_size}, "
            f"output_size: {self.out_size})>"
        )
