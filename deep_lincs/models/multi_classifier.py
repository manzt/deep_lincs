import pandas as pd
import numpy as np
import altair as alt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

from .base_network import BaseNetwork


class MultiClassifier(BaseNetwork):
    """Represents a classifier for multiple metadata fields
    
    Parameters
    ----------
    dataset : ``Dataset``
            An instance of a ``Dataset`` intended to train and evaluate a model.
            
    targets : ``list(str)``
            Valid lists of metadata fields which define multiple classification tasks.
            
    test_sizes : tuple, (optional, default ( ``0.2`` , ``0.2`` ))
            Size of test splits for dividing the dataset into training, validation, and, testing

    Attributes
    ----------
    targets : ``list(str)``
            Targets for model.
            
    train : ``Dataset``
            Dataset used to train the model.
    
    val : ``Dataset``
            Dataset used during training as validation.
    
    test : ``Dataset``
            Dataset used to evaluate the model.
            
    model : ``tensorflow.keras.Model``
            Compiled and trained model.
            
    in_size : ``int`` 
            Size of inputs (generally 978 for L1000 landmark genes).
    
    out_size : ``int``
            Sum total of classification categories.
    """
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
        """Defines how model is built and compiled

        Parameters
        ----------
        hidden_layers : ``list(int)``
                A list describing the size of the hidden layers.

        dropout_rate : ``float`` (optional: default ``0.0``)
                Dropout rate used during training. Applied to all hidden layers.

        activation : ``str``, (optional: default ``"relu"``)
                Activation function used in hidden layers.

        optimizer : ``str``, (optional: default ``"adam"``)
                Optimizer used during training.
                
        final_activation : ``str`` (optional: default ``"softmax"``)
                Activation function used in final layer.
                
        loss : ``str`` (optional: default ``"categorical_crossentropy"``)
                Loss function.

        Returns
        -------
                ``None``
        """
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
            optimizer=optimizer, loss=loss, metrics=["accuracy"]
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
        """Evaluates model and plots a confusion matrix of classification results

        Parameters
        ----------
        normalize : ``bool``, (optional: default ``True``)
                Whether to normalize counts to frequencies.

        zero_diag : ``bool`` (optional: default ``False``)
                Whether to zero the diagonal of matrix. Useful for examining which categories 
                are most frequently misidenitfied.

        size : ``int``, (optional: default ``300``)
                Size of the plot in pixels.

        color_scheme : ``str``, (optional: default ``"lightgreyteal"``)
                Color scheme in heatmap. Can be any from https://vega.github.io/vega/docs/schemes/.

        Returns
        -------
                ``altair.Chart`` object
        """
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
