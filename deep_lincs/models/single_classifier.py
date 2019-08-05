import pandas as pd
import numpy as np
import altair as alt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

from .base_network import BaseNetwork


class SingleClassifier(BaseNetwork):
    """Represents a classifier for a single metadata field
    
    Parameters
    ----------
    dataset : ``Dataset``
            An instance of a ``Dataset`` intended to train and evaluate a model.
            
    target : ``str``
            Valid metadata field defining task for classification.
            
    test_sizes : tuple, (optional, default ( ``0.2`` , ``0.2`` ))
            Size of test splits for dividing the dataset into training, validation, and, testing

    Attributes
    ----------
    target : ``str``
            Target task of model.
            
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
            Total of target categories.
    """
    def __init__(self, dataset, target, **kwargs):
        dataset._data[target] = pd.Categorical(dataset._data[target])
        super(SingleClassifier, self).__init__(dataset=dataset, target=target, **kwargs)
        self.in_size, self.out_size = self._get_in_out_size(dataset, target)

    def compile_model(
        self, hidden_layers, dropout_rate=0.0, activation="relu", optimizer="adam"
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

        Returns
        -------
                ``None``
        """
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

        color_scheme : ``str``, (optional: default ``"viridis"``)
                Color scheme in heatmap. Can be any from https://vega.github.io/vega/docs/schemes/.

        Returns
        -------
                ``altair.Chart`` object
        """
        y_dummies = pd.get_dummies(self.test.sample_meta[self.target])
        classes = y_dummies.columns.tolist()
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
            .round(2)
        )

        base = alt.Chart(df).encode(
            x=alt.X("index", title=None),
            y=alt.Y("variable", title=None),
            tooltip=["value"],
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("value", scale=alt.Scale(scheme=color_scheme))
        )

        text = base.mark_text(size=0.5 * (size / len(classes))).encode(
            text=alt.Text("value")
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
