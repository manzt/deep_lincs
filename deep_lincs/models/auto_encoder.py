from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Metric, CosineSimilarity

import pandas as pd
import numpy as np
import altair as alt

from ..hidden_embedding import HiddenEmbedding
from .base_network import BaseNetwork
from .metrics import PearsonsR


class AutoEncoder(BaseNetwork):
    """Represents an simple autoencoder
    
    Parameters
    ----------
    dataset : ``Dataset``
            An instance of a ``Dataset`` intended to train and evaluate a model.
            
    test_sizes : tuple, (optional, default ( ``0.2`` , ``0.2`` ))
            Size of test splits for dividing the dataset into training, validation, and, testing

    Attributes
    ----------
    
    encoder : ``tensorflow.keras.Model``
            Encoder for the AutoEncoder model. 
            
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
            Same as input size since model is an autoencoder.
    """
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
        """Defines how model is built and compiled

        Parameters
        ----------
        hidden_layers : ``list(int)``
                A list describing the size of the hidden layers.

        dropout_rate : ``float`` (optional: default ``0.0``)
                Dropout rate used during training. Applied to all hidden layers.

        activation : ``str``, (optional: default ``"relu"``)
                Activation function used in hidden layers.
                
        final_activation : ``str`` (optional: default ``"softmax"``)
                Activation function used in final layer.
                
        optimizer : ``str``, (optional: default ``"adam"``)
                Optimizer used during training.
                
        l1_reg : ``float`` (optional: default ``None``)
                Level of L1 regularization applied to the hidden embedding (smallest hidden layer).

        Returns
        -------
                ``None``
        """
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
