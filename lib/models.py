import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.metrics import Metric


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


def create_AE(hidden_layers, activation="relu", optimizer="adam", out_size=978):
    model = Sequential()
    model.add(
        layers.Dense(hidden_layers[0], activation=activation, input_shape=(out_size,))
    )

    for nunits in hidden_layers[1:]:
        model.add(layers.Dense(nunits, activation=activation))

    model.add(layers.Dense(out_size, activation="relu"))

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[
            tf.keras.metrics.CosineSimilarity(),
            PearsonsR(),  # custom correlation metric
        ],
    )

    return model
