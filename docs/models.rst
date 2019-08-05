.. _models:

Extendable Models
=================

DeepLincs offers :class:`AutoEncoder`, :class:`MultiClassifier`, 
:class:`SingleClassifier`, as simple high-level APIs for building 
and training a deep neural network on a L1000 :class:`Dataset`. 

While the networks defined for each model above are simple, 
each of these object can be subclassed, allowing for the user to 
override the ``compile_model`` method and build and more complicated 
model for an identical task. For example, here is a 
`self normalizing classifier <https://keras.io/examples/reuters_mlp_relu_vs_selu/>`_.

.. code-block:: python

   from deep_lincs.models import SingleClassifier
   from tensorflow.keras.models import Model
   from tensorflow.keras.layers import Dense, Activation
   from tensorflow.keras.layers.noise import AlphaDropout

   class SelfNormalizingClassifier(SingleClassifier):
        # overrides SingleClassifier method
        def compile_model(hidden_layers, dropout_rate, opt="adam"):
            inputs = Input(shape=(self.in_size,))
            dense1 = hidden_layers.pop(0)
            x = Dense(dense1, kernel_initializer="lecun_normal")(inputs)
            x = Acitvation("selu")(x)
            x = AlphaDropout(dropout_rate)(x)

            for h in hidden_layers:
                x = Dense(h, kernel_initializer="lecun_normal")(x)
                x = Activation("selu")(x)
                x = AlphaDropout(dropout_rate)(x)

            outputs = Dense(self.out_size, activation="softmax")(x)
            model = Model(inputs, outputs)
            model.compile(
                loss='categorical_crossentropy', 
                optimizer=opt, 
                metrics=['accuracy']
            )
            # set model attribute
            self.model = model

The subsequent code to train this model with a :class:`Dataset` is included below. 
All :class:`deep_lincs.models` follow the same order of method calls.

.. code-block:: python

    snc = SelfNormalizingClassifier(dataset, target="subtype")
    snn.prepare_tf_datasets(batch_size=128)
    snn.compile_model([128, 128, 128], dropout_rate=0.15)
    snn.fit(epochs=20)
    

AutoEncoder
-----------
.. currentmodule:: deep_lincs.models

.. autoclass:: AutoEncoder
   
   .. automethod:: __init__
   
   .. rubric:: Methods

   .. autosummary::
   
      ~AutoEncoder.__init__
      ~BaseNetwork.prepare_tf_datasets
      ~AutoEncoder.compile_model
      ~BaseNetwork.fit
      ~BaseNetwork.save
      ~BaseNetwork.summary


MultiClassifier
---------------
.. currentmodule:: deep_lincs.models

.. autoclass:: MultiClassifier
   
   .. automethod:: __init__
   
   .. rubric:: Methods

   .. autosummary::
   
      ~MultiClassifier.__init__
      ~BaseNetwork.prepare_tf_datasets
      ~MultiClassifier.compile_model
      ~BaseNetwork.fit
      ~BaseNetwork.evaluate
      ~BaseNetwork.predict
      ~BaseNetwork.save
      ~BaseNetwork.summary
      ~MultiClassifier.plot_confusion_matrix


SingleClassifier
----------------
.. currentmodule:: deep_lincs.models

.. autoclass:: SingleClassifier
   
   .. automethod:: __init__
   
   .. rubric:: Methods

   .. autosummary::
   
      ~SingleClassifier.__init__
      ~BaseNetwork.prepare_tf_datasets
      ~SingleClassifier.compile_model
      ~BaseNetwork.fit
      ~BaseNetwork.evaluate
      ~BaseNetwork.predict
      ~BaseNetwork.save
      ~BaseNetwork.summary
      ~SingleClassifier.plot_confusion_matrix
