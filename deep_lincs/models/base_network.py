class BaseNetwork:
    """A wrapper to train and evaluate a Keras model on a Dataset. 

    Parameters
    ----------
    dataset : ``Dataset``
            An instance of a ``Dataset`` intended to train and evaluate a model.
            
    target : ``str``
            Valid metadata field or "self". Defines classification task or whether model in an autoencoder.
            
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
    """
    def __init__(self, dataset, target, test_sizes=(0.2, 0.2)):
        self.target = target
        self.train, self.val, self.test = dataset.train_val_test_split(*test_sizes)
        self.model = None
        self._dataset_preprocessed = False

    def prepare_tf_datasets(self, batch_size, batch_normalize=None):
        """Defines how to prepare a prefetch dataset for training and model evaluation

        Parameters
        ----------
        batch_size : ``int``
                Batch size during training and for model evaluation. 
        
        batch_normalize : ``str`` (default: ``None``)
                Normalization applied to each batch during training and evaluation.
                Can be one of ``"z_score"`` or ``"standard_scale"``. Default is ``None``.
                
        Returns
        -------
                ``None``
        
        >>> model.prepare_tf_datasets(batch_size=128)
        """
        self._batch_size = batch_size
        self.train_dset, self.val_dset, self.test_dset = [
            lincs_dset(self.target, batch_size, batch_normalize)
            for lincs_dset in [self.train, self.val, self.test]
        ]
        self._dataset_preprocessed = True

    def compile_model(self):
        pass

    def fit(self, epochs=5, shuffle=True, **kwargs):
        """Trains model on training dataset

        Parameters
        ----------
        epochs : ``int``
                Number of training epochs 
        
        shuffle : ``bool`` (default: ``True``)
                Whether to shuffle batches during training.
        
        kwargs : (optional)
                Additional keyword arguments for ``tensorflow.keras.model.fit``. 
                This is where ``tensorflow.keras.callbacks`` should be supplied, such
                as Tensorboard or EarlyStopping.
                
        Returns
        -------
                ``None``
        """
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
            **kwargs,
        )

    def evaluate(self, inputs=None):
        """Evaluates model

        Parameters
        ----------
        inputs : ``tensorflow.data.dataset``, (optional: default ``None``)
                If no tf.dataset is provided, the model is evaluated on internal 
                test dataset.
                
        Returns
        -------
                ``list`` of evalutation metrics.
        """
        if inputs is None:
            return self.model.evaluate(self.test_dset)
        else:
            return self.model.evaluate(inputs)

    def predict(self, inputs=None):
        """Feeds inputs forward through the network

        Parameters
        ----------
        inputs : ``tensorflow.data.dataset`` or ``array`` or ``dataframe``, (optional: default ``None``)
                Inputs fed through the network. If not provided, the model uses the 
                internal testing data to make a prediction.
                
        Returns
        -------
                ``array`` of final activations.
        """
        if inputs is None:
            return self.model.predict(self.test_dset)
        else:
            return self.model.predict(inputs)

    def save(self, file_name):
        """Saves model as hdf5

        Parameters
        ----------
        file_name : ``str``
                Name of output file.
                
        Returns
        -------
                ``None``
        """
        self.model.save(file_name)

    def summary(self):
        """Prints verbose summary of model
                
        Returns
        -------
                ``None``
        """
        return self.model.summary()
