# DeepLINCS

A deep learning toolkit for [LINCS](http://www.lincsproject.org/) L1000 expression data.

Check out the documentation [here](https://deep-lincs.readthedocs.io).

## Installation
```bash
$ pip install deep-lincs
```

## Getting started
The data for 1.3 million L1000 profiles are availabe [on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742). The script load_files.sh fetches the `Level 3` data along with all metadata available. The largest file is quite big (~50Gb) so please be patient.

```bash
$ git clone https://github.com/manzt/deep_lincs.git && cd deep_lincs
$ source load_files.sh # download raw data from GEO
$ cd notebooks 
$ jupyter lab # get started in a notebook 
```

## L1000 Dataset
The `Dataset` class is built with a variety of methods to load, subset, filter, and combine expression and metadata. 
```python
from deep_lincs import Dataset

# Select samples 
cell_ids = ["VCAP", "MCF7", "PC3"]
pert_types = ["trt_cp", "ctl_vehicle", "ctl_untrt"]

# Loading a Dataset
dataset = Dataset.from_yaml("settings.yaml", cell_id=cell_ids, pert_type=pert_types)

# Normalizing the expression data
dataset.normalize_by_gene("standard_scale")

# Chainable methods
subset = dataset.sample_rows(5000).filter_rows(pert_id=["ctl_vehicle", "ctl_untrt"])
```

## Models
Models interface with the `Dataset` class to make training and evaluating different arcitectures simple.


### Single Classifier

```python
from deep_lincs.models import SingleClassifier

model = SingleClassifier(dataset, target="cell_id")
model.prepare_tf_datasets(batch_size=64)
model.compile_model(hidden_layers=[128, 128, 64, 32], dropout_rate=0.1)
model.fit(epochs=10)

model.evaluate() # Evaluates on isntance test Dataset
model.evaluate(subset) # Evalutates model on user-defined Dataset
```

### Multiple Classifier

```python
from deep_lincs.models import MutliClassifier

targets = ["cell_id", "pert_type"]
model = MutliClassifier(dataset, target=targets)
model.prepare_tf_datasets(batch_size=64)
model.compile_model(hidden_layers=[128, 128, 64, 32])
model.fit(epochs=10)

model.evaluate() # Evaluates on isntance test Dataset
model.evaluate(subset) # Evalutates model on user-defined Dataset
```

### Autoencoder

```python
from deep_lincs.models import AutoEncoder

model = AutoEncoder(dataset)
model.prepare_tf_datasets(batch_size=64)
model.compile_model(hidden_layers=[128, 32, 128], l1_reg=0.01)
model.fit(epochs=10)

model.encoder.predict() # Gives encodings for instance test Dataset
model.encoder.predict(subset) # Gives encodings for user-defined Dataset
```
