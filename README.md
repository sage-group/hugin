# Hugin

Hugin helps scientists run Machine Learning experiments on geospatial raster data.

Overall Hugin aims to facilitate experimentation with multiple machine learning problems, like:

 - Classification
 - Segmentation
 - Super-Resolution

Currently Hugin builds on top of the Keras machine learning library but it also aims to support, in the future, additional backends like scikit-learn.

------------------

## Installation


### Prerequisits

As previously mentioned Hugin builds on top of Keras but, depending on your usecase it might also depend on TensorFlow 
for providing additional IO functionality (particularly cloud storage), or for supporting models specific for various Keras backends

### Using pip

#### From PyPi

`ToDo`

#### From source repository

You can install Hugin using the following command:
```bash
pip install git+http://github.com/mneagul/hugin#egg=hugin
```

### From source code

When installing from source code we recommend installation inside a specially created virtual environment.

Installing from source code involves running the `setup.py` inside you python environment.

```python
python setup.py install
```

## Using

Using Hugin involves two steps:
 - training
 - prediction

Both steps are driven using dedicated configuration files.

### Training

Training can be started as follows:

```bash
hugin train --config training_config.yaml
```

An example training configuration can be found in [docs/examples/train.yaml](docs/examples/train.yaml).

### Prediction

Prediction can be started as follows:

```bash
hugin predict \
    --ensemble-config prediction.yaml \
    --input-dir /path/to/input/dir \
    --output-dir /tmp/output
```

An example prediction configuration can be found in [docs/examples/predict.yaml](docs/examples/predict.yaml)

The `predict` command requires at least three arguments: 
 * `--ensemble-config`: representing the prediction configuration file
 * `--input-dir`: representing the directory holding data that should server as input for prediction
 * `--output-dir`: directory for storing the outputs (predictions)

## Developing your own models

Developing your own is really simple. The only thing needed is to create a python
file that creates the corresponding Keras model.

The code building the model needs to be resolvable by Hugin: it needs to be available in the `PYTHONPATH`.

Let's consider you are preparing a new segmentation related experiment.

The most simple approach would be to create a new directory containing both the source code and model configuration, like in 
the following example:

```
mysegmentation/
├── model.py
├── predict.yaml
└── train.yaml
```

The files involved in this example are:
* `model.py`: it contains the source code for the model. An example can be found in the source distribution, in [src/hugin/models/unet/unetv14.py](src/hugin/models/unet/unetv14.py)
* `train.yaml`: the configuration used for training. An example can be found in [docs/examples/train.yaml](docs/examples/train.yaml)
* `predict.yaml`: the configuration used for prediction. An example can be found in [docs/examples/predict.yaml](docs/examples/predict.yaml)

After creating your model and preparing your experiment configuration you can start training, by running:

```bash
hugin train --config train.yaml
```

As specified in the model configuration the model will train for `10` epochs and produce the final model.