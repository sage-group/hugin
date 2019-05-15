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

An example training configuration can be found in the [train_s2_forestry.yaml](etc/usecases/s2-forestry/train_s2_forestry.yaml) configuration file.

### Prediction

Prediction can be started as follows:

```bash
hugin predict \
    --ensemble-config prediction.yaml \
    --input-dir /path/to/input/dir \
    --output-dir /tmp/output
```

An example prediction configuration can be found in the [predic_s2_forestry.yaml](etc/usecases/s2-forestry/predic_s2_forestry.yaml) configuration file.
