# -*- coding: utf-8 -*-
import tempfile

__license__ = \
    """Copyright 2019 West University of Timisoara
    
       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at
    
           http://www.apache.org/licenses/LICENSE-2.0
    
       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
    """

import getpass
import pickle
import socket
import time
from logging import getLogger

import os
import yaml
from numpy import random

from ..tools.IOUtils import IOUtils

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import numpy as np

from ..io import DataGenerator, ThreadedDataGenerator, CategoricalConverter

log = getLogger(__name__)


def train_sklearn(model_name,
                  window_size,
                  stride_size,
                  model_config,
                  mapping,
                  train_datasets,
                  validation_datasets,
                  pre_callbacks=[]):
    make_categorical = False
    swap_axes = True

    train_data = DataGenerator(train_datasets,
                               None,  # Autodetect
                               window_size,
                               stride_size,
                               mapping["inputs"],
                               mapping["target"],
                               make_categorical=make_categorical,
                               swap_axes=swap_axes,
                               loop=False,
                               postprocessing_callbacks=pre_callbacks)

    validation_data = DataGenerator(validation_datasets,
                                    None,  # Autodetect
                                    window_size,
                                    stride_size,
                                    mapping["inputs"],
                                    mapping["target"],
                                    make_categorical=make_categorical,
                                    swap_axes=swap_axes,
                                    loop=False)

    tiles = [tile for tile in train_data]
    validation_tiles = [tile for tile in validation_data]
    assert len(tiles) == 1
    assert len(validation_tiles) == 1
    train_tiles, ground_tiles = tiles[0]
    validation_tiles, validation_ground_tiles = validation_tiles[0]
    num_train_datasets = len(train_datasets)
    num_validation_datasets = len(validation_datasets)
    train_tiles = train_tiles.reshape((num_train_datasets * window_size[0] * window_size[1], train_tiles.shape[-1]))
    validation_tiles = validation_tiles.reshape(
        (num_validation_datasets * window_size[0] * window_size[1], validation_tiles.shape[-1]))

    ground_tiles = ground_tiles.flatten()
    ground_tiles = ground_tiles > 0
    validation_ground_tiles = validation_ground_tiles.flatten()
    validation_ground_tiles = validation_ground_tiles > 0

    fit_options = model_config["fit"]

    classifier = model_config["implementation"]
    classifier.fit(train_tiles, ground_tiles,
                   eval_set=((validation_tiles, validation_ground_tiles),),
                   **fit_options)
    log.info("Starting training")
    path = model_config.get("path")
    path = path.format(model_name=model_name,
                       time=str(time.time()),
                       hostname=socket.gethostname(),
                       user=getpass.getuser())
    log.info("Model will be saved to: %s", path)

    log.info("Saving model")
    with open(path, "wb") as f:
        pickle.dump(classifier, f)
    log.info("Done saving")
    log.info("Training completed")


def train_keras(model_name,
                window_size,
                stride_size,
                model_config,
                mapping,
                train_datasets,
                validation_datasets,
                pre_callbacks=(),
                enable_multi_gpu=False,
                gpus=None,
                cpu_merge=True,
                cpu_relocation=False,
                batch_size=None,
                random_seed=None,
                ):
    log.info("Starting keras training")

    import tensorflow as tf

    # Seed initialization should happed as early as possible
    if random_seed is not None:
        log.info("Setting Tensorflow random seed to: %d", random_seed)
        try:
            tf.set_random_seed(random_seed)
        except AttributeError: # Tf 2.0 fix
            tf.random.set_seed(random_seed)

    from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
    from ..tools.callbacks import ModelCheckpoint, CSVLogger
    from keras.optimizers import Adam
    from ..tools.utils import import_model_builder
    from keras.models import load_model
    from keras.utils import multi_gpu_model

    if batch_size is None:
        batch_size = model_config.get("batch_size", None)
    coregistration = model_config.get("coregistration", True)
    model_path = model_config["model_path"]
    model_loss = model_config.get("loss", "categorical_crossentropy")
    log.info("Using loss: %s", model_loss)
    model_metrics = model_config.get("metrics", "accuracy")
    # Make code compatible with previous version
    format_converter = model_config.get("format_converter", CategoricalConverter(2))
    swap_axes = model_config["swap_axes"]
    train_epochs = model_config["train_epochs"]
    prefetch_queue_size = model_config.get("prefetch_queue_size", 10)
    input_channels = len(mapping["inputs"])
    train_data = DataGenerator(train_datasets,
                               batch_size,
                               mapping["inputs"],
                               mapping["target"],
                               format_converter=format_converter,
                               swap_axes=swap_axes,
                               postprocessing_callbacks=pre_callbacks,
                               default_window_size=window_size,
                               default_stride_size=stride_size,
                               coregistration=coregistration)

    #train_data = ThreadedDataGenerator(train_data, queue_size=prefetch_queue_size)



    model_builder, model_builder_custom_options = import_model_builder(model_config["model_builder"])
    model_builder_option = model_config.get("options", {})

    steps_per_epoch = getattr(model_config, "steps_per_epoch", len(train_data) // batch_size)

    log.info("Traing data has %d tiles", len(train_data))
    log.info("steps_per_epoch: %d", steps_per_epoch)

    if len(validation_datasets) > 0:
        validation_data = DataGenerator(validation_datasets,
                                        batch_size,
                                        mapping["inputs"],
                                        mapping["target"],
                                        format_converter=format_converter,
                                        swap_axes=swap_axes,
                                        default_window_size=window_size,
                                        default_stride_size=stride_size)

        validation_data = ThreadedDataGenerator(validation_data, queue_size=prefetch_queue_size)

        validation_steps_per_epoch = getattr(model_config, "validation_steps_per_epoch", len(validation_data) // batch_size)

        log.info("Validation data has %d tiles", len(validation_data))
        log.info("validation_steps_per_epoch: %d", validation_steps_per_epoch)


    load_only_weights = model_config.get("load_only_weights", False)
    checkpoint = model_config.get("checkpoint", None)
    callbacks = []
    early_stopping = model_config.get("early_stopping", None)
    adaptive_lr = model_config.get("adaptive_lr", None)
    tensor_board = model_config.get("tensor_board", False)
    tb_log_dir = model_config.get("tb_log_dir", os.path.join("/tmp/", model_name))  # TensorBoard log directory
    tb_log_dir = tb_log_dir.format(model_name=model_name,
                                   time=str(time.time()),
                                   hostname=socket.gethostname(),
                                   user=getpass.getuser())
    keras_logging = model_config.get("log", None)
    if not keras_logging:
        log.info("Keras logging is disabled")
    else:
        csv_log_file = keras_logging.format(model_name=model_name,
                                            time=str(time.time()),
                                            hostname=socket.gethostname(),
                                            user=getpass.getuser())
        dir_head, dir_tail = os.path.split(csv_log_file)
        if dir_tail and not IOUtils.file_exists(dir_head):
            log.info("Creating directory: %s", dir_head)
            IOUtils.recursive_create_dir(dir_head)
        log.info("Logging training data to csv file: %s", csv_log_file)
        csv_logger = CSVLogger(csv_log_file, separator=',', append=False)
        callbacks.append(csv_logger)

    if tensor_board:
        log.info("Registering TensorBoard callback")
        log.info("Event log dir set to: {}".format(tb_log_dir))
        tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)
        callbacks.append(tb_callback)
        log.info("To access TensorBoard run: tensorboard --logdir {} --port <port_number> --host <host_ip> ".format(
            tb_log_dir))

    if checkpoint:
        checkpoint_file = checkpoint["path"]
        log.info("Registering checkpoint callback")
        destination_file = checkpoint_file % {
            'model_name': model_name,
            'time': str(time.time()),
            'hostname': socket.gethostname(),
            'user': getpass.getuser()}
        dir_head, dir_tail = os.path.split(destination_file)
        if dir_tail and not IOUtils.file_exists(dir_head):
            log.info("Creating directory: %s", dir_head)
            IOUtils.recursive_create_dir(dir_head)
        log.info("Checkpoint data directed to: %s", destination_file)
        checkpoint_options = checkpoint.get("options", {})
        checkpoint_callback = ModelCheckpoint(destination_file, **checkpoint_options)
        callbacks.append(checkpoint_callback)

    log.info("Starting training")

    options = {
        'epochs': train_epochs,
        'callbacks': callbacks
    }

    if len(validation_datasets) and len(validation_data) > 0 and validation_steps_per_epoch:
        log.info("We have validation data")
        options['validation_data'] = validation_data
        options["validation_steps"] = validation_steps_per_epoch
        if early_stopping:
            log.info("Enabling early stopping %s", str(early_stopping))
            callback_early_stopping = EarlyStopping(**early_stopping)
            options["callbacks"].append(callback_early_stopping)
        if adaptive_lr:
            log.info("Enabling reduce lr on plateu: %s", str(adaptive_lr))
            callback_lr_loss = ReduceLROnPlateau(**adaptive_lr)
            options["callbacks"].append(callback_lr_loss)
    else:
        log.warn("No validation data available. Ignoring")

    final_model_location = model_path.format(model_name=model_name,
                                             time=str(time.time()),
                                             hostname=socket.gethostname(),
                                             user=getpass.getuser())
    log.info("Model path is %s", final_model_location)

    existing_model_location = None
    if IOUtils.file_exists(final_model_location):
        existing_model_location = final_model_location

    if existing_model_location is not None and not load_only_weights:
        log.info("Loading existing model from: %s", existing_model_location)
        custom_objects = {}
        if model_builder_custom_options is not None:
            custom_objects.update(model_builder_custom_options)
        if enable_multi_gpu:
            with tf.device('/cpu:0'):
                model = load_model(existing_model_location, custom_objects=custom_objects)
        else:
            model = load_model(existing_model_location, custom_objects=custom_objects)
        log.info("Model loaded!")
    else:
        log.info("Building model")
        model_options = model_builder_option
        model_options['n_channels'] = input_channels
        input_height, input_width = window_size
        model_options['input_width'] = model_builder_option.get('input_width', input_width)
        model_options['input_height'] = model_builder_option.get('input_height', input_height)
        activation = model_config.get('activation', None)
        if activation:
            model_options["activation"] = activation
        if enable_multi_gpu:
            with tf.device('/cpu:0'):
                model = model_builder(**model_options)
        else:
            model = model_builder(**model_options)
        log.info("Model built")
        if load_only_weights and existing_model_location is not None:
            log.info("Loading weights from %s", existing_model_location)
            model.load_weights(existing_model_location)
            log.info("Finished loading weights")
    optimiser = model_config.get("optimiser", None)
    if optimiser is None:
        log.info("No optimiser specified. Using default Adam")
        optimiser = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    if enable_multi_gpu:
        log.info("Using Keras Multi-GPU Training")
        fit_model = multi_gpu_model(model, gpus=gpus, cpu_merge=cpu_merge, cpu_relocation=cpu_relocation)
    else:
        log.info("Using Keras default GPU Training")
        fit_model = model

    log.info("Compiling model")
    fit_model.compile(loss=model_loss, optimizer=optimiser, metrics=model_metrics)
    log.info("Model compiled")
    model.summary()

    fit_model.fit_generator(train_data, steps_per_epoch, **options)

    log.info("Saving model to %s", os.path.abspath(final_model_location))
    dir_head, dir_tail = os.path.split(final_model_location)
    if dir_tail and not IOUtils.file_exists(dir_head):
        log.info("Creating directory: %s", dir_head)
        IOUtils.recursive_create_dir(dir_head)

    model.save(final_model_location)

    log.info("Done saving")
    log.info("Training completed")


def train_handler(config, args):
    if args.switch_to_prefix:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        current_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
        log.info("Switching to %s", current_dir)
        os.chdir(current_dir)
        log.info("Current dir: %s", os.path.abspath(os.getcwd()))
    with IOUtils.open_file(args.config, "r") as cfg_file:
        config = yaml.load(cfg_file, Loader=Loader)

    model_name = config["model_name"]
    model_type = config["model_type"]
    random_seed = config.get("random_seed", None)
    model_config = config["model"]
    tilling_config = config.get("tilling", {})
    if 'window_size' in tilling_config:
        window_size = tilling_config["window_size"]
    else:
        log.warning("Using deprectated `window_size` location")
        window_size = config["window_size"]
    if 'stride_size' in tilling_config:
        stride_size = tilling_config["stride_size"]
    else:
        log.warning("Using deprectated `stride_size` location")
        stride_size = config["stride_size"]

    if random_seed is not None:
        log.info("Setting Python and NumPy seed to: %d", random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        log.warning("No random seed specified!")

    limit_validation_datasets = config.get("limit_validation_datasets", None)
    limit_train_datasets = config.get("limit_train_datasets", None)

    data_source = config.get("data_source")
    mapping = config["mapping"]
    augment = config.get("augment", False)
    input_channels = len(mapping["inputs"])
    log.info("Input has %d channels", input_channels)
    log.info("Model type is: %s", model_type)

    if args.split is None:
        dataset_cache = config.get("dataset_cache", None)
        if dataset_cache is not None:
            log.debug("dataset_cache is set from config to %s", dataset_cache)
            dataset_cache = dataset_cache.format(model_name=model_name,
                                                 time=str(time.time()),
                                                 hostname=socket.gethostname(),
                                                 user=getpass.getuser())
        else:
            dataset_cache = tempfile.NamedTemporaryFile().name
    else:
        if not IOUtils.file_exists(args.split):
            raise FileNotFoundError("Invalid split file")
        dataset_cache = args.split

    log.info("dataset_cache will be directed to: %s", dataset_cache)

    if data_source.input_source is None:
        data_source.set_input_source(args.input)
    log.info("Using datasource: %s", data_source)

    if not IOUtils.file_exists(dataset_cache):
        log.info("Loading datasets")
        train_datasets, validation_datasets = data_source.get_dataset_loaders()
        dump = (train_datasets._datasets, validation_datasets._datasets)

        log.info("Saving dataset cache to %s", dataset_cache)

        with IOUtils.open_file(dataset_cache, "w") as f:
            f.write(yaml.dump(dump, Dumper=Dumper))
    else:
        log.info("Loading training datasets from %s", dataset_cache)
        train_datasets, validation_datasets = yaml.load(IOUtils.open_file(dataset_cache), Loader=Loader)
        train_datasets, validation_datasets = data_source.build_dataset_loaders(train_datasets, validation_datasets)

    train_datasets.loop = True
    validation_datasets.loop = True

    if limit_validation_datasets:
        validation_datasets = validation_datasets[:limit_validation_datasets]

    if limit_train_datasets:
        train_datasets = train_datasets[:limit_train_datasets]

    pre_callbacks = []
    if augment:
        log.info("Enabling global level augmentation. Verify if this is desired!")

        def augment_callback(X, y):
            from ..preprocessing.augmentation import Augmentation
            aug = Augmentation(config)
            return aug.augment(X, y)

        pre_callbacks.append(augment_callback)

    log.info("Using %d training datasets", len(train_datasets))
    log.info("Using %d validation datasets", len(validation_datasets))

    if model_type == "keras":
        train_keras(model_name, window_size, stride_size, model_config, mapping, train_datasets, validation_datasets,
                    pre_callbacks=pre_callbacks,
                    enable_multi_gpu=args.keras_multi_gpu,
                    gpus=args.keras_gpus,
                    cpu_merge=args.keras_disable_cpu_merge,
                    cpu_relocation=args.keras_enable_cpu_relocation,
                    batch_size=args.keras_batch_size,
                    random_seed=random_seed,
                    )
        log.info("Keras Training completed")
    elif model_type == "sklearn":
        train_sklearn(model_name, window_size, stride_size, model_config, mapping, train_datasets, validation_datasets)
        log.info("Scikit Training completed")
    else:
        log.critical("Unknown model type: %s", model_type)
