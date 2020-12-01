import math

import os
from logging import getLogger

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence


from hugin.engine.core import RasterModel
from hugin.tools.callbacks import CSVLogger
from hugin.tools.utils import import_model_builder

log = getLogger(__name__)


class KerasModel(RasterModel):
    def __init__(self,
                 model_path,
                 model_builder,
                 *args,
                 optimizer=None,
                 destination=None,  # Hugin specific
                 checkpoint=None,  # Hugin specific
                 use_tpu=None,
                 enable_multi_gpu=False,  # Hugin specific
                 num_gpus=2,  # Number of GPU's to use
                 cpu_merge=True,  #
                 cpu_relocation=False,
                 loss=None,
                 loss_weights=None,
                 model_builder_options={},
                 epochs=1000,
                 verbose=1,
                 callbacks=None,
                 class_weight=None,
                 max_queue_size=10,
                 workers=0,
                 use_multiprocessing=False,
                 shuffle=True,
                 initial_epoch=0,
                 steps_per_epoch=None,
                 validation_steps_per_epoch=None,
                 load_only_weights=False,
                 metrics=None,
                 custom_objects={},
                 **kwargs):
        RasterModel.__init__(self, *args, **kwargs)
        self.custom_objects = custom_objects
        self.model_path = model_path
        self.destination = destination
        self.checkpoint = checkpoint
        self.use_tpu = use_tpu
        self.num_gpus = num_gpus
        if self.use_tpu is not None and self.enable_multi_gpu:
            raise ValueError("Can't use both multi_gpu and TPU's")
        self.cpu_merge = cpu_merge
        self.cpu_relocation = cpu_relocation
        self.load_only_weights = load_only_weights
        self.enable_multi_gpu = enable_multi_gpu
        self.model_builder_options = model_builder_options
        if 'input_shapes' not in self.model_builder_options:
            self.model_builder_options.update(input_shapes=self.input_shapes)
        if 'output_shapes' not in self.model_builder_options:
            self.model_builder_options.update(output_shapes=self.output_shapes)
        self.model_builder = model_builder
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = [] if not callbacks else callbacks
        self.class_weight = class_weight
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.shuffle = shuffle
        self.initial_epoch = initial_epoch
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.keras_metrics = metrics

        if model_builder:
            model_builder, model_builder_custom_options = import_model_builder(model_builder)
            self.model_builder = model_builder
            self.custom_objects.update(model_builder_custom_options)
            if 'name' not in self.model_builder_options:
                self.model_builder_options.update(name=self.name)
        self.model = None

    def predict(self, batch, batch_size=None):
        if self.model is None:
            self.__load_model()
        batch_size = batch_size if batch_size else self.batch_size
        prediction = self.model.predict(batch, batch_size=batch_size)
        return prediction

    def __load_model(self):
        import tensorflow as tf
        log.info("Loading keras model from %s", self.model_path)
        if not self.load_only_weights:
            if self.use_tpu:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                log.info("All TPU devices: %s", tf.config.list_logical_devices('TPU'))
                strategy = tf.distribute.TPUStrategy(resolver)
                with strategy.scope():
                    self.model = load_model(self.model_path, custom_objects=self.custom_objects)
            elif self.enable_multi_gpu:
                with tf.device('/cpu:0'):
                    self.model = load_model(self.model_path, custom_objects=self.custom_objects)
            else:
                self.model = load_model(self.model_path, custom_objects=self.custom_objects)

        else:
            if self.enable_multi_gpu:
                with tf.device('/cpu:0'):
                    self.model = self.__create_model()
            else:
                self.model = self.__create_model()
        log.info("Finished loading")
        return self.model

    def __create_model(self, train_data=None):
        import tensorflow as tf
        if self.enable_multi_gpu:
            with tf.device('/cpu:0'):
                return self.__create_model_impl(train_data)
        else:
            return self.__create_model_impl(train_data)

    def __extract_shapes(self, spec):
        pass

    def __create_model_impl(self, train_data):
        log.info("Building model")
        model_builder_options = self.model_builder_options
        if model_builder_options.get('input_shapes') is None:
            if self.input_shapes is None:
                shapes = train_data.get_input_shapes()
                model_builder_options["input_shapes"] = shapes
            else:
                model_builder_options['input_shapes'] = self.input_shapes
        if model_builder_options.get('output_shapes') is None:
            model_builder_options['output_shapes'] = self.output_shapes
        model = self.model_builder(**model_builder_options)
        log.info("Model built")
        self.model = model
        return model

    def fit_generator(self, train_data, validation_data=None):
        log.info("Training from generators")
        fit_options = {}
        if self.steps_per_epoch is None:
            if isinstance(train_data, Sequence):
                pass
            else:
                steps_per_epoch = math.ceil(len(train_data) / self.batch_size)
                fit_options.update(steps_per_epoch=steps_per_epoch)
        else:
            fit_options.update(steps_per_epoch=self.steps_per_epoch)

        if self.validation_steps_per_epoch is None:
            if validation_data is not None:
                if isinstance(validation_data, Sequence):
                    pass
                else:
                    validation_steps_per_epoch = math.ceil(len(validation_data) / self.batch_size)
                    fit_options.update(steps_per_epoch=validation_steps_per_epoch)
            else:
                validation_steps_per_epoch = None
        else:
            fit_options.update(steps_per_epoch=validation_steps_per_epoch)

        if os.path.exists(self.model_path):
            log.info("Loading existing model")
            model = self.__load_model()
        else:
            model = self.__create_model(train_data)
            if self.enable_multi_gpu:
                log.info("Using Keras Multi-GPU Training")
                gpus = self.num_gpus
                if gpus is None:
                    gpus = os.environ['HUGIN_KERAS_GPUS']
                cpu_merge = self.cpu_merge
                cpu_relocation = self.cpu_relocation
                model = multi_gpu_model(model,
                                        gpus=gpus,
                                        cpu_merge=cpu_merge,
                                        cpu_relocation=cpu_relocation)

            model.compile(self.optimizer,
                          loss=self.loss,
                          loss_weights=self.loss_weights,
                          metrics=self.keras_metrics
                          )
            print(model.summary())

        callbacks = []

        for callback in self.callbacks:
            callbacks.append(callback)
        log.info(f"Registered callbacks: {callbacks}")
        log.debug(f"Checkpoint: {self.checkpoint}")
        log.debug(f"Validation data: {validation_data}")
        if self.checkpoint and validation_data is not None:
            if not self.destination is not None:
                log.warning("Destination not specified. Checkpoints will not be saved")
            else:
                log.info(f"Checkpoints will be saved in {self.destination}")
                monitor = self.checkpoint.get('monitor', 'val_loss')
                verbose = self.checkpoint.get('verbose', 1)
                save_best_only = self.checkpoint.get('save_best_only', False)
                save_weights_only = self.checkpoint.get('save_weights_only', False)
                mode = self.checkpoint.get('mode', 'auto')
                period = self.checkpoint.get('period', 1)
                filename = self.checkpoint.get('filename', "checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5")
                checkpoint_destination = os.path.join(self.destination, "checkpoints")
                if not os.path.exists(checkpoint_destination):
                    os.makedirs(checkpoint_destination)
                filepath = os.path.join(checkpoint_destination, filename)
                log.info("Registering model checkpoing")
                callbacks.append(ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=verbose,
                                                 save_best_only=save_best_only, save_weights_only=save_weights_only,
                                                 mode=mode, period=period))
        if self.destination is not None and validation_data is not None:
            log_destination = os.path.join(self.destination, "logs.txt")
            callbacks.append(CSVLogger(log_destination))

        fit_options.update(epochs=self.epochs,
                           verbose=self.verbose,
                           callbacks=callbacks,
                           validation_data=validation_data,
                           class_weight=self.class_weight,
                           max_queue_size=self.max_queue_size,
                           workers=self.workers,
                           use_multiprocessing=self.use_multiprocessing,
                           shuffle=self.shuffle,
                           initial_epoch=self.initial_epoch)

        print (fit_options)

        model.fit_generator(train_data, **fit_options)

    def save(self, destination=None):
        log.info("Saving Keras model to %s", destination)
        if not os.path.exists(destination):
            os.makedirs(destination)
        destination = os.path.join(destination, "model.hdf5")
        self.model.save(destination)
