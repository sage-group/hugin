import re
import os
import math

import h5py
import numpy as np
import rasterio

from logging import getLogger
from tempfile import TemporaryFile, NamedTemporaryFile

from hugin.engine.core import metric_processor
from hugin.io.loader import CategoricalConverter as TrainingCategoricalConverter
from .core import NullMerger, postprocessor
from ..io import DataGenerator, DatasetGenerator
from ..io.loader import adapt_shape_and_stride


log = getLogger(__name__)


class MultipleSceneModel:
    """
    This class is intended to be inherited by classes aimed to predict on multiple scenes
    """

    def __init__(self,
                 scene_id_filter=None,
                 randomize_training=True):
        """

        :param scene_id_filter: Regex for filtering scenes according to their id (optional)
        :param randomize_training: Randomize list of traning data between epochs
        """

        self.scene_id_filter = None if not scene_id_filter else re.compile(scene_id_filter)
        self.randomize_training = randomize_training

    def predict_scenes_proba(self, scenes, predictor=None):
        """Run the predictor on all input scenes

        :param scenes: An iterable object yielding tuples like (scene_id, type_mapping)
        :param predictor: The predictor to use for predicting scenes (defaults to self)
        :return: a list of predictions according to model configuration
        """

        predictor = self if predictor is None else predictor
        for scene_id, scene_data in scenes:
            if self.scene_id_filter and self.scene_id_filter.match(scene_id) is None:
                continue
            log.info("Classifying %s", scene_id)
            output = (scene_id, scene_data, predictor.predict_scene_proba((scene_id, scene_data)))
            yield output

    def train_scenes(self, scenes, validation_scenes=None, trainer = None):
        trainer = self if trainer is None else trainer

        inputs = self.mapping["inputs"]
        target = self.mapping["target"]
        preprocessors = self.pre_processors if self.pre_processors else []
        validation_scenes = [] if validation_scenes is None else validation_scenes
        scenes.randomise_on_loop = self.randomize_training
        train_data = DataGenerator(scenes,
                                   self.predictor.batch_size,
                                   inputs,
                                   target,
                                   format_converter=self.format_converter,
                                   swap_axes=self.predictor.swap_axes,
                                   postprocessing_callbacks=preprocessors,
                                   default_window_size=self.window_size,
                                   default_stride_size=self.stride_size
                                   )

        input_shapes, output_shapes = train_data.mapping_sizes

        if self.predictor.input_shapes is None:
            self.predictor.input_shapes = input_shapes
        if self.predictor.output_shapes is None:
            self.predictor.output_shapes = output_shapes


        log.info("Training data has %d tiles", len(train_data))
        validation_data = None
        if validation_scenes:
            validation_data = DataGenerator(validation_scenes,
                                            self.predictor.batch_size,
                                            inputs,
                                            target,
                                            format_converter=self.format_converter,
                                            swap_axes=self.predictor.swap_axes,
                                            postprocessing_callbacks=preprocessors,
                                            default_window_size=self.window_size,
                                            default_stride_size=self.stride_size)
            log.info("Validation data has %d tiles", len(validation_data))
        else:
            log.info("No validation data")

        options = {}
        if self.predictor.destination is None:
            self.predictor.destination = self.destination

        log.info("Running training from %s", trainer.predictor)
        self.predictor.fit_generator(train_data, validation_data, **options)
        log.info("Training completed")

class BaseSceneModel:
    def __init__(self, post_processors=None, pre_processors=None, metrics=None, gti_component=None):
        self.post_processors = post_processors
        self.pre_processors = pre_processors
        self.metrics = metrics
        self.gti_component=gti_component

    @postprocessor
    def predict_scene_proba(self, *args, **kwargs):
        raise NotImplementedError()

class CoreScenePredictor(BaseSceneModel):
    def __init__(self, predictor,
                 name=None,
                 mapping=None,
                 stride_size=None,
                 window_size=None,
                 output_shape=None,
                 prediction_merger=NullMerger,
                 post_processors=None, # Run after we get the data form predictors
                 pre_processors=None, # Run before sending the data to predictors
                 format_converter=TrainingCategoricalConverter(2),
                 metrics=None):
        """

        :param predictor: Predictor to be used for predicting
        :param name: Name of the model (optional)
        :param mapping: Mapping of input files to input data
        :param stride_size: Stride size to be used by `predict_scene_proba`
        :param output_shape: Output shape of the prediction (Optional). Inferred from input image size
        """
        BaseSceneModel.__init__(self, post_processors=post_processors, pre_processors=pre_processors, metrics=metrics)
        self.predictor = predictor

        instance_path = ".".join([self.__module__, self.__class__.__name__])
        self.name = "%s[%s]:%s" % (instance_path, name if name is not None else self.__hash__(), predictor.name)
        self.model_name = name if name is not None else self.__hash__()
        self.stride_size = stride_size if stride_size is not None else self.predictor.input_shape[0]
        self.window_size = window_size
        if not mapping: raise TypeError("Missing `mapping` specification in %s" % self.name)
        self.mapping = mapping
        self.input_mapping = mapping.get('inputs')
        self.output_mapping = mapping.get('output', None)
        self.output_shape = output_shape
        self.prediction_merger_class = prediction_merger
        self.format_converter = format_converter

    @metric_processor
    @postprocessor
    def predict_scene_proba(self, scene, dataset_loader=None):
        """Runs the predictor on the input scene
        This method might call `predict` for image patches

        :param scene: An input scene
        :return: a prediction according to model configuration
        """
        log.info("Generating prediction representation")
        scene_id, scene_data = scene
        output_mapping = self.mapping.get("target", {})

        tile_loader = DatasetGenerator((scene,)) if dataset_loader is None else dataset_loader.get_dataset_loader(scene)
        tile_loader.reset()
        data_generator = DataGenerator(tile_loader,
                                       batch_size=self.predictor.batch_size,
                                       input_mapping=self.input_mapping,
                                       output_mapping=None,
                                       swap_axes=self.predictor.swap_axes,
                                       loop=False,
                                       default_window_size=self.window_size,
                                       default_stride_size=self.stride_size)

        if len(output_mapping) == 1: # Output is defined by GTI
            rio_raster = scene_data[output_mapping[0][0]]
            output_window_shape, output_stride_size = adapt_shape_and_stride(rio_raster,
                                                                             data_generator.primary_scene,
                                                                             self.predictor.input_shape,
                                                                             self.stride_size)
        else:
            output_window_shape = self.window_size
            output_stride_size = self.stride_size

        output_shape = self.output_shape
        if output_shape:
            image_height, image_width = output_shape
        else:
            image_height, image_width = data_generator.primary_scene.shape

        tile_height, tile_width = output_window_shape
        window_width, window_height = tile_width, tile_height # ???

        if tile_width != self.stride_size:
            xtiles = math.ceil((image_width - tile_width) / float(self.stride_size) + 2)
        else:
            xtiles = math.ceil((image_width - tile_width) / float(self.stride_size) + 1)

        if tile_height != self.stride_size:
            ytiles = math.ceil((image_height - tile_height) / float(self.stride_size) + 2)
        else:
            ytiles = math.ceil((image_height - tile_height) / float(self.stride_size) + 1)


        merger = None
        stride = self.stride_size
        ytile = 0

        def prediction_generator(dgen):
            for data in dgen:
                in_arrays, out_arrays = data
                prediction = self.predictor.predict(in_arrays, batch_size=self.predictor.batch_size)
                for i in range(0, prediction.shape[0]):
                    yield prediction[i]

        predictions = prediction_generator(data_generator)
        while ytile < ytiles:
            y_start = ytile * stride
            y_end = y_start + window_height
            if y_end > image_height:
                y_start = image_height - window_height
                y_end = y_start + window_height
            ytile += 1
            xtile = 0
            while xtile < xtiles:
                x_start = xtile * stride
                x_end = x_start + window_width
                if x_end > image_width:
                    x_start = image_width - window_width
                    x_end = x_start + window_width
                xtile += 1

                # Now we know the position
                prediction = next(predictions)
                tile_prediction = prediction.reshape(tile_height,
                                                     tile_width,
                                                     prediction.shape[-1])
                if merger is None:
                    merger = self.prediction_merger_class(image_height, image_width, prediction.shape[-1],
                                                          prediction.dtype)

                merger.update(y_start, y_end, x_start, x_end, tile_prediction)

        if merger is None:
            log.warning("No data merged in prediction")
            return None
        prediction = merger.get_prediction()
        return prediction


class RasterScenePredictor(CoreScenePredictor, MultipleSceneModel):
    def __init__(self, model, *args, scene_id_filter=None, **kwargs):
        CoreScenePredictor.__init__(self, model, *args, **kwargs)
        MultipleSceneModel.__init__(self, scene_id_filter=scene_id_filter)

class RasterSceneTrainer(CoreScenePredictor, MultipleSceneModel):
    def __init__(self, model, *args, destination=None, scene_id_filter=None, **kwargs):
        CoreScenePredictor.__init__(self, model, *args, **kwargs)
        MultipleSceneModel.__init__(self, scene_id_filter=scene_id_filter)
        self.destination = destination

    def predict_scene_proba(self, scene, dataset_loader=None):
        raise NotImplementedError()

    def save(self, destination=None):
        destination = destination if destination is not None else self.destination
        if not os.path.exists(destination):
            os.makedirs(destination)
        final_destination = os.path.join(destination, self.predictor.model_name)
        self.predictor.save(final_destination)


class BaseEnsembleScenePredictor(BaseSceneModel, MultipleSceneModel):
    def __init__(self, predictors,  *args, name=None, resume=False, cache_file=None, **kwargs):
        post_processors = kwargs.pop('post_processors', None)
        metrics = kwargs.pop('metrics', None)
        instance_path = ".".join([self.__module__, self.__class__.__name__])
        self.name = "%s[%s]" % (instance_path, name if name is not None else self.__hash__(), )
        gti_component = kwargs.pop('gti_component', None)
        BaseSceneModel.__init__(self, post_processors=post_processors, metrics=metrics, gti_component=gti_component)
        MultipleSceneModel.__init__(self, *args, **kwargs)
        self.predictors = predictors
        for predictor in predictors:
            predictor["predictor"].metrics = self.metrics
            if predictor["predictor"].gti_component is None:
                predictor["predictor"].gti_component = self.gti_component
        self.resume = resume
        cache_file = cache_file if cache_file is not None else NamedTemporaryFile("w+b").name
        self.cache = h5py.File(cache_file, 'a')
        self.metrics_store = {}
        log.info("Ensemble predictions stored in: %s", cache_file)

    def predict_scenes_proba(self, scenes):
        log.debug("Computing predictions for models")
        metrics_store = self.metrics_store
        for predictor_configuration in self.predictors:
            scenes.reset()
            predictor = predictor_configuration["predictor"]
            log.info("Predicting using predictor: %s", predictor)
            for scene_id, _, result in super(BaseEnsembleScenePredictor, self).predict_scenes_proba(scenes, predictor):
                prediction, metrics = result
                dataset_name = "%s/%s" % (predictor.name, scene_id)
                if self.resume:
                    if dataset_name in self.cache.keys():
                        log.info("Scene %s already predicted using %s. Skipping!", dataset_name, predictor.name)
                        continue
                else:
                    if dataset_name in self.cache.keys():
                        del self.cache[dataset_name]
                log.debug("Storing prediction for %s under %s", scene_id, dataset_name)
                self.cache[dataset_name] = prediction

                metrics_store[scene_id] = metrics
                #print("Metrics from predict: %s", metrics)
        scenes.reset()
        for scene in scenes:
            scene_id, scene_data = scene
            log.debug("Ensembling prediction for %s", scene_id)
            result = self.predict_scene_proba(scene)
            prediction, metrics = result
            #print ("predict_scenes_proba => ", metrics)
            #print ("predict_scenes_proba => ", self.metrics_store[scene_id])
            scene_metrics = {}
            scene_metrics.update(metrics)
            scene_metrics.update(self.metrics_store[scene_id])
            result = (prediction, scene_metrics)
            log.debug("Done ensembling")
            yield (scene_id, scene_data, result)


class AvgEnsembleScenePredictor(BaseEnsembleScenePredictor):
    @metric_processor
    @postprocessor
    def predict_scene_proba(self, scene):
        scene_id, scene_data = scene
        total_weight = 0
        sum_array = None
        for predictor_configuration in self.predictors:
            predictor = predictor_configuration["predictor"]
            predictor_weight = predictor_configuration.get("weight", 1)
            total_weight += predictor_weight
            dataset_name = "%s/%s" % (predictor.name, scene_id)
            log.debug("Using prediction from h5 dataset: %s", dataset_name)

            prediction = self.cache[dataset_name][()]
            if sum_array is None:
                sum_array = np.zeros(prediction.shape, dtype=prediction.dtype)
            sum_array += predictor_weight * prediction
        result = sum_array / total_weight

        return result



class SceneExporter(object):
    def __init__(self, destination=None, metric_destination=None):
        self.destination = destination
        self.metric_destination = metric_destination
    @property
    def destination(self):
        return self._destination

    @destination.setter
    def destination(self, destination):
        self._destination = destination

    def save_scene(self, scene_id, scene_data, prediction, destination=None):
        raise NotImplementedError()

    def flow_prediction_from_source(self, loader, predictor):
        predictions = predictor.predict_scenes_proba(loader)
        overall_metrics = {}
        for scene_id, scene_data, result in predictions:
            prediction, metrics = result
            self.save_scene(scene_id, scene_data, prediction)
            overall_metrics[scene_id] = metrics
        import json
        #print (json.dumps(overall_metrics, indent=4))
            #print (metrics)


class MultipleFormatExporter(SceneExporter):
    def __init__(self, *args, exporters=[], **kwargs):
        SceneExporter.__init__(self, *args, **kwargs)
        self.exporters = exporters

    def save_scene(self, *args, destination=None, **kwargs):
        destination = self.destination if destination is None else destination
        for exporter in self.exporters:
            exporter.save_scene(*args, destination=destination, **kwargs)

class RasterIOSceneExporter(SceneExporter):
    def __init__(self, *args,
                 srs_source_component=None,
                 rasterio_options={},
                 rasterio_creation_options={},
                 filename_pattern="{scene_id}.tif", **kwargs):
        SceneExporter.__init__(self, *args, **kwargs)
        self.srs_source_component = srs_source_component
        self.rasterio_options = rasterio_options
        self.rasterio_creation_options = rasterio_creation_options
        self.filename_pattern = filename_pattern

    def save_scene(self, scene_id, scene_data, prediction, destination=None, destination_file=None):
        if destination_file is None:
            destination_file=self.filename_pattern.format(scene_id=scene_id)
        destination = self.destination if destination is None else destination
        destination_file = os.path.join(destination, destination_file)
        log.info("Saving scene %s to %s", scene_id, destination_file)
        with rasterio.Env(**(self.rasterio_options)):
            profile = {}
            if self.srs_source_component is not None:
                src = scene_data[self.srs_source_component]
                profile.update (src.profile)
            num_out_channels = prediction.shape[-1]
            profile.update(self.rasterio_creation_options)
            profile.update(dtype=prediction.dtype, count=num_out_channels)
            if 'driver' not in profile:
                profile['driver'] = 'GTiff'
            if profile['driver'] == 'GTiff':
                if 'compress' not in profile:
                    profile['compress'] = 'lzw'
            prediction_height, prediction_width = prediction.shape[0:2]
            if 'height' not in profile:
                profile['height'] = prediction_height
            if 'width' not in profile:
                profile['width'] = prediction_width
            with rasterio.open(destination_file, "w", **profile) as dst:
                for idx in range(0, num_out_channels):
                    dst.write(prediction[:, :, idx], idx + 1)
