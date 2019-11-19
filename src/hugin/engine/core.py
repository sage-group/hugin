import os
import pickle
from logging import getLogger

import numpy as np
from sklearn.preprocessing import StandardScaler

log = getLogger(__name__)


def metric_processor(func):
    def __metric_handler(self, scene, *args, **kwargs):
        metrics = kwargs.pop('_metrics', {})
        original_result = func(self, scene, *args, _metrics=metrics, **kwargs)
        scene_id, scene_data = scene
        my_metrics = {}
        if self.name not in metrics:
            metrics[self.name] = my_metrics
        else:
            my_metrics = metrics[self.name]
        if self.metrics:
            if self.gti_component in scene_data:
                gti = scene_data[self.gti_component]
                for metric_name, metric_calculator in self.metrics.items():
                    my_metrics[metric_name] = metric_calculator(original_result, gti.read())
            else:
                log.warning("Missing GTI data for GTI component: %s", self.gti_component)
        return (original_result, metrics)

    return __metric_handler


def postprocessor(func):
    def __postprocessor_handler(self, *args, **kwargs):
        kwargs.pop("_metrics", None)
        result = func(self, *args, **kwargs)
        postprocessors = getattr(self, 'post_processors')
        if not postprocessors:
            log.debug("No post-processors found for: %s", self)

        if callable(postprocessors):
            return postprocessors(result)

        try:
            iter(postprocessors)
            isiter = True
        except TypeError:
            isiter = False

        if isiter:
            for processor in postprocessors:
                if callable(processor):
                    log.debug("Running processor %s", processor)
                    result = processor(result)
                else:
                    log.warning("Non-Callable processor %s", processor)
            return result
        else:
            log.debug("No valid post-processors found for: %s", self)
            return result

    return __postprocessor_handler


def identity_processor(arg):
    return arg


class CategoricalConverter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, probability_array):
        prediction = np.argmax(probability_array, -1).astype(np.uint8)
        return prediction.reshape(prediction.shape + (1,))


class RasterModel(object):
    def __init__(self,
                 name=None,
                 batch_size=1,
                 swap_axes=True,
                 input_shapes=None,
                 output_shapes=None,
                 # input_shape=None,
                 # output_shape=None
                 ):
        """Base model object handling prediction

        :param name: Name of the model (optional)
        :param batch_size: Batch size used by `predict_scene_proba` when sending data `predict`. Default: 1
        :param swap_axes: Swap input data axes
        :param input_shape: Window size to be used by `predict_scene_proba`
        """
        instance_path = ".".join([self.__module__, self.__class__.__name__])
        self.name = "%s[%s]" % (instance_path, name if name is not None else self.__hash__())
        self.model_name = name if name is not None else self.__hash__()

        self.batch_size = batch_size
        self.swap_axes = swap_axes
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        # self.input_shape = input_shape
        # self.output_shape = output_shape

    def predict(self, batch):
        """Runs the predictor on the input tile batch

        :param batch: The input batch
        :return: returns a prediction according to model configuration
        """
        raise NotImplementedError()

    def save(self, destination=None):
        raise NotImplementedError()

    def fit_generator(self, train_data, validation_data=None):
        raise NotImplementedError()


class PredictionMerger(object):
    def __init__(self, height, width, depth, dtype):
        self.height = height
        self.width = width
        self.depth = depth
        self.dtype = dtype

    def update(self, xstart, xend, ystart, yend, prediction):
        raise NotImplementedError()

    def get_prediction(self):
        raise NotImplementedError()


class NullMerger(PredictionMerger):
    def __init__(self, *args, **kwargs):
        super(NullMerger, self).__init__(*args, **kwargs)
        self.tile_array = np.zeros((self.height, self.width, self.depth), dtype=self.dtype)

    def update(self, xstart, xend, ystart, yend, prediction):
        self.tile_array[xstart: xend, ystart: yend] = prediction

    def get_prediction(self):
        return self.tile_array



class AverageMerger(PredictionMerger):
    def __init__(self, *args, **kwargs):
        super(AverageMerger, self).__init__(*args, **kwargs)
        self.tile_array = np.zeros((self.height, self.width, self.depth), dtype=self.dtype)
        self.tile_freq_array = np.zeros((self.height, self.width), dtype=np.uint8)

    def update(self, xstart, xend, ystart, yend, prediction):
        self.tile_array[xstart: xend, ystart: yend] += prediction
        self.tile_freq_array[xstart: xend, ystart: yend] += 1

    def get_prediction(self):
        tile_array = self.tile_array.copy()
        channels = tile_array.shape[-1]
        unique, counts = np.unique(self.tile_freq_array, return_counts=True)
        # print(dict(zip(unique, counts)))
        for i in range(0, channels):
            tile_array[:, :, i] = tile_array[:, :, i] / self.tile_freq_array
        return tile_array


class SkLearnStandardizer(RasterModel):
    def __init__(self, model_path, with_gti = True, *args, copy=True, with_mean=True, with_std=True, **kw):
        RasterModel.__init__(self, *args, **kw)
        self.destination = model_path
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._with_gti = with_gti
        self._single_input = True
        self._scalers = []
        self._multiple_input_scalers = {}

    def fit_generator(self, train_data, validation_data=None):
        from tqdm import tqdm
        count = len(train_data)
        train_data.loop = False
        pbar = tqdm(total=count)
        with pbar:
            for i in range(0, count):
                pbar.update(1)
                tile = next(train_data)

                try:
                    for input_name, data in tile[0].items():
                        if input_name not in self._multiple_input_scalers:
                            self._multiple_input_scalers[input_name] =\
                                (StandardScaler(copy=self.copy, with_std=self.with_std,
                                                with_mean=self.with_mean),
                                 StandardScaler(copy=self.copy, with_std=self.with_std,
                                                with_mean=self.with_mean))

                            in_scaler = self._multiple_input_scalers[input_name][0]
                            out_scaler = self._multiple_input_scalers[input_name][0]

                            in_data = tile[0][input_name].ravel()
                            out_data = tile[1].ravel()

                            inp = in_data.reshape((in_data.shape + (1,)))
                            outp = out_data.reshape((out_data.shape + (1,)))
                            in_scaler.partial_fit(inp)
                            out_scaler.partial_fit(inp)
                    self._single_input = False

                # Then we have single input
                except AttributeError:
                    self._scalers.append((StandardScaler(copy=self.copy, with_std=self.with_std,
                                                         with_mean=self.with_mean),
                                          StandardScaler(copy=self.copy, with_std=self.with_std,
                                                        with_mean=self.with_mean)))

                    in_scaler = self._scalers[i][0]
                    out_scaler = self._scalers[i][1]

                    in_data = tile[0].ravel()
                    out_data = tile[1].ravel()

                    inp = in_data.reshape((in_data.shape + (1,)))
                    outp = out_data.reshape((out_data.shape + (1,)))
                    in_scaler.partial_fit(inp)
                    out_scaler.partial_fit(inp)

    def save(self, destination):
        if not os.path.exists(destination):
            os.makedirs(destination)

        if self._single_input:
            for id, scaler in enumerate(self._scalers):
                for i, mode in enumerate(['input', 'gti_']):
                    scaler_destination = os.path.join(destination, mode + str(id) + '.pkl')
                    outs = pickle.dumps(scaler[i])
                    with open(scaler_destination, "wb") as f:
                        f.write(outs)

        else:
            for input_name, scaler in self._multiple_input_scalers.items():
                for i, mode in enumerate(['', '_gti']):
                    scaler_destination = os.path.join(destination, input_name + mode + '.pkl')
                    outs = pickle.dumps(scaler[i])
                    with open(scaler_destination, "wb") as f:
                        f.write(outs)


class IdentityModel(RasterModel):
    def __init__(self, *args, num_loops=1, **kwargs):
        RasterModel.__init__(self, *args, **kwargs)
        self.destination = None
        self.num_loops = num_loops

    def fit_generator(self, train_data, validation_data=None):
        for i in range(self.num_loops):
            tdata = len(train_data)
            vdata = len(validation_data) if validation_data is not None else 0
            for j in range(0, tdata):
                data = next(train_data)
            for j in range(0, vdata):
                data = next(validation_data)

    def predict(self, batch, batch_size=None):
        return batch


    def save(self, destination=None):
        # raise NotImplementedError()
        pass


def identity_metric(prediction, gti):
    return 1

class RasterGenerator(object):
    """
    Base class used by handlers generating new data components
    """
    def __call__(self, scene_components):
        """
        Generate a new scene component
        Args:
            scene_components: existing scene components

        Returns: Should return an rasterio dataset

        """
        raise NotImplementedError()


class CloneComponentGenerator(RasterGenerator):
    """
    Generator generating an clone of an existing component
    """
    def __init__(self, base_component):
        """

        Args:
            base_component: the component we should use for detecting the size, etc
        """
        self._base_component = base_component

    def __call__(self, scene_components):
        from rasterio.io import MemoryFile
        base_component = scene_components[self._base_component]
        return MemoryFile(base_component)
