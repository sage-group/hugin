import math
import os
import sys
import random
import numpy as np
from logging import getLogger

import zarr

from .dataset_loaders import ArrayLoader
from dask.array import from_zarr, Array
from zarr.storage import LRUStoreCache
from tensorflow.keras.utils import Sequence
from urllib.parse import urlparse, parse_qs

log = getLogger(__name__)


class ArraySequence(Sequence):
    def __init__(self,
                 input_component_mapping: dict,
                 output_component_mapping: dict,
                 batch_size: int,
                 standardisers: dict = None,
                 selected_indices: Array = None,
                 randomise: bool = False,
                 maximum_samples: int = None,
                 sample_weights = None,
                 slice_timestamps: list = None,
                 flat: bool = True):

        self.input_component_mapping = input_component_mapping
        self.output_component_mapping = output_component_mapping
        self.randomise = randomise
        self.selected_indices = selected_indices
        self.batch_size = batch_size if batch_size is not None else 1
        self.maximum_samples = maximum_samples
        self.standardisers = standardisers
        self.sample_weights = sample_weights
        self.slice_timestamps = slice_timestamps
        self.flat = flat

        if self.slice_timestamps is not None:
            assert isinstance(self.slice_timestamps, (tuple, list)), "Timestamp slice should be list or tuple"
            assert len(self.slice_timestamps) == 3 or len(self.slice_timestamps) == 2
            if len(self.slice_timestamps) == 3:
                self.start_slice_idx, self.end_slice_idx, self.slice_step = self.slice_timestamps
            elif len(self.slice_timestamps) == 2:
                self.start_slice_idx, self.end_slice_idx = self.slice_timestamps
                self.slice_step = 1
            else:
                raise NotImplementedError()

    @property
    def selected_indices(self):
        if self.__selected_indices is None:
            one_array = self.input_component_mapping[list(self.input_component_mapping.keys())[0]]
            self.__selected_indices = np.arange(0, len(one_array))
            data = self.__selected_indices
            if self.maximum_samples is not None:
                data = data[:self.maximum_samples]
            return data
        else:
            data = self.__selected_indices
            if self.maximum_samples is not None:
                data = data[:self.maximum_samples]
            return data

    @selected_indices.setter
    def selected_indices(self, v: Array):
        data = np.array(
            v) if v is not None else v  # Convert to NumPy array to prevent issues, memory impact should be minimal as there should be a limited amount if indices
        self.__selected_indices = data
        self.__shuffle_indices()

    def __shuffle_indices(self):
        if self.randomise and self.__selected_indices is not None:
            np.random.shuffle(self.__selected_indices)

    def on_epoch_end(self):
        self.__shuffle_indices()

    def __len__(self):
        length = math.ceil(len(self.selected_indices) / self.batch_size)
        return int(length)

    def get_real_length(self) -> int:
        return len(self.selected_indices)

    def __iter__(self):
        def __iterator():
            for i in range(0, len(self)):
                yield self[i]

        return __iterator

    def _get_real_idx(self, idx, height, width):
        if self.flat:
            return idx
        row, col = idx // width, idx % width
        return (row, col)

    def __getitem__(self, requested_index):
        inputs = {}
        outputs = {}

        indices = self.selected_indices

        start_idx = requested_index * self.batch_size
        end_idx = (requested_index + 1) * self.batch_size

        for flat_idx in indices[start_idx:end_idx]:
            for key, value in self.input_component_mapping.items():
                if key not in inputs:
                    inputs[key] = []
                idx = self._get_real_idx(flat_idx, value.shape[0], value.shape[1])
                if len(idx) == 2:
                    dask_data = value[idx[0],idx[1], ...]
                else:
                    dask_data = value[idx]
                if self.slice_timestamps:
                    dask_data = dask_data[self.start_slice_idx:self.end_slice_idx:self.slice_step, ...]
                data = np.array(dask_data)
                standardiser = self.standardisers.get(key) if self.standardisers else None
                if standardiser is not None:
                    data = data.astype(np.float64)
                    for i in range(0, len(standardiser)):
                        channel_standardizer = standardiser[i]
                        old_shape = data[..., i].shape
                        data[..., i] = channel_standardizer.transform(data[..., i].reshape(-1, 1), copy=False).reshape(
                            old_shape)
                inputs[key].append(data)

            for key, value in self.output_component_mapping.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value[idx])

        source = {k: np.array(np.stack(v)) for k, v in inputs.items()}
        targets = {k: np.array(np.stack(v)) for k, v in outputs.items()}
        result = [source, targets]
        if self.sample_weights is not None:
            result.append(self.sample_weights)
        return tuple(result)

    def get_input_shapes(self):
        shapes = {}
        for key, value in self.input_component_mapping.items():
            #if self.slice_timestamps:
            #    print (value.shape, self.slice_timestamps)
            #    value = value[:, self.start_slice_idx:self.end_slice_idx:self.slice_step, ...]
            if self.flat:
                shp = value.shape[1:]
            else:
                shp = value.shape[2:]
            if self.slice_timestamps:
                shp = (self.end_slice_idx-self.start_slice_idx, ) + shp[1:]
            shapes[key] = shp
        return shapes


class ZarrArrayLoader(ArrayLoader):
    def __init__(self,
                 inputs: dict,
                 targets: dict,
                 source=None,
                 flat: bool = True,
                 split_test_index_array: Array = None,
                 split_train_index_array: Array = None,
                 randomise: bool = False,
                 random_seed: int = None,
                 maximum_training_samples: int = None,
                 maximum_validation_samples: int = None,
                 class_weights = None,
                 sample_weights = None,
                 slice_timestamps = None):
        super(ZarrArrayLoader, self).__init__()
        self.inputs = {}
        self.flat = flat
        self.input_standardizers = {}
        self.split_test_index_array_path = split_test_index_array
        self.split_train_index_array_path = split_train_index_array
        self.split_test_index_array = None
        self.split_train_index_array = None
        self.randomise = randomise
        self.random_seed = random_seed
        self.maximum_training_samples = maximum_training_samples
        self.maximum_validation_samples = maximum_validation_samples
        self.class_weights = class_weights
        self.sample_weights = sample_weights
        self.slice_timestamps = slice_timestamps
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        if source is None:
            if 'DATASOURCE_URL' not in os.environ:
                raise TypeError(
                    "Missing source specification. Should be specified directly or as the `DATASOURCE_URL` environment variable")
            else:
                source = os.environ['DATASOURCE_URL']
                log.info(f"Using storage configuration from environment file `DATASOURCE_URL`: {source}")
        up = urlparse(source)
        storage_options = {}
        if not up.scheme:
            self.source = source
            self.source = zarr.open_group(source, mode="r", storage_options=storage_options)
        else:
            source = f"{up.scheme}://{up.netloc}{up.path}"
            for k, v in parse_qs(up.query, keep_blank_values=True).items():
                value = v[0] if v[0] else None
                storage_options[k] = value
            if up.scheme == "s3":
                s3_endpoint = os.environ.get('S3_CUSTOM_ENDPOINT_URL')
                if s3_endpoint:
                    storage_options['client_kwargs'] = {
                        'endpoint_url': s3_endpoint
                    }

                import s3fs
                s3 = s3fs.S3FileSystem(anon=False, **storage_options)
                s3_store = s3fs.S3Map(root=f'{up.netloc}{up.path}', s3=s3, check=False)
                self.source = zarr.group(store=s3_store, overwrite=False)
            else:
                self.source = zarr.open_group(source, mode="r", storage_options=storage_options, cache_attrs=False)

        self.storage_options = storage_options

        log.info("Randomise: %s", self.randomise)
        log.info("Max training samples: %s", self.maximum_training_samples)
        log.info("Max validation samples: %s", self.maximum_validation_samples)
        if self.split_test_index_array_path:
            self.split_test_index_array = np.array(self.source[self.split_test_index_array_path])
        if self.split_train_index_array_path:
            self.split_train_index_array = np.array(self.source[self.split_train_index_array_path])

        if self.class_weights is not None:
            if isinstance(self.class_weights, str): # We received a path inside the `source`:
                self.class_weights = np.array(self.source[self.class_weights])
            elif isinstance(self.class_weights, Array):
                self.class_weights = np.array(self.class_weights)
            else:
                raise NotImplementedError("No support for the specified type of class_weights")

        if self.sample_weights is not None:
            if isinstance(self.sample_weights, str): # We received a path inside the `source`:
                self.sample_weights = np.array(self.source[self.sample_weights])
            elif isinstance(self.sample_weights, Array):
                self.sample_weights = np.array(self.sample_weights)
            elif isinstance(self.sample_weights, list):
                self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
            else:
                raise NotImplementedError("No support for the specified type of sample_weights")

        for input_name, input_path in inputs.items():
            shape = None
            kwds = {}
            if isinstance(input_path, dict):
                dask_chunk_size = input_path.get('dask_chunk_size')
                if dask_chunk_size is not None:
                    kwds.update(chunks=dask_chunk_size)
                standardizers = input_path.get('standardizers')
                shape = input_path.get('sample_reshape', None)
                input_path = input_path.get('component')
                if standardizers is not None:
                    zarr_array = from_zarr(source, standardizers, storage_options=self.storage_options)
                    self.input_standardizers[input_name] = np.array(zarr_array)
            kwds.update(component=input_path)
            #zarr_array = from_zarr(source, **kwds, storage_options=self.storage_options)
            zarr_array = self.source[input_path].view(fill_value=0)
            #zarr_array = from_zarr(self.source[input_path])
            #zarr_array = zarr_array.view(fill_value=0)
            self.inputs[input_name] = zarr_array
            if shape is not None:
                1/0
                self.inputs[input_name] = self.inputs[input_name].reshape(shape)
        self.outputs = {}
        for output_name, output_path in targets.items():
            shape = None
            kwds = {}
            if isinstance(output_path, dict):
                dask_chunk_size = output_path.get('dask_chunk_size')
                if dask_chunk_size is not None:
                    kwds.update(chunks=dask_chunk_size)
                shape = output_path.get('sample_reshape', None)
                output_path = output_path.get('component')
            kwds.update(component=output_path)
            kwds.update(storage_options=self.storage_options)
            zarr_array = self.source[output_path].view(fill_value=0)
            self.outputs[output_name] = zarr_array
            if shape is not None:
                1/0
                outer_dimension = self.outputs[output_name].shape[0]
                self.outputs[output_name] = self.outputs[output_name].reshape((outer_dimension,) + tuple(shape))

    def __str__(self):
        return f"{self.source}"

    def get_training(self, batch_size: int) -> ArraySequence:
        """
        Generates a training sequence to be used by external consumers.

        :param batch_size: Batch size used by the :class:`ArraySequence`
        :return: returns an :class:`ArraySequence` containing the data or a subset of it
        """
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_train_index_array,
                             randomise=self.randomise, maximum_samples=self.maximum_training_samples,
                             standardisers=self.input_standardizers, sample_weights=self.sample_weights, slice_timestamps=self.slice_timestamps, flat=self.flat)

    def get_validation(self, batch_size: int) -> ArraySequence:
        if self.split_test_index_array is None:
            return None
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_test_index_array,
                             randomise=self.randomise, maximum_samples=self.maximum_validation_samples,
                             standardisers=self.input_standardizers, slice_timestamps=self.slice_timestamps, flat=self.flat)

    def get_test(self, batch_size: int) -> ArraySequence:
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_test_index_array,
                             randomise=self.randomise, maximum_samples=self.maximum_validation_samples,
                             standardisers=self.input_standardizers, slice_timestamps=self.slice_timestamps, flat=self.flat)

    def get_mask(self):
        raise NotImplementedError()

    def get_class_weights(self):
        if self.class_weights is None:
            return None
        else:
            return np.array(self.class_weights)

    def get_sample_weights(self):
        if self.sample_weights is None:
            return None
        else:
            return np.array(self.sample_weights)

def flatten_generator_data(data):
    keys = sorted(list(data.keys()))
    result = []
    for key in keys:
        value = data[key]
        value = value.reshape(value.shape[1:])  ## Remove batch dimension. Works as long batch size is 1
        result.append(value)
    return tuple(result)


class ZarrArrayLoaderTFData(ZarrArrayLoader):
    def __init__(self, *args, **kwargs):
        super(ZarrArrayLoaderTFData, self).__init__(*args, **kwargs)

    def get_tfdataset(self, data):
        if len(data) == 0:
            return None

        # Get first entry so we discover the types and shapes
        first_inputs, first_outputs = data[0]

        input_dtypes = [tf.dtypes.as_dtype(e.dtype) for e in flatten_generator_data(first_inputs)]
        output_dtypes = [tf.dtypes.as_dtype(e.dtype) for e in flatten_generator_data(first_outputs)]
        output_types = (tuple(input_dtypes), tuple(output_dtypes))

        #
        def _standardise_data():
            for input_data, output_data in data.__iter__()():
                inputs = flatten_generator_data(input_data)
                outputs = flatten_generator_data(output_data)
                yield inputs, outputs

        # Create the tf Dataset
        return tf.data.Dataset.from_generator(_standardise_data, output_types=output_types)

    def get_training(self, batch_size: int):
        training_data = super(ZarrArrayLoaderTFData, self).get_training(1)
        return self.get_tfdataset(training_data)

    def get_validation(self, batch_size: int):
        validation_data = super(ZarrArrayLoaderTFData, self).get_validation(1)
        return self.get_tfdataset(validation_data)
