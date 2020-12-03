import math
import os
import sys
import numpy as np
from logging import getLogger

from .dataset_loaders import ArrayLoader
from dask.array import from_zarr, Array
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
                 maximum_samples: int = None):

        self.input_component_mapping = input_component_mapping
        self.output_component_mapping = output_component_mapping
        self.randomise = randomise
        self.selected_indices = selected_indices
        self.batch_size = batch_size if batch_size is not None else 1
        self.maximum_samples = maximum_samples
        self.standardisers = standardisers

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
    def selected_indices(self, v : Array):
        data = np.array(v) if v is not None else v # Convert to NumPy array to prevent issues, memory impact should be minimal as there should be a limited amount if indices
        self.__selected_indices = data
        self.__shuffle_indices()

    def __shuffle_indices(self):
        if self.randomise:
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

    def __getitem__(self, idx):
        inputs = {}
        outputs = {}

        indices = self.selected_indices

        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        for idx in indices[start_idx:end_idx]:
            for key, value in self.input_component_mapping.items():
                if key not in inputs:
                    inputs[key] = []
                dask_data = value[idx]
                #log.debug(f"Fetching data for {idx} from value of {key}={value}")
                data = np.array(dask_data)
                #log.debug(f"Fetched data for {idx} from value of {key}={value}")
                standardiser = self.standardisers.get(key)  if self.standardisers else None
                if standardiser is not None:
                    data = data.astype(np.float64)
                    for i in range(0, len(standardiser)):
                        channel_standardizer = standardiser[i]
                        old_shape = data[..., i].shape
                        data[..., i] = channel_standardizer.transform(data[..., i].reshape(-1, 1), copy=False).reshape(old_shape)
                inputs[key].append(data)

            for key, value in self.output_component_mapping.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value[idx])

        source = {k:np.array(np.stack(v)) for k,v in inputs.items()}
        targets = {k:np.array(np.stack(v)) for k, v in outputs.items()}
        return source, targets


    def get_input_shapes(self):
        shapes = {}
        for key, value in self.input_component_mapping.items():
            shapes[key] = value.shape[1:]
        return shapes

class ZarrArrayLoader(ArrayLoader):
    def __init__(self,
                 inputs: dict,
                 targets: dict,
                 source=None,
                 split_test_index_array: Array = None,
                 split_train_index_array: Array = None,
                 randomise: bool = False,
                 maximum_training_samples: int = None,
                 maximum_validation_samples: int = None):
        super(ZarrArrayLoader, self).__init__()
        self.inputs = {}
        self.input_standardizers = {}
        self.split_test_index_array_path = split_test_index_array
        self.split_train_index_array_path = split_train_index_array
        self.split_test_index_array = None
        self.split_train_index_array = None
        self.randomise = randomise
        self.maximum_training_samples = maximum_training_samples
        self.maximum_validation_samples = maximum_validation_samples
        if source is None:
            if 'DATASOURCE_URL' not in os.environ:
                raise TypeError("Missing source specification. Should be specified directly or as the `DATASOURCE_URL` environment variable")
            else:
                source = os.environ['DATASOURCE_URL']
                log.info(f"Using storage configuration from environment file `DATASOURCE_URL`: {source}")
        up = urlparse(source)
        storage_options = {}
        if not up.scheme:
            self.source = source
        else:
            source = f"{up.scheme}://{up.netloc}{up.path}"
            for k,v in parse_qs(up.query, keep_blank_values=True).items():
                value = v[0] if v[0] else None
                storage_options[k]=value
            if up.scheme == "s3":
                s3_endpoint = os.environ.get('S3_CUSTOM_ENDPOINT_URL')
                if s3_endpoint:
                    storage_options['client_kwargs'] = {
                        'endpoint_url': s3_endpoint
                    }

        self.storage_options = storage_options
        self.source = source
        log.info("Randomise: %s", self.randomise)
        log.info("Max training samples: %s", self.maximum_training_samples)
        log.info("Max validation samples: %s", self.maximum_validation_samples)
        if self.split_test_index_array_path:
            self.split_test_index_array = from_zarr(source, component=self.split_test_index_array_path, storage_options=storage_options)
        if self.split_train_index_array_path:
            self.split_train_index_array = from_zarr(source, component=self.split_train_index_array_path, storage_options=storage_options)

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
                    self.input_standardizers[input_name] = np.array(from_zarr(source, standardizers, storage_options=self.storage_options))
            kwds.update(component=input_path)
            print (f"Kwds: {kwds}")
            print (f"storage_options: {self.storage_options}")
            self.inputs[input_name] = from_zarr(source, **kwds, storage_options=self.storage_options)
            if shape is not None:
                self.inputs[input_name] = self.inputs[input_name].reshape(shape)
        self.outputs = {}
        for output_name, output_path in targets.items():
            shape = None
            kwds = {}
            if isinstance(output_path, dict) :
                dask_chunk_size = output_path.get('dask_chunk_size')
                if dask_chunk_size is not None:
                    kwds.update(chunks=dask_chunk_size)
                shape = output_path.get('sample_reshape', None)
                output_path = output_path.get('component')
            kwds.update(component=output_path)
            kwds.update(storage_options=self.storage_options)
            self.outputs[output_name] = from_zarr(source, **kwds)
            if shape is not None:
                outer_dimension = self.outputs[output_name].shape[0]
                self.outputs[output_name] = self.outputs[output_name].reshape((outer_dimension,) + tuple(shape))

    def __str__(self):
        return f"{self.source}"


    def get_training(self, batch_size : int) -> ArraySequence:
        """
        Generates a training sequence to be used by external consumers.

        :param batch_size: Batch size used by the :class:`ArraySequence`
        :return: returns an :class:`ArraySequence` containing the data or a subset of it
        """
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_train_index_array, randomise=self.randomise, maximum_samples=self.maximum_training_samples, standardisers=self.input_standardizers)

    def get_validation(self, batch_size : int) -> ArraySequence:
        if self.split_test_index_array is None:
            return None
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_test_index_array, randomise=self.randomise, maximum_samples=self.maximum_validation_samples, standardisers=self.input_standardizers)

    def get_test(self, batch_size : int) -> ArraySequence:
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_test_index_array, randomise=self.randomise, maximum_samples=self.maximum_validation_samples, standardisers=self.input_standardizers)

    def get_mask(self):
        raise NotImplementedError()
