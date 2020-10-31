import math
import numpy as np
from logging import getLogger

from .dataset_loaders import ArrayLoader
from dask.array import from_zarr, Array
from tensorflow.keras.utils import Sequence

log = getLogger(__name__)

def _data_generator(source_array, batch_size : int):
    yield None

class ArrayDataGenerator(object):
    def __init__(self, input_component_mapping: dict, output_component_mapping: dict, batch_size: int):
        self.input_component_mapping = input_component_mapping
        self.output_component_mapping = output_component_mapping
        self.batch_size = batch_size if batch_size is not None else 1
        self._index_iterator = self.__get_batch_indexes()

    def __get_batch_indexes(self):
        while True:
            total_entries = len(self)
            number_of_batches = total_entries // self.batch_size
            remainder = total_entries % self.batch_size
            for batch_idx in range(0, number_of_batches):
                batch_start_idx = batch_idx*self.batch_size
                batch_end_idx = batch_start_idx + self.batch_size
                yield (batch_start_idx, batch_end_idx)
            if remainder > 0:
                yield (batch_end_idx, total_entries)

    def get_input_shapes(self):
        shapes = {}
        for key, value in self.input_component_mapping.items():
            shapes[key] = value.shape[1:]
        return shapes

    def __len__(self):
        one_array = self.input_component_mapping[list(self.input_component_mapping.keys())[0]]
        return len(one_array)

    def __next__(self):
        return self.next()

    def next(self):
        start_index, end_index = next(self._index_iterator)
        inputs = {}
        outputs = {}
        for key, value in self.input_component_mapping.items():
            inputs[key] = value[start_index:end_index, ...]
        for key, value in self.output_component_mapping.items():
            pass
        return (inputs, outputs)

class ArraySequence(Sequence):
    def __init__(self,
                 input_component_mapping: dict,
                 output_component_mapping: dict,
                 batch_size: int,
                 selected_indices: Array = None,
                 randomise: bool = False,
                 maximum_samples: int = None):

        self.input_component_mapping = input_component_mapping
        self.output_component_mapping = output_component_mapping
        self.selected_indices = selected_indices
        self.batch_size = batch_size if batch_size is not None else 1
        self.randomise = randomise
        self.maximum_samples = maximum_samples

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
        data = np.array(v) # Convert to NumPy array to prevent issues, memory impact should be minimal as there should be a limited amount if indices
        self.__selected_indices = data

    def on_epoch_end(self):
        if self.randomise:
            np.random.shuffle(self.__selected_indices)

    def __len__(self):
        length = math.ceil(len(self.selected_indices) / self.batch_size)
        return int(length)

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
                inputs[key].append(value[idx])
            for key, value in self.output_component_mapping.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value[idx])

        inputs = {k:np.stack(v) for k,v in inputs.items()}
        outputs = {k:np.stack(v) for k, v in outputs.items()}

        return inputs, outputs


    def get_input_shapes(self):
        shapes = {}
        for key, value in self.input_component_mapping.items():
            shapes[key] = value.shape[1:]
        return shapes


class ZarrArrayLoader(ArrayLoader):
    def __init__(self,
                 source,
                 inputs: dict,
                 targets: dict,
                 split_test_index_array: Array = None,
                 split_train_index_array: Array = None,
                 randomise: bool = False,
                 maximum_training_samples: int = None,
                 maximum_validation_samples: int = None):
        super(ZarrArrayLoader, self).__init__()
        self.inputs = {}
        self.split_test_index_array_path = split_test_index_array
        self.split_train_index_array_path = split_train_index_array
        self.split_test_index_array = None
        self.split_train_index_array = None
        self.randomise = randomise
        self.maximum_training_samples = maximum_training_samples
        self.maximum_validation_samples = maximum_validation_samples
        log.info("Randomise: %s", self.randomise)
        log.info("Max training samples: %s", self.maximum_training_samples)
        log.info("Max validation samples: %s", self.maximum_validation_samples)
        if self.split_test_index_array_path:
            self.split_test_index_array = from_zarr(source, component=self.split_test_index_array_path)
        if self.split_train_index_array_path:
            self.split_train_index_array = from_zarr(source, component=self.split_train_index_array_path)

        for input_name, input_path in inputs.items():
            shape = None
            kwds = {}
            if isinstance(input_path, dict) :
                dask_chunk_size = input_path.get('dask_chunk_size')
                if dask_chunk_size is not None:
                    kwds.update(chunks=dask_chunk_size)
                standardizers = input_path.get('standardizers')
                shape = input_path.get('sample_reshape', None)
                input_path = input_path.get('component')
            kwds.update(component=input_path)
            self.inputs[input_name] = from_zarr(source, **kwds)
            #log.info("Input %s has shape %s chunks: %s", input_name, self.inputs[input_name].shape, self.inputs[input_name].chunks)
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
            self.outputs[output_name] = from_zarr(source, **kwds)
            if shape is not None:
                outer_dimension = self.outputs[output_name].shape[0]
                self.outputs[output_name] = self.outputs[output_name].reshape((outer_dimension,) + tuple(shape))


    def get_training(self, batch_size : int) -> _data_generator:
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_train_index_array, randomise=self.randomise, maximum_samples=self.maximum_training_samples)

    def get_validation(self, batch_size : int) -> _data_generator:
        if self.split_test_index_array is None:
            return None
        return ArraySequence(self.inputs, self.outputs, batch_size, selected_indices=self.split_test_index_array, randomise=self.randomise, maximum_samples=self.maximum_validation_samples)

    def get_mask(self):
        raise NotImplementedError()
