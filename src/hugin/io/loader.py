# -*- coding: utf-8 -*-
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

import logging
import threading
from collections import OrderedDict
from queue import Queue


import backoff
import math
import numpy as np
from keras.utils import to_categorical

from rasterio.windows import Window

log = logging.getLogger(__name__)


class NullFormatConverter(object):
    def __init__(self):
        pass

    def __call__(self, entry):
        return entry


class CategoricalConverter(object):
    def __init__(self, num_classes=2):
        self._num_classes = num_classes

    def __call__(self, entry):
        # entry = entry.reshape(entry.shape + (1, ))
        cat = to_categorical(entry, self._num_classes)
        return cat


class BinaryCategoricalConverter(CategoricalConverter):
    """
    Converter used for representing Urband3D Ground Truth / GTI
    """

    def __init__(self, do_categorical=True):
        CategoricalConverter.__init__(self, 2)
        self.do_categorical = do_categorical

    def __call__(self, entry):
        entry = entry > 0
        if self.do_categorical:
            return CategoricalConverter.__call__(self, entry)
        return entry


class MultiClassToBinaryCategoricalConverter(BinaryCategoricalConverter):
    def __init__(self, class_label, do_categorical=True):
        BinaryCategoricalConverter.__init__(self, do_categorical)
        self.class_label = class_label

    def __call__(self, entry):
        entry = entry.copy()
        entry[entry != self.class_label] = 0
        return BinaryCategoricalConverter.__call__(self, entry)


class ColorMapperConverter(object):
    def __init__(self, color_map):
        self._color_map = color_map
        raise NotImplementedError()

    def __call__(self):
        pass


def adapt_shape_and_stride(scene, base_scene, shape, stride, offset='ul'):
    if scene == base_scene:
        return shape, stride
    x_geo_orig, y_geo_orig = base_scene.xy(shape[0], shape[1], offset=offset)

    computed_shape = scene.index(x_geo_orig, y_geo_orig)
    computed_stride, _ = scene.index(*base_scene.xy(stride, stride, offset=offset))

    return computed_shape, computed_stride

class TileGenerator(object):
    def __init__(self, scene, shape=None, mapping=(), stride=None, swap_axes=False, normalize=False, copy=False):
        """
        @shape: specify the shape of the window/tile. None means the window covers the whole image
        @stride: the stride used for moving the windows
        @mapping: how to map bands to the dataset data
        """
        self._scene = scene
        self._shape = shape
        self._mapping = mapping
        self._stride = stride
        self.swap_axes = swap_axes
        self._normalize = normalize
        self._count = 0
        self._copy = copy
        if self._stride is None and self._shape is not None:
            self._stride = self._shape[0]

    def __iter__(self):
        return self.generate_tiles_for_dataset()

    def __len__(self):
        if not self._shape:
            return 0
        input_mapping = self._mapping
        mapped_scene = augment_mapping_with_datasets(self._scene, input_mapping)

        # pick the bigest image
        max_component_area = 0
        max_component = None
        for entry in mapped_scene:
            backing_store = entry["backing_store"]
            area = backing_store.height * backing_store.width
            if area > max_component_area:
                max_component_area = area
                max_component = backing_store

        scene_width = max_component.width
        scene_height = max_component.height

        tile_width, tile_height = self._shape
        if tile_width != self._stride:
            num_horiz = math.ceil((scene_width - tile_width) / float(self._stride) + 2)
        else:
            num_horiz = math.ceil((scene_width - tile_width) / float(self._stride) + 1)
        if tile_height != self._stride:
            num_vert = math.ceil((scene_height - tile_height) / float(self._stride) + 2)
        else:
            num_vert = math.ceil((scene_height - tile_height) / float(self._stride) + 1)

        return int(num_horiz * num_vert)

    @backoff.on_exception(backoff.expo, OSError, max_time=120)
    def read_window(self, dset, band, window):
        data = dset.read(band, window=window)

        self._count += 1
        return data

    def _generate_tiles_for_mapping(self, dataset, mapping, target_shape, target_stride):
        if not mapping: return

        window_width, window_height = target_shape

        mapping_level_preprocessing = mapping.get('preprocessing', [])
        augmented_mapping = augment_mapping_with_datasets(dataset, mapping)


        image_height, image_width = augmented_mapping[0]["backing_store"].shape

        tile_width, tile_height = target_shape
        stride = target_stride
        if tile_width != stride:
            xtiles = math.ceil((image_width - tile_width) / float(stride) + 2)
        else:
            xtiles = math.ceil((image_width - tile_width) / float(stride) + 1)

        if tile_height != stride:
            ytiles = math.ceil((image_height - tile_height) / float(stride) + 2)
        else:
            ytiles = math.ceil((image_height - tile_height) / float(stride) + 1)

        ytile = 0

        data = []
        buffer = None
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

                window = Window(x_start, y_start, window_width, window_height)
                count = 0
                for map_entry in augmented_mapping:
                    count += 1
                    backing_store = map_entry["backing_store"]
                    channel = map_entry["channel"]
                    normalization_value = map_entry.get("normalize", None)
                    transform_expression = map_entry.get("transform", None)
                    preprocessing_callbacks = map_entry.get("preprocessing", [])

                    band = self.read_window(backing_store, channel, window)
                    if normalization_value is not None and normalization_value != 1:
                        band = band / normalization_value
                    if transform_expression is not None:
                        raise NotImplementedError("Snuggs expressions are currently not implemented")

                    all_callbacks = mapping_level_preprocessing + preprocessing_callbacks
                    for callback in all_callbacks:
                        band = callback(band)

                    if buffer is None:
                        buffer = np.zeros((len(augmented_mapping),) + band.shape, dtype=band.dtype)
                    if buffer.dtype != band.dtype:
                        buffer = buffer.astype(np.find_common_type([buffer.dtype, band.dtype], []))
                    buffer[count - 1] = band

                img_data = buffer if not self._copy else buffer.copy()
                count = 0

                if self.swap_axes:
                    img_data = np.swapaxes(np.swapaxes(img_data, 0, 1), 1, 2)
                yield img_data

    def generate_tiles_for_dataset(self):
        input_mapping, output_mapping = self._mapping
        if output_mapping is None:
            output_mapping = {}
        output_generators = {}
        input_generators = {}
        primary_mapping = [v for k, v in input_mapping.items() if v.get("primary", False)][0]
        primary_shape = primary_mapping['window_shape']
        primary_stride = primary_mapping['stride']
        primary_channels = primary_mapping['channels']
        primary_base_scene = self._scene[primary_channels[0][0]]

        for mapping_name, mapping in input_mapping.items():
            base_channel = mapping['channels'][0][0]
            target_shape, target_stride = mapping["window_shape"], mapping["stride"]
            input_generators[mapping_name] = self._generate_tiles_for_mapping(self._scene, mapping, target_shape,
                                                                              target_stride)

        for mapping_name, mapping in output_mapping.items():
            base_channel = mapping['channels'][0][0]
            target_shape, target_stride = mapping["window_shape"], mapping["stride"]
            output_generators[mapping_name] = self._generate_tiles_for_mapping(self._scene, mapping, target_shape,
                                                                               target_stride)

        while True:
            inputs = {}
            outputs = {}
            try:
                for mapping_name, generator in input_generators.items():
                    inputs[mapping_name] = next(generator)
                for mapping_name, generator in output_generators.items():
                    outputs[mapping_name] = next(generator)
                yield (inputs, outputs)
            except StopIteration:
                return



def augment_mapping_with_datasets(dataset, mapping):
    augmented_mapping = []
    if 'channels' in mapping:
        mapping = mapping['channels']
    for entry in mapping:
        if isinstance(entry, dict):
            new_entry = entry.copy()
            if "preprocessing" not in new_entry:
                new_entry["preprocessing"] = []
        elif isinstance(entry, list) or isinstance(entry, tuple):
            new_entry = OrderedDict({
                "type": entry[0],
                "channel": entry[1],
                "normalize": 1,
                "preprocessing": []
            })
            if len(entry) > 2:
                new_entry["normalize"] = entry[2]
        else:
            raise NotImplementedError("Unsupported format for mapping")

        new_entry['backing_store'] = dataset[new_entry["type"]]
        augmented_mapping.append(new_entry)

    return augmented_mapping


def make_categorical(y, num_classes=None):
    from keras.utils import to_categorical
    cat = to_categorical(y, num_classes)
    return cat


def make_categorical2(entry, num_classes=None):
    input_shape = entry.shape
    if input_shape and input_shape[0] == 1:
        input_shape = tuple(input_shape[1:])
    flaten = entry.ravel()
    flaten[:] = flaten[:] > 0  # make it binary
    if not num_classes:
        num_classes = np.max(flaten) + 1

    categorical = np.zeros((flaten.shape[0], num_classes))
    categorical[np.arange(flaten.shape[0]), flaten] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class DataGenerator(object):
    def __init__(self,
                 datasets,
                 batch_size,
                 input_mapping,
                 output_mapping,
                 loop=True,
                 format_converter=NullFormatConverter(),
                 swap_axes=False,
                 postprocessing_callbacks=[],
                 optimise_huge_datasets=True,
                 default_window_size=None,
                 default_stride_size=None,
                 copy=False):


        self._copy = copy

        if type(input_mapping) is list or type(input_mapping) is tuple:
            input_mapping = self._convert_input_mapping(input_mapping)
        if type(output_mapping) is list or type(output_mapping) is tuple:
            output_mapping = self._convert_output_mapping(output_mapping)

        if len(input_mapping.keys()) == 1: # Single mapping. Should be primary by default
            input_mapping[list(input_mapping.keys())[0]]['primary'] = True

        primary_mapping = [input_mapping[m] for m in input_mapping if input_mapping[m].get('primary', False)]



        if len(primary_mapping) > 1:
            raise TypeError("More then one primary mappings")
        elif not primary_mapping:
            raise TypeError("No primary mapping")
        primary_mapping = primary_mapping[0]
        self._primary_mapping = primary_mapping

        self._datasets = datasets

        primary_mapping_type_id = self._primary_mapping['channels'][0][0]
        first = next(self._datasets)
        scene_id, scene_data = first

        self._datasets.reset()
        if self._datasets.datasets:
            if primary_mapping_type_id not in scene_data:
                raise KeyError("Unknown type id: %s", primary_mapping_type_id)
            self.primary_scene = scene_data[primary_mapping_type_id]
        else:  # No scenes available
            self.primary_scene = None

        if default_window_size is not None:
            window_size = default_window_size
        elif self.primary_scene is not None:
            window_size = (self.primary_scene.width, self.primary_scene.height)
            log.warning("Missing default window_size using computed one: %s", window_size)
        else:
            window_size = None
        self.primary_window_shape = primary_mapping.get('window_shape', window_size)
        if self.primary_window_shape is None:
            raise AttributeError("No way for computing window shape")

        if 'window_shape' not in primary_mapping:
            primary_mapping['window_shape'] = self.primary_window_shape

        if 'stride' not in primary_mapping:
            if default_stride_size is not None:
                self.primary_stride = default_stride_size
            else:
                self.primary_stride = self.primary_window_shape[0]
        else:
            self.primary_stride = primary_mapping['stride']

        if 'stride' not in primary_mapping:
            primary_mapping['stride'] = self.primary_stride

        self._mapping = (input_mapping, output_mapping)

        for entry in self._mapping:
            if entry is None: # Probablyy missing GTI
                continue
            for mapping_type, mapping_value in entry.items():
                if 'window_shape' not in mapping_value or 'stride' not in mapping_value:
                    scene_type = mapping_value['channels'][0][0]
                    coregistration = mapping_value.get('coregistration', {})
                    offset_type = coregistration.get('offset', "ul")
                    current_scene = scene_data[scene_type]
                    entry_shape, entry_stride = adapt_shape_and_stride(current_scene, self.primary_scene, self.primary_window_shape, self.primary_stride, offset=offset_type)
                    mapping_value['window_shape'] = entry_shape
                    mapping_value['stride'] = entry_stride

        self._swap_axes = swap_axes
        self._postprocessing_callbacks = postprocessing_callbacks
        self._num_tiles = None
        self._optimise_huge_datasets = optimise_huge_datasets
        self._format_converter = format_converter
        if batch_size is None:
            self._batch_size = len(self)
        else:
            self._batch_size = batch_size

        if loop:
            self.__output_generator_object = self._looping_output_generator()
        else:
            self.__output_generator_object = self._output_generator()

    @property
    def mapping_sizes(self):
        input_sizes = {}
        output_sizes = {}
        input_mapping, output_mapping = self._mapping
        for input_name, input_value in input_mapping.items():
            input_sizes[input_name] = tuple(input_value["window_shape"]) + (len(input_value["channels"]), )
        if output_mapping is not None:
            for output_name, output_value in output_mapping.items():
                output_sizes[output_name] = tuple(output_value["window_shape"]) + (len(output_value["channels"]), )

        return (input_sizes, output_sizes)


    def _convert_mapping(self, endpoint, mapping, primary):
        new_mapping = {}
        mapping_name = endpoint + "_1"
        new_mapping[mapping_name] = {
            'primary': primary,
            'channels': mapping
        }
        if hasattr(self, 'primary_window_shape') and self.primary_window_shape is not None:
            new_mapping[mapping_name]['window_shape'] = self.primary_window_shape
        if hasattr(self, 'primary_stride') and self.primary_stride is not None:
            new_mapping[mapping_name]['stride'] = self.primary_stride

        return new_mapping

    def _convert_input_mapping(self, mapping, primary=True):
        return self._convert_mapping("input", mapping, primary)

    def _convert_output_mapping(self, mapping, primary=False):
        return self._convert_mapping("output", mapping, primary)

    def next(self):
        return self.__next__(self)

    def __len__(self):
        """This is a huge resource hog! Avoid it!"""
        if self._num_tiles is not None:
            return self._num_tiles
        self._num_tiles = 0
        dataset_loader = self._datasets

        input_stride = self.primary_stride
        input_window_shape = self.primary_window_shape
        input_channels = self._primary_mapping['channels']
        for scene_id, scene_data in dataset_loader:
            tile_generator = TileGenerator(scene_data,
                                           input_window_shape,
                                           stride=input_stride,
                                           mapping=input_channels,
                                           swap_axes=self._swap_axes)
            self._num_tiles += len(tile_generator)
            if self._optimise_huge_datasets:
                self._num_tiles = self._num_tiles * len(dataset_loader)
                break
        dataset_loader.reset()
        return self._num_tiles

    def __next__(self):
        return next(self.__output_generator_object)

    def __iter__(self):
        return self

    def _looping_output_generator(self):
        while True:
            for data in self._output_generator():
                yield data

    def _flaten_simple_input(self, inp):
        if len(inp.keys()) != 1:
            return inp
        main_key = list(inp.keys())[0]
        main_value = inp[main_key]
        return main_value

    def _output_generator(self):
        dataset_loader = self._datasets
        count = 0

        input_data = {}
        output_data = {}

        scene_count = 0
        callbacks = self._postprocessing_callbacks

        for scene in dataset_loader:
            scene_count += 1
            scene_id, scene_data = scene
            tile_generator = TileGenerator(scene_data,
                                           self.primary_window_shape,
                                           stride=self.primary_stride,
                                           mapping=self._mapping,
                                           swap_axes=self._swap_axes)

            for entry in tile_generator:
                count += 1

                input_patches, output_patches = entry

                for callback in callbacks:
                    input_patches, output_patches = callback(input_patches, output_patches)

                for in_patch_name, in_patch_value in input_patches.items():
                    if in_patch_name not in input_data:
                        input_data[in_patch_name] = np.zeros((self._batch_size,) + in_patch_value.shape, dtype=in_patch_value.dtype)
                    input_data[in_patch_name][count - 1] = in_patch_value

                for out_patch_name, out_patch_value in output_patches.items():
                    new_path_value = self._format_converter(out_patch_value)
                    if out_patch_name not in output_data:
                        output_data[out_patch_name] = np.zeros((self._batch_size, ) + new_path_value.shape, dtype=new_path_value.dtype)
                    output_data[out_patch_name][count - 1] = new_path_value

                if count == self._batch_size:
                    in_arrays = {}
                    for k, v in input_data.items():
                        in_arrays[k] = v if not self._copy else v.copy()
                    out_arrays = {}
                    for k,v in output_data.items():
                        out_arrays[k] = v if not self._copy else v.copy()

                    yield (self._flaten_simple_input(in_arrays), self._flaten_simple_input(out_arrays))
                    count = 0

        if count > 0:
            in_arrays = {}
            for k, v in input_data.items():
                subset = v[:count,:,:]
                in_arrays[k] = subset if not self._copy else subset.copy()
            out_arrays = {}
            for k, v in output_data.items():
                subset = v[:count, :, :]
                out_arrays[k] = subset if not self._copy else subset.copy()

            yield (self._flaten_simple_input(in_arrays), self._flaten_simple_input(out_arrays))



class ThreadedDataGenerator(threading.Thread):
    def __init__(self, data_generator, queue_size=4):
        self._queue_size = queue_size
        self._data_generator = data_generator
        self._q = Queue(maxsize=self._queue_size)
        self._len = len(self._data_generator)
        self._data_generator_flow = self._flow_data()
        threading.Thread.__init__(self)
        self.setName("ThreadedDataGenerator")
        self.setDaemon(True)
        self.start()

    def run(self):
        for d in self._data_generator:
            self._q.put(d)
        self._q.put(None)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._data_generator_flow)

    def _flow_data(self):
        while True:
            d = self._q.get()
            if d is None:
                break
            yield d


if __name__ == "__main__":
    pass
