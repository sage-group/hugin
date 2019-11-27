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

import atexit
import logging
import os
import random
import re
from collections import OrderedDict
from hashlib import sha224
from tempfile import NamedTemporaryFile, mkdtemp
from urllib.parse import urlparse

import geopandas as gp
import rasterio
import yaml
from geopandas import GeoDataFrame
from rasterio import RasterioIOError, MemoryFile
from rasterio.io import DatasetReader

from hugin.engine.core import RasterGenerator
from hugin.tools.IOUtils import IOUtils

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader as Loader
    from yaml import Dumper as Dumper

log = logging.getLogger(__name__)


class BaseLoader(object):

    def __init__(self,
                 data_pattern: str = None,
                 validation_source=None,
                 randomise=False,
                 randomise_training=False,
                 randomise_validation=False,
                 mapping=None,
                 type_format="{type}",
                 id_format="{name}-{idx}",
                 custom_attributes={},
                 filter=lambda dataset_id, match_components, dataset: True,
                 validation_percent=0,
                 prepend_path="",
                 rasterio_env={},
                 cache_io=False,
                 persist_file=None,
                 dynamic_types={}):
        """

        Args:
            data_pattern: Pattern used for grouping data
            validation_source:
            randomise: Whether the data should be randomised or not
            randomise_training: Whether the training set should be randomised
            randomise_validation: Whether the validation set should be randomised
            mapping: A mapping between discovered datasets and data types
            type_format: Format used for grouping between discovered datasets and types
            id_format: Format used for identifying data types
            custom_attributes:
            filter:
            validation_percent:
            prepend_path:
            rasterio_env:
            cache_io:
            persist_file:
        """

        if not data_pattern:
            raise ValueError("Missing Template")

        self.persist_file = persist_file
        self._data_pattern = data_pattern
        self._re = re.compile(data_pattern)
        self._validation_source = validation_source
        self._prepend_path = prepend_path
        self._id_format = id_format
        self._type_format = type_format

        self._randomise = randomise
        self._randomise_training = randomise_training
        self._randomise_validation = randomise_validation
        self._validation_percent = validation_percent
        self._mapping = mapping
        self._custom_attributes = custom_attributes
        self.rasterio_env = rasterio_env
        self.cache_io = cache_io

        self._datasets = OrderedDict()
        self._validation_datasets = OrderedDict()
        self._dynamic_types = dynamic_types

        self._filter = filter

        if self.persist_file and os.path.exists(self.persist_file):
            log.info("Loading datasets from persistence file: %s", self.persist_file)
            with open(self.persist_file, "r") as f:
                data = yaml.load(f, Loader=Loader)
                self._evaluation_list = data['evaluation']
                self._train_list = data['training']
        else:
            self.scan_datasets()


    def scan_datasets(self):
        self._datasets.clear()
        self.update_datasets()
        if self._validation_source:
            self.update_datasets(directory=self._validation_source, datasets=self._validation_datasets)
        self.__update_split()
        # self._evaluation_list, self._train_list
        if self.persist_file:
            persist_data = {
                'evaluation': self._evaluation_list,
                'training': self._train_list
            }
            log.info("Persisting datasets to: %s", self.persist_file)
            with open(self.persist_file, "w") as f:
                yaml.dump(persist_data, f, Dumper=Dumper)

        # Add dynamic types
        self._update_dynamic_types()

    def _update_dynamic_types(self):
        """
        Updates all datasets with dynamic entries
        Returns:

        """
        for scene_id, scene_data in self._datasets.items():
            for type_name, type_handler in self._dynamic_types.items():
                scene_data[type_name] = type_handler


    def __len__(self) -> int:
        """Computes the number of datasets

        Returns: The number of datasets

        """
        return len(self._datasets)

    def __update_split(self):
        _datasets = list(self._datasets.items())
        if self._randomise:
            random.shuffle(_datasets)

        _validation_datasets = list(self._validation_datasets.items())
        if len(_validation_datasets) > 0:
            if self._randomise_training:
                random.shuffle(_datasets)
            if self._randomise_validation:
                random.shuffle(_validation_datasets)
            self._train_list = tuple(_datasets)
            self._evaluation_list = tuple(_validation_datasets[:])
        else:
            num_evaluation = int(len(_datasets) * self._validation_percent)
            self._evaluation_list, self._train_list = tuple(_datasets[:num_evaluation]), tuple(
                _datasets[num_evaluation:])

    def get_full_datasets(self):
        return self._datasets

    def get_training_datasets(self):
        return self._train_list

    def get_validation_datasets(self):
        return self._evaluation_list

    def get_dataset_id(self, components):
        return self._id_format.format(**components)

    def get_dataset_by_id(self, dataset_id, dataset=None):
        if dataset is None:
            dataset = self._datasets
        return dataset[dataset_id]

    def remove_dataset_by_id(self, dataset_id, dataset=None):
        if dataset is None:
            dataset = self._datasets
        return dataset.pop(dataset_id)

    def update_dataset(self, dataset=None, dataset_id=None, match_components={}, dataset_path=None):
        if dataset is None:
            dataset = self._datasets

        if self._custom_attributes:
            for k, v in self._custom_attributes.items():
                match_components[k] = v(**match_components)
        if dataset_id is None and not match_components.get("__id_generated", False):
            dataset_id = self.get_dataset_id(match_components)
            match_components["__id_generated"] = True

        if dataset_id not in dataset:
            dataset[dataset_id] = {}
            components = dataset[dataset_id]
        else:
            components = dataset[dataset_id]
        dataset_type = self._type_format.format(**match_components)

        if dataset_type in components:
            raise KeyError("Already registered: %s %s %s" % (dataset_id, dataset_type, dataset_path))
        components[dataset_type] = dataset_path
        return dataset_id

    def update_datasets(self, *args, **kwargs):
        raise NotImplementedError()

    def get_dataset_loaders(self):
        return self.build_dataset_loaders(self.get_training_datasets(),
                                          self.get_validation_datasets())

    def get_dataset_loader(self, dataset, rasterio_env=None, _cache_data=None):
        rasterio_env = rasterio_env if rasterio_env is not None else self.rasterio_env
        _cache_data = _cache_data if _cache_data is not None else self.cache_io
        return DatasetGenerator((dataset, ), rasterio_env=rasterio_env, _cache_data=_cache_data)

    def build_dataset_loaders(self, training_datasets, validation_datasets):
        training_loader = DatasetGenerator(training_datasets, rasterio_env=self.rasterio_env, _cache_data=self.cache_io)
        validation_loader = DatasetGenerator(validation_datasets, rasterio_env=self.rasterio_env,
                                             _cache_data=self.cache_io)
        return (training_loader, validation_loader)



class FileLoader(BaseLoader):
    def __init__(self, input_source, *args, **kw):
        self._input_source = input_source
        BaseLoader.__init__(self, *args, **kw)

    def update_datasets(self, filter=None):
        if filter is None:
            filter = self._filter

        close_file = True
        log.info("Updateing datasets from file list: %s", self._input_source)
        if hasattr(self._input_source, 'read'):
            input_file = self._input_source
            close_file = False
        elif isinstance(self._input_source, str) and self._input_source.startswith("gs://"):
            log.info("Using tensorflow for IO")
            from tensorflow.python.lib.io.file_io import FileIO
            input_file = FileIO(self._input_source, "r")
            log.info("Tensorflow reported size: %d", input_file.size())
        else:
            input_file = open(self._input_source)

        lines = input_file.readlines()
        for line in lines:
            fpath = line.strip()
            parts = fpath.split("/")
            file_name = parts[-1]
            match = self._re.match(file_name)
            if not match:
                continue
            match_components = match.groupdict(default='')
            dataset_path = self._prepend_path + fpath
            dataset_id = self.update_dataset(match_components=match_components, dataset_path=dataset_path)
            dataset = self.get_dataset_by_id(dataset_id)
            if not filter(dataset_id, match_components, dataset):
                self.remove_dataset_by_id(dataset_id)
        if close_file:
            input_file.close()


class FileSystemLoader(BaseLoader):
    def __init__(self, input_source, *args, **kw):
        self.input_source = input_source
        BaseLoader.__init__(self, *args, **kw)

    def update_datasets(self, input_source=None, datasets=None, filter=None):
        if input_source is None:
            input_source = self.input_source
        if datasets is None:
            datasets = self.get_full_datasets()

        if filter is None:
            filter = self._filter
        for directory_entry in os.walk(input_source, followlinks=True):
            directory_name = directory_entry[0]
            directory_members = directory_entry[2]
            for file_name in directory_members:
                match = self._re.match(file_name)
                fpath = os.path.join(directory_name, file_name)
                if not match:
                    continue

                match_components = match.groupdict(default='')
                dataset_path = self._prepend_path + fpath
                dataset_id = self.update_dataset(dataset=datasets, match_components=match_components,
                                                 dataset_path=dataset_path)
                dataset = self.get_dataset_by_id(dataset_id, dataset=datasets)

                if not filter(dataset_id, match_components, dataset):
                    self.remove_dataset_by_id(dataset_id)

class DatasetGenerator(object):
    def __init__(self, datasets, loop=False, randomise_on_loop=True, rasterio_env={}, _cache_data=False, _delete_temporary_cache=True):
        self._datasets = list(datasets)
        self.rasterio_env = rasterio_env
        self._curent_position = 0
        self.loop = loop
        self._cache_data = _cache_data
        self.randomise_on_loop = randomise_on_loop # Reshuffle data after loop
        if self._cache_data:
            self._temp_dir = mkdtemp("cache", "hugin")

        def cleanup_dir(temp_dir):
            IOUtils.delete_recursively(temp_dir)

        if self._cache_data and _delete_temporary_cache:
            atexit.register(cleanup_dir, self._temp_dir)

    @property
    def loop(self):
        return self._loop

    @loop.setter
    def loop(self, val):
        self._loop = val

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, val):
        self._datasets = val

    def __len__(self):
        return len(self._datasets)

    def reset(self):
        self._curent_position = 0

    def __iter__(self):
        return self


    def __next__(self):
        length = len(self)
        if length == 0:
            raise StopIteration()
        if self._curent_position == length:
            if self._loop:
                if self.randomise_on_loop:
                    random.shuffle(self._datasets)
                self.reset()
            else:
                raise StopIteration()

        entry = self._datasets[self._curent_position]
        env = getattr(self, 'rasterio_env', {})
        self._curent_position += 1
        entry_name, entry_components = entry
        new_components = {}
        cache_data = self._cache_data
        use_tensorflow_io = False
        for component_name, component_path_entry in entry_components.items():
            if isinstance(component_path_entry, (RasterGenerator, GeoDataFrame, MemoryFile)):
                new_components[component_name] = component_path_entry
                continue
            elif isinstance(component_path_entry, GeoDataFrame):
                new_components[component_name] = component_path_entry
                continue
            elif isinstance(component_path_entry, DatasetReader):
                component_path = component_path_entry.name
            elif isinstance(component_path_entry, str):
                component_path = component_path_entry
            else:
                raise NotImplementedError("Unsupported type for component value")
            local_component_path = component_path
            url_components = urlparse(component_path)
            if not url_components.scheme:
                cache_data = False
                if url_components.path.startswith('/vsigs/'):
                    cache_data = True  # We should check if we run inside GCP ML Engine
                    use_tensorflow_io = True
                    component_path = url_components.path[6:]
                    component_path = "gs:/" + component_path
            else:
                if url_components.scheme == 'file':
                    local_component_path = url_components.path
                    use_tensorflow_io = False
                    cache_data = False

            with rasterio.Env(**env):
                if use_tensorflow_io:
                    real_path = component_path
                    data = IOUtils.open_file(real_path, "rb").read()
                    if cache_data:
                        hash = sha224(component_path.encode("utf8")).hexdigest()
                        hash_part = "/".join(list(hash)[:3])
                        dataset_path = os.path.join(self._temp_dir, hash_part)
                        if not IOUtils.file_exists(dataset_path):
                            IOUtils.recursive_create_dir(dataset_path)
                        dataset_path = os.path.join(dataset_path, os.path.basename(component_path))
                        if not IOUtils.file_exists(dataset_path):
                            f = IOUtils.open_file(dataset_path, "wb")
                            f.write(data)
                            f.close()
                        component_src = self.get_component_file_descriptor(dataset_path)
                    else:
                        with NamedTemporaryFile() as tmpfile:
                            tmpfile.write(data)
                            tmpfile.flush()
                            component_src = self.get_component_file_descriptor(tmpfile.name)
                else:
                    component_src = self.get_component_file_descriptor(local_component_path)
                new_components[component_name] = component_src

        # Trigger the generation of the dynamic components
        for component_name, component_path in new_components.items():
            if isinstance(component_path, RasterGenerator):
                new_components[component_name] = component_path(new_components)

        return entry_name, new_components

    def get_component_file_descriptor(self, file_path):
        try:
            return rasterio.open(file_path)
        except RasterioIOError:
            return gp.read_file(file_path)
