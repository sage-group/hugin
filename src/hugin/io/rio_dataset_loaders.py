# -*- coding: utf-8 -*-
__license__ = """Copyright 2023 West University of Timisoara

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
import os
import random
from hashlib import sha224
from tempfile import NamedTemporaryFile, mkdtemp
from urllib.parse import urlparse

import rasterio
from rasterio import MemoryFile
from rasterio.io import DatasetReader

from ..engine.core import RasterGenerator
from ..tools.IOUtils import IOUtils


class DatasetGenerator(object):
    def __init__(
        self,
        datasets,
        loop=False,
        randomise_on_loop=True,
        rasterio_env={},
        _cache_data=False,
        _delete_temporary_cache=True,
    ):
        self._datasets = list(datasets)
        self.rasterio_env = rasterio_env
        self._curent_position = 0
        self.loop = loop
        self._cache_data = _cache_data
        self.randomise_on_loop = randomise_on_loop  # Reshuffle data after loop
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
        from geopandas import GeoDataFrame

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
        env = getattr(self, "rasterio_env", {})
        self._curent_position += 1
        entry_name, entry_components = entry
        new_components = {}
        cache_data = self._cache_data
        use_tensorflow_io = False
        for component_name, component_path_entry in entry_components.items():
            if isinstance(
                component_path_entry, (RasterGenerator, GeoDataFrame, MemoryFile)
            ):
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
                if url_components.path.startswith("/vsigs/"):
                    cache_data = True  # We should check if we run inside GCP ML Engine
                    use_tensorflow_io = True
                    component_path = url_components.path[6:]
                    component_path = "gs:/" + component_path
            else:
                if url_components.scheme == "file":
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
                        dataset_path = os.path.join(
                            dataset_path, os.path.basename(component_path)
                        )
                        if not IOUtils.file_exists(dataset_path):
                            f = IOUtils.open_file(dataset_path, "wb")
                            f.write(data)
                            f.close()
                        component_src = self.get_component_file_descriptor(dataset_path)
                    else:
                        with NamedTemporaryFile() as tmpfile:
                            tmpfile.write(data)
                            tmpfile.flush()
                            component_src = self.get_component_file_descriptor(
                                tmpfile.name
                            )
                else:
                    component_src = self.get_component_file_descriptor(
                        local_component_path
                    )
                new_components[component_name] = component_src

        # Trigger the generation of the dynamic components
        for component_name, component_path in new_components.items():
            if isinstance(component_path, RasterGenerator):
                new_components[component_name] = component_path(new_components)

        return entry_name, new_components

    def get_component_file_descriptor(self, file_path):
        return rasterio.open(file_path)
