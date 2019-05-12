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

from hugin.io.loader import DatasetLoader
from hugin.io.scanners import FileScanner, FilesystemScanner


class BaseDatasetLoader(object):
    def __init__(self, data_pattern,
                 id_format,
                 type_format,
                 dataset_filter=lambda x, y, z: True,
                 custom_attributes={},
                 input_source=None,  # Source from where to read the data
                 validation_source=None,  # Source from where to read the validation data
                 prepend_path="",
                 validation_percent=0,
                 randomise_datasets=False,
                 rasterio_env={},
                 cache_io=False,
                 randomise_training_datasets=False,
                 randomise_validation_datasets=False,
                 ):
        self.data_pattern = data_pattern
        self.custom_attributes = custom_attributes
        self.id_format = id_format
        self.type_format = type_format
        self.dataset_filter = dataset_filter
        self.input_source = input_source
        self.cache_io = cache_io
        self.validation_source = validation_source
        self.validation_percent = validation_percent
        self.randomise_datasets = randomise_datasets
        self.randomise_training_datasets = randomise_training_datasets
        self.randomise_validation_datasets = randomise_validation_datasets
        self.prepend_path = prepend_path
        self.rasterio_env = rasterio_env

    def set_input_source(self, input_source):
        self.input_source = input_source

    def get_dataset_loader(self):
        dataset_scanner = self.get_dataset_scanner()
        return self.build_dataset_loaders(dataset_scanner.get_training_datasets(),
                                          dataset_scanner.get_validation_datasets())

    def build_dataset_loaders(self, training_datasets, validation_datasets):
        training_loader = DatasetLoader(training_datasets, rasterio_env=self.rasterio_env, _cache_data=self.cache_io)
        validation_loader = DatasetLoader(validation_datasets, rasterio_env=self.rasterio_env,
                                          _cache_data=self.cache_io)
        return (training_loader, validation_loader)

    def get_dataset_scanner(self):
        raise NotImplementedError()


class FileSystemLoader(BaseDatasetLoader):
    def __init__(self, *args, **kws):
        BaseDatasetLoader.__init__(self, *args, **kws)

    def get_dataset_scanner(self):
        scanner = FilesystemScanner(self.input_source,
                                    self.data_pattern,
                                    validation_source=self.validation_source,
                                    custom_attributes=self.custom_attributes,
                                    id_format=self.id_format,
                                    type_format=self.type_format,
                                    filter=self.dataset_filter,
                                    prepend_path=self.prepend_path,
                                    validation_percent=self.validation_percent,
                                    randomise=self.randomise_datasets,
                                    randomise_training=self.randomise_training_datasets,
                                    randomise_validation=self.randomise_validation_datasets)
        return scanner


class FileLoader(BaseDatasetLoader):
    def __init__(self, *args, **kws):
        BaseDatasetLoader.__init__(self, *args, **kws)

    def get_dataset_scanner(self):
        scanner = FileScanner(self.input_source,
                              self.data_pattern,
                              custom_attributes=self.custom_attributes,
                              id_format=self.id_format,
                              type_format=self.type_format,
                              filter=self.dataset_filter,
                              prepend_path=self.prepend_path,
                              validation_percent=self.validation_percent,
                              randomise=self.randomise_datasets)
        return scanner


class SceneDummyLoader(BaseDatasetLoader):
    def __init__(self, scene):
        pass

    def get_dataset_loader(self):
        dataset_scanner = self.get_dataset_scanner()
        return self.build_dataset_loaders(dataset_scanner.get_training_datasets(),
                                          dataset_scanner.get_validation_datasets())
