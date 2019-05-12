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
from collections import OrderedDict

import os
import random
import re

log = logging.getLogger(__name__)


class DatasetScanner(object):

    def __init__(self,
                 template=None,
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
                 prepend_path=""):

        if template:
            self._template = template
        if template:
            self._re = re.compile(template)
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
        self._datasets = OrderedDict()
        self._validation_datasets = OrderedDict()

        self._filter = filter
        self.update_datasets()
        if self._validation_source:
            self.update_datasets(directory=self._validation_source, datasets=self._validation_datasets)
        self.__update_split()

    def __len__(self):
        return len(self._datasets)

    def __update_split(self):
        _datasets = list(self._datasets.items())
        if self._randomise:
            random.shuffle(_datasets)

        _validation_datasets = list(self._validation_datasets.items())
        if len(_validation_datasets) > 0:
            # num_evaluation = int(len(_validation_datasets) * self._validation_percent)
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

    def scan_datasets(self):
        raise NotImplementedError()

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
        """"""

        # print("dataset_id={} match_components={} dataset_path={}".format(dataset_id, match_components,dataset_path))

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
            raise KeyError("Already registered: %s %s" % (dataset_type, dataset_path))
        components[dataset_type] = dataset_path
        return dataset_id


class FileScanner(DatasetScanner):
    def __init__(self, source_file, *args, **kw):
        self._source_file = source_file
        DatasetScanner.__init__(self, *args, **kw)

    def update_datasets(self, filter=None):
        if filter is None:
            filter = self._filter

        file_list = []
        log.info("Updateing datasets from file list: %s", self._source_file)
        if self._source_file.startswith("gs://"):
            log.info("Using tensorflow for IO")
            from tensorflow.python.lib.io.file_io import FileIO
            input_file = FileIO(self._source_file, "r")
            log.info("Tensorflow reported size: %d", input_file.size())
        else:
            input_file = open(self._source_file)

        lines = input_file.readlines()
        for line in lines:
            fpath = line.strip()
            parts = fpath.split("/")
            file_name = parts[-1]
            directory_name = "/".join(parts[:-1])
            match = self._re.match(file_name)
            if not match:
                continue
            match_components = match.groupdict()
            dataset_path = self._prepend_path + fpath
            dataset_id = self.update_dataset(match_components=match_components, dataset_path=dataset_path)
            dataset = self.get_dataset_by_id(dataset_id)
            if not filter(dataset_id, match_components, dataset):
                self.remove_dataset_by_id(dataset_id)
        input_file.close()


class FilesystemScanner(DatasetScanner):

    def __init__(self, directory, *args, **kw):
        self._directory = directory
        DatasetScanner.__init__(self, *args, **kw)

    def update_datasets(self, directory=None, datasets=None, filter=None):
        if directory is None:
            directory = self._directory
        if datasets is None:
            datasets = self.get_full_datasets()

        if filter is None:
            filter = self._filter
        for directory_entry in os.walk(directory, followlinks=True):
            directory_name = directory_entry[0]
            directory_members = directory_entry[2]
            for file_name in directory_members:
                match = self._re.match(file_name)
                fpath = os.path.join(directory_name, file_name)
                if not match:
                    continue

                match_components = match.groupdict()
                dataset_path = self._prepend_path + fpath
                dataset_id = self.update_dataset(dataset=datasets, match_components=match_components,
                                                 dataset_path=dataset_path)
                dataset = self.get_dataset_by_id(dataset_id, dataset=datasets)

                if not filter(dataset_id, match_components, dataset):
                    self.remove_dataset_by_id(dataset_id)
