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
log = logging.getLogger(__name__)
from .loader import TileGenerator, DataGenerator, ThreadedDataGenerator, CategoricalConverter
from .dataset_loaders import FileLoader
from .dataset_loaders import FileSystemLoader
try:
    from .rio_dataset_loaders import DatasetGenerator
except ImportError:
    log.exception("DatasetGenerator could not be imported")
from .dataset_loaders import ArrayLoader
try:
    from .zarr_loader import ZarrArrayLoader
except ImportError:
    log.exception("ZarrArrayLoader could not be imported")
