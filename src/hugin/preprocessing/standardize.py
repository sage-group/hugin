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

import pickle
from urllib.parse import urlparse

import fsspec
import numpy as np


class SkLearnStandardizer(object):
    def __init__(self, path, standardize_output=False, fsspec_storage_options={}):
        url_spec = urlparse(path)
        self.fs = fsspec.filesystem(url_spec.scheme, **fsspec_storage_options)
        self.path = path
        self.standardize_output = standardize_output
        with self.fs.open(path, "rb") as f:
            self.model = pickle.load(f)

    def __call__(self, input_data, output_data=None):
        if isinstance(input_data, dict):
            new_input = {}
            new_output = {}
            for k, v in input_data.items():
                old_shape = v.shape[:]
                new_input[k] = self.model.transform(v.reshape(-1, 1)).reshape(old_shape)
            if self.standardize_output:
                for k, v in input_data.items():
                    old_shape = v.shape[:]
                    new_output[k] = self.model.transform(v.reshape(-1, 1)).reshape(
                        old_shape
                    )

            return (new_input, new_output if new_output else output_data)

        elif isinstance(input_data, (np.ndarray, np.generic)) and output_data is None:
            old_shape = input_data.shape[:]
            result = self.model.transform(input_data.reshape(-1, 1)).reshape(old_shape)
            # result[input_data == 0] = np.nan
            return result
        else:
            raise NotImplementedError("Unsupported transform scenario")
