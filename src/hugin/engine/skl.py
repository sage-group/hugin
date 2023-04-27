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

import os
import pickle
from logging import getLogger

import tqdm

from ..engine.core import RasterModel

log = getLogger(__name__)


class SkLearnModel(RasterModel):
    def __init__(self, model=None,
                 model_path: str = None,
                 loop: bool = False,
                 **kwargs):
        """Trains or predictis using an SciKit-Model

        :param model: reference to an object compatible to the SciKit-Learn API
        :param model_path: path to an pickled version of a model
        :param loop: specifies if the model should loop forever
        :param kwargs: passed to RasterModel
        """
        RasterModel.__init__(self, **kwargs)
        self.model_path = model_path
        self.model = model
        self.loop = loop
        if self.model_path is not None and os.path.exists(self.model_path):
            log.info(f"Loading model from {self.model_path}")
            with open(self.model_path, "rb") as model_pickle:
                self.model = pickle.load(model_pickle)
            log.info("Model Loaded")
        elif self.model is None:
            log.error("No model specified")
            raise ValueError("No model specified")

        if not hasattr(self.model, "partial_fit"):
            raise ValueError(
                "Model does not implement the SciKit-Learn `partial_fit` API"
            )
        log.info(f"Using {self.model}")

    def fit_generator(self, train_data, validation_data=None, *_):
        """Incremental fitting for compliant models

        :param train_data: X values
        :param validation_data: y values
        :param _: Other arguments ignored
        :return:
        """
        log.info("Starting fitting")
        count = 0
        for batch in tqdm.tqdm(train_data):
            if not self.loop:
                if count == len(train_data):
                    break
            count += 1
            X, y = batch
            self.model.partial_fit(X, y)
        log.info("Finished fitting")

    def save(self, destination=None):
        log.info("Saving SciKit-Learn model to %s", destination)
        destination_file = os.path.join(destination, f"{self.model_name}.pickle")
        if not os.path.exists(destination):
            os.makedirs(destination)
        log.info(f"Pickling model to {destination_file}")
        with open(destination_file, "wb") as model_pickle:
            pickle.dump(self.model, model_pickle)
            log.info(f"Pickled model to {destination_file}")
