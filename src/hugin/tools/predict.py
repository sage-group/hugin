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

import logging
import os

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

log = logging.getLogger(__name__)


def predict_handler(args):
    from ..engine.scene import ArrayModelPredictor, RasterScenePredictor

    ensemble_config_file = args.config
    input_dir = args.input_dir
    output_dir = args.output_dir

    ensemble_config = yaml.load(ensemble_config_file, Loader=Loader)

    data_source = ensemble_config["data_source"]
    predictor = ensemble_config["predictor"]
    saver = ensemble_config["output"]
    experiment_configuration = ensemble_config.get("configuration", {})
    experiment_configuration["args"] = args

    workspace_directory = experiment_configuration.get("workspace", None)
    if workspace_directory is not None:
        workspace_directory = workspace_directory.format(**experiment_configuration)

    if workspace_directory and predictor.base_directory is None:
        predictor.base_directory = workspace_directory

    if isinstance(predictor, RasterScenePredictor):
        if data_source.input_source is None:
            data_source.set_input_source(input_dir)

        log.info("Using datasource: %s", data_source)
        log.info("Attempting to classify data in %s", data_source.input_source)

        dataset_loader, _ = data_source.get_dataset_loaders()
        log.info("classifying %d datasets", len(dataset_loader))

        if output_dir is not None:
            if not os.path.exists(output_dir):
                log.info("Creating output directory: %s", output_dir)
                os.makedirs(output_dir)
            saver.base_directory = output_dir
        saver.flow_prediction_from_source(dataset_loader, predictor)
    elif isinstance(predictor, ArrayModelPredictor):
        log.info("Using array source: %s", data_source)
        saver.flow_prediction_from_array_loader(data_source, predictor)
