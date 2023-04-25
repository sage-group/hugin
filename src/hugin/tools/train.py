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
from urllib.parse import parse_qs, urlparse

import fsspec
import yaml

from hugin.engine.scene import ArrayModel, RasterSceneTrainer

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

try:
    from setproctitle import setproctitle
except ImportError:

    def setproctitle(*args, **kwargs):
        pass


log = logging.getLogger(__name__)


def train_handler(args):
    input_dir = args.input_dir
    up = urlparse(args.config)
    storage_options = {}
    if not up.scheme:
        config_file = open(args.config, "r")
    else:
        source = f"{up.scheme}://{up.netloc}{up.path}"
        for k, v in parse_qs(up.query, keep_blank_values=True).items():
            value = v[0] if v[0] else None
            storage_options[k] = value
        config_file = fsspec.open(source, mode="rt", **storage_options)

    with config_file as f:
        config = yaml.load(f, Loader=Loader)

    experiment_configuration = config.get("configuration", {})
    experiment_configuration["args"] = args

    data_source = config.get("data_source", None)
    trainer = config["trainer"]

    workspace_directory = experiment_configuration.get("workspace", None)
    experiment_name = experiment_configuration.get("name", None)

    if workspace_directory is not None:
        workspace_directory = workspace_directory.format(**experiment_configuration)
    else:
        raise ValueError("No workspace directory specified")
    if not os.path.exists(workspace_directory):
        os.makedirs(workspace_directory)

    title = f"hugin train ({experiment_name})"
    setproctitle(title)
    log.info(f"workspace directory: {workspace_directory}")
    if workspace_directory and trainer.base_directory is None:
        trainer.base_directory = workspace_directory
        trainer.destination = workspace_directory

    if isinstance(trainer, RasterSceneTrainer):
        if input_dir is not None:
            data_source.input_source = input_dir

        log.info("Using datasource: %s", data_source)
        log.info("Attempting to train with data in %s", data_source.input_source)

        dataset_loader, validation_loader = data_source.get_dataset_loaders()
        log.info("Training on %d datasets", len(dataset_loader))
        log.info("Using %d datasets for validation", len(validation_loader))

        dataset_loader.loop = True
        validation_loader.loop = True
        trainer.train_scenes(dataset_loader, validation_scenes=validation_loader)

    elif isinstance(trainer, ArrayModel):
        log.info("Training using an array model")
        import dask

        with dask.config.set(scheduler="threads", num_workers=8):
            trainer.train(data_source)
    else:
        raise NotImplementedError("Specified trainer type is not supported")

    log.info("Training completed")
    log.info("Saving configuration model")
    trainer.save()
