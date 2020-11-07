import logging
from urllib.parse import urlparse, parse_qs

import yaml
import fsspec
from hugin.engine.scene import RasterSceneTrainer, ArrayModel

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

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

    data_source = config.get("data_source", None)
    trainer = config["trainer"]
    training_configuration = config["configuration"]

    destination = training_configuration["model_path"]
    keys = {
        'name': trainer.model_name
    }
    destination = destination.format(**keys)
    trainer.destination = destination

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
        trainer.train(data_source)
    else:
        raise NotImplementedError("Specified trainer type is not supported")

    log.info("Training completed")
    log.info("Saving configuration to: %s", destination)
    trainer.save()