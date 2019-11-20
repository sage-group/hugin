import logging

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

log = logging.getLogger(__name__)

def train_handler(args):
    input_dir = args.input_dir
    config = yaml.load(args.config, Loader=Loader)
    data_source = config["data_source"]
    trainer = config["trainer"]
    training_configuration = config["configuration"]

    destination = training_configuration["model_path"]
    keys = {
        'name': trainer.model_name
    }
    destination = destination.format(**keys)
    trainer.destination = destination

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
    log.info("Training completed")
    log.info("Saving configuration to: %s", destination)
    trainer.save()