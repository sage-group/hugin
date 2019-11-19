import logging
import os

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

log = logging.getLogger(__name__)


def predict_handler(args):
    ensemble_config_file = args.config
    input_dir = args.input_dir
    output_dir = args.output_dir

    ensemble_config = yaml.load(ensemble_config_file, Loader=Loader)

    data_source = ensemble_config["data_source"]
    predictor = ensemble_config["predictor"]
    saver = ensemble_config["output"]

    if data_source.input_source is None:
        data_source.set_input_source(input_dir)

    log.info("Using datasource: %s", data_source)
    log.info("Atempting to classify data in %s", data_source.input_source)

    dataset_loader, _ = data_source.get_dataset_loaders()
    log.info("classifying %d datasets", len(dataset_loader))

    if output_dir is not None:
        if not os.path.exists(output_dir):
            log.info("Creating output directory: %s", output_dir)
            os.makedirs(output_dir)
        saver.destination = output_dir

    saver.flow_prediction_from_source(dataset_loader, predictor)
