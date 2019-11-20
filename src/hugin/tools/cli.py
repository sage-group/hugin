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
import argparse
from logging import getLogger
from logging.config import dictConfig

import yaml
from pkg_resources import resource_stream

from hugin.tools.predict import predict_handler
from hugin.tools.train import train_handler

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

internal_logging_config = yaml.load(resource_stream(__name__, "/../_data/logging-config.yaml"), Loader=Loader)
dictConfig(internal_logging_config)

log = getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='HuginEO -- Machine Learning for Earth Observation tool')
    subparsers = parser.add_subparsers(help='Available commands')

    train_parser = subparsers.add_parser('train', help="Train a model")
    train_parser.add_argument('--config', type=argparse.FileType('r'), required=False, default=None,
                              help='Path to config file')
    train_parser.add_argument('--input-dir', required=False, default=None, help=argparse.SUPPRESS)
    train_parser.set_defaults(func=train_handler)

    predict_parser = subparsers.add_parser('predict', help='Run prediction')
    predict_parser.add_argument('--config', type=argparse.FileType('r'), required=False, default=None,
                                help='Path to config file')
    predict_parser.add_argument('--input-dir', required=True, help=argparse.SUPPRESS)
    predict_parser.add_argument('--output-dir', required=False, default=None)
    predict_parser.set_defaults(func=predict_handler)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
