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
    parser = argparse.ArgumentParser(description='Hugin')
    parser.add_argument('--config', type=argparse.FileType('r'), required=False, default=None,
                        help='Path to config file')
    subparsers = parser.add_subparsers(help='Available commands')

    parser_trainv2 = subparsers.add_parser('train', help="Train a model")
    parser_trainv2.add_argument('--config', required=True, type=argparse.FileType('r'),
                                help="Path to configuration file")
    parser_trainv2.add_argument('--input-dir', required=False, default=None)
    parser_trainv2.set_defaults(func=train_handler)

    parser_predictv3 = subparsers.add_parser('predict', help='Run prediction')
    parser_predictv3.add_argument('--ensemble-config', required=True,
                                  type=argparse.FileType('r'),
                                  help='Path to the ensable configuration')
    parser_predictv3.add_argument('--input-dir', required=True)
    parser_predictv3.add_argument('--data-source', required=False, default='directory',
                                  help='Data source for input data. Defaults to directory')
    parser_predictv3.add_argument('--output-dir', required=False, default=None)
    parser_predictv3.add_argument('--output-text', default=None, type=argparse.FileType('w'))
    parser_predictv3.add_argument('--scoring-gti', default=None, type=str, help="Component to use for scoring")
    parser_predictv3.set_defaults(func=predict_handler)

    args = parser.parse_args()

    if args.config is not None:
        config = yaml.load(args.config, Loader=Loader)
    else:
        config = {}

    if hasattr(args, "func"):
        args.func(config, args)


if __name__ == "__main__":
    main()
