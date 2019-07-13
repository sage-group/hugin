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
from logging.config import dictConfig
from pkg_resources import resource_stream

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

internal_logging_config = yaml.load(resource_stream(__name__, "/../_data/logging-config.yaml"), Loader=Loader)
dictConfig(internal_logging_config)

import argparse
from logging import getLogger

from hugin.tools.predictv3 import predict_handlerv3
from hugin.tools.trainv2 import train_handler as train_handlerv2


def predict_handler(*args, **kw):
    from .predictv2 import predict_handler as predict_handlerv2
    return predict_handlerv2(*args, **kw)


def train_handler(*args, **kw):
    from .train import train_handler as train_handler_impl
    return train_handler_impl(*args, **kw)


log = getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Hugin')
    parser.add_argument('--config', type=argparse.FileType('r'), required=False, default=None,
                        help='Path to config file')
    subparsers = parser.add_subparsers(help='Available commands')

    parser_trainv2 = subparsers.add_parser('trainv2', help="Train a model")
    parser_trainv2.add_argument('--config', required=True, type=argparse.FileType('r'), help="Path to configuration file")
    parser_trainv2.add_argument('--input-dir', required=False, default=None)
    parser_trainv2.set_defaults(func=train_handlerv2)

    parser_train = subparsers.add_parser('train', help='Train a model')
    parser_train.add_argument('--switch-to-prefix', action='store_true', default=False,
                              help="Chdir to python sysprefix")
    parser_train.add_argument('--split', required=False, default=None)
    parser_train.add_argument('--input', required=False)
    parser_train.add_argument('--config', required=True, type=str, help="Path to configuration file")
    parser_train.add_argument('--keras-batch-size', type=int, default=None, help="Override batch size")
    parser_train.add_argument('--keras-multi-gpu', action='store_true', default=False,
                              help="Enable Keras support for multi-gpu")
    parser_train.add_argument('--keras-gpus', default=None, type=int,
                              help="Number >= 2 or list of integers, number of GPUs or list of GPU IDs on which to create model replicas")
    parser_train.add_argument('--keras-disable-cpu-merge', action='store_false', default=True, help="Disable CPU merge")
    parser_train.add_argument('--keras-enable-cpu-relocation', action='store_true', default=False,
                              help="Use CPU Relocation")
    parser_train.add_argument('--job-dir', type=str, required=False, help="Only needed for GCP ML Engine")
    parser_train.set_defaults(func=train_handler)
    parser_predictv2 = subparsers.add_parser('predict', help='Run prediction')
    parser_predictv2.add_argument('--ensemble-config', required=True,
                                  type=argparse.FileType('r'),
                                  help='Path to the ensable configuration')
    parser_predictv2.add_argument('--input-dir', required=True)
    parser_predictv2.add_argument('--data-source', required=False, default='directory',
                                  help='Data source for input data. Defaults to directory')
    parser_predictv2.add_argument('--output-dir', required=False, default=None)
    parser_predictv2.add_argument('--output-text', default=None, type=argparse.FileType('w'))
    parser_predictv2.add_argument('--scoring-gti', default=None, type=str, help="Component to use for scoring")
    parser_predictv2.set_defaults(func=predict_handler)

    parser_predictv3 = subparsers.add_parser('predictv3', help='Run prediction')
    parser_predictv3.add_argument('--ensemble-config', required=True,
                                  type=argparse.FileType('r'),
                                  help='Path to the ensable configuration')
    parser_predictv3.add_argument('--input-dir', required=True)
    parser_predictv3.add_argument('--data-source', required=False, default='directory',
                                  help='Data source for input data. Defaults to directory')
    parser_predictv3.add_argument('--output-dir', required=False, default=None)
    parser_predictv3.add_argument('--output-text', default=None, type=argparse.FileType('w'))
    parser_predictv3.add_argument('--scoring-gti', default=None, type=str, help="Component to use for scoring")
    parser_predictv3.set_defaults(func=predict_handlerv3)

    args = parser.parse_args()

    if args.config is not None:
        config = yaml.load(args.config, Loader=Loader)
    else:
        config = {}

    if hasattr(args, "func"):
        args.func(config, args)


if __name__ == "__main__":
    main()
