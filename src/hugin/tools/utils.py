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

import importlib
from logging import getLogger

import keras.backend as K
import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix

log = getLogger(__name__)


def import_model_builder(model_spec):
    model_components = model_spec.split(":")
    if len(model_components) != 2:
        log.critical("Invalid model specifier")
        raise ModuleNotFoundError("Invalid model name specified")

    module_name, function = model_components
    log.info("Importing model builder using specification: %s", model_spec)
    module = importlib.import_module(module_name)
    if module is None:
        log.error("Could not find module %s", module_name)
        raise ModuleNotFoundError("Could not find module %s" % module_name)
    model_generator = getattr(module, function, None)
    if model_generator is None:
        log.critical("Could not find function: %s", function)
        raise AttributeError("Missing function %s" % function)
    log.info("Module loaded")
    model_generator_custom_objects = getattr(model_generator, 'custom_objects', {})

    return (model_generator, model_generator_custom_objects)


def convert_list_tuple(dmap):
    def to_tuple(lst):
        return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

    for k, v in dmap.items():
        dmap[k] = to_tuple(v)

    return dmap


def jaccard_coef(y_true, y_pred, smooth=1e-12):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)



def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)



def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def custom_objects(objects):
    def __wrap(f):
        setattr(f, "custom_objects", objects)
        return f

    return __wrap




def kappa_scorer(y_true, y_pred):
    '''
    Cohen's kappa: a statistic that measures inter_annotator agreement.
    :param y_true:
    :param y_pred:
    :return:
    '''
    kappa_coef = cohen_kappa_score(y_true, y_pred)
    return kappa_coef


def acc_scorer(y_true, y_pred):
    '''
    Compute overall accuracy classification score
    NOTE: This might be identical to Jaccard similarity score for binary classification!
    :param y_true:
    :param y_pred:
    :return:
    '''
    acc_score = accuracy_score(y_true, y_pred)
    return acc_score


def avg_scorer(y_true, y_pred):
    '''
    Compute average precision (AP) from predicted scores
    :param y_true:
    :param y_pred:
    :return:
    '''
    avg_score = average_precision_score(y_true, y_pred)
    return avg_score


# Compute mean_iou per class https://www.davidtvs.com/keras-custom-metrics/
def mean_iou(y_true, y_pred):
    # Wraps np_mean_iou method and uses it as a TensorFlow op.
    # Takes numpy arrays as its arguments and returns numpy arrays as
    # its outputs.
    return tf.py_func(np_mean_iou, [y_true, y_pred], tf.float32)


def np_mean_iou(y_true, y_pred, labels):
    conf = confusion_matrix(y_true, y_pred, labels)

    # Compute the IoU and mean IoU from the confusion matrix
    true_positive = np.diag(conf)
    false_positive = np.sum(conf, 0) - true_positive
    false_negative = np.sum(conf, 1) - true_positive

    # Just in case we get a division by 0, ignore/hide the error and set the value to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / (true_positive + false_positive + false_negative)
    iou[np.isnan(iou)] = 0

    return conf, iou, np.mean(iou).astype(np.float32)


def overall_accuracy(conf_matrix):
    true_positive = np.diag(conf_matrix)

    sum_true_positive = np.sum(true_positive)
    total_pixels = np.sum(conf_matrix)

    oa = sum_true_positive / total_pixels * 1.0

    return oa
