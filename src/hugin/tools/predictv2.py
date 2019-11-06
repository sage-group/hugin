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

import pickle
from logging import getLogger

import h5py
import numpy as np
import os
import rasterio
import re
import skimage.io as io
import yaml
from geojson import Feature, FeatureCollection, dump as jsdump
from keras.models import load_model
from shapely.geometry import Polygon
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_similarity_score, \
    zero_one_loss, hamming_loss, average_precision_score, cohen_kappa_score, confusion_matrix

from ..io import DataGenerator
from ..io.loader import adapt_shape_and_stride
from ..tools.predict import get_probabilities_from_tiles, categorical_prediction, to_polygons
from ..tools.utils import import_model_builder, np_mean_iou, overall_accuracy

log = getLogger(__name__)


def stack_predictions(model_spec):
    log.info("Stacking predictions")
    assert (len(model_spec) < 2)
    stack = np.zeros(model_spec[0][2].shape, model_spec[0][2].dtype)
    for model_name, model_config, scene_data in model_spec:
        reconstructed = categorical_prediction(scene_data)
        stack = np.where(stack != 0, stack, reconstructed)
    log.info("Finished stacking")
    return stack


def ensemble_avg(model_spec):
    log.info("Averaging predictions")
    sum_array = np.zeros(model_spec[0][2].shape, model_spec[0][2].dtype)
    total_weight = 0
    for model_name, model_config, scene_data in model_spec:
        weight = model_config.get("weight", 1)
        total_weight += weight
        sum_array += weight * scene_data[()]
    result = sum_array / total_weight
    log.info("Finished averaging")
    return result


def run_final_postprocessings(postprocessings, final_data, probability_data):
    for proc in postprocessings:
        log.debug("Running %s", proc)
        final_data, probability_data = proc(final_data, probability_data)
    return final_data, probability_data

def predict_handler(config, args):
    ensemble_config_file = args.ensemble_config
    ensemble_config = yaml.load(ensemble_config_file)
    data_source = ensemble_config["data_source"]
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_prediction_h5 = os.path.join(output_dir, "predictions.h5")
    output_text = args.output_text
    models_config = ensemble_config["models"]
    postprocessings = ensemble_config.get("postprocessing", [])
    params = ensemble_config.get("params", {})
    prediction_formater = ensemble_config.get("prediction_formater", categorical_prediction)

    if data_source.input_source is None:
        data_source.set_input_source(args.input_dir)

    log.info("Using datasource: %s", data_source)
    log.info("Atempting to classify data in %s", data_source.input_source)

    dataset_loader, _ = data_source.get_dataset_loaders()

    log.info("classifying %d datasets", len(dataset_loader))

    if output_dir is not None:
        if not os.path.exists(output_dir):
            log.info("Creating output directory: %s", output_dir)
            os.makedirs(output_dir)

    log.info("Predictions will be stored in: %s", output_prediction_h5)
    h5 = h5py.File(output_prediction_h5, 'w')

    for model_name, model_config in models_config.items():
        model_type = model_config.get("type", "keras")
        model_match_id_pattern = model_config.get('match_id', '.*')
        model_match_id = re.compile(model_match_id_pattern)
        model_match_id = re.compile(model_match_id_pattern)
        window_size = model_config["window_size"]
        stride_size = model_config["stride_size"]
        swap_axes = model_config["swap_axes"]
        batch_size = model_config["batch_size"]
        tile_merge_strategy = model_config.get("tile_merge_strategy", "average")
        mapping = model_config["mapping"]
        ensemble_options = ensemble_config["ensemble"]
        ensemble_method = ensemble_options.get("method", "average")
        ensemble_handlers = {
            'average': ensemble_avg
        }
        if ensemble_method not in ensemble_handlers:
            raise KeyError("Unknown method")
        ensemble_handler = ensemble_handlers[ensemble_method]
        predicter = None
        model = None

        if model_type == "sklearn":
            model_path = model_config["path"]
            sklearn_model = pickle.load(open(model_path, "rb"))

            class __model_wrapper(object):
                def __init__(self, model):
                    self._model = model

                def predict(self, data, batch_size):
                    num_sample, width, height, num_chan = data.shape
                    data = data.reshape((num_sample * width * height, num_chan))
                    result = self._model.predict_proba(data)
                    result = result.reshape((num_sample, width, height, 2))
                    return result

            model = __model_wrapper(sklearn_model)
        elif model_type == "keras":
            model_path = model_config["path"]
            model_builder = model_config.get("builder", None)
            custom_objects = {}
            if model_builder:
                _, model_builder_custom_options = import_model_builder(model_builder)
                custom_objects.update(model_builder_custom_options)
            log.info("Loading keras model from %s", model_path)
            model = load_model(model_path, custom_objects=custom_objects)
            log.info("Finished loading")
            # model = None
        else:
            raise NotImplementedError("Unsupported model type: %s" % model_type)
        for scene in dataset_loader:
            scene_id, scene_data = scene

            if model_match_id.match(scene_id) is None:
                continue
            log.info("Classifying %s using %s", scene_id, model_name)
            input_mapping = mapping["inputs"]
            output_mapping = mapping.get("target", {})

            tile_loader = data_source.get_dataset_loader(scene)

            tile_loader.reset()
            data_generator = DataGenerator(tile_loader,
                                           batch_size=batch_size,
                                           input_mapping=input_mapping,
                                           output_mapping=None,
                                           swap_axes=swap_axes,
                                           loop=False,
                                           default_window_size=window_size,
                                           default_stride_size=stride_size)

            if len(output_mapping) == 1:
                rio_raster = scene_data[output_mapping[0][0]]
                output_window_shape, output_stride_size = adapt_shape_and_stride(rio_raster,
                                                                                 data_generator.primary_scene,
                                                                                 window_size, stride_size)
            else:
                output_window_shape = model_config.get('output_window_size', model_config.get('window_size'))
                output_stride_size = model_config.get('output_stride_size', model_config.get('stride_size'))

            output_shape = ensemble_config.get('output_shape', None)
            if output_shape:
                output_width, output_height = output_shape
            else:
                output_width, output_height = data_generator.primary_scene.shape

            image_probs = get_probabilities_from_tiles(
                model,
                data_generator,
                output_width,
                output_height,
                output_window_shape[0],
                output_window_shape[1],
                output_stride_size,
                batch_size,
                merge_strategy=tile_merge_strategy
            )
            dataset_name = "%s/%s" % (model_name, scene_id)
            log.info("Saving predictions to dataset: %s", dataset_name)
            h5[dataset_name] = image_probs

    models = list(models_config.keys())  # ?
    if output_text:
        log.info("Saving TXT output to: %s", output_text.name)

    annotation_idx = 0
    prediction_count = 0
    global_score_f1 = 0
    global_score_accuracy = 0
    global_score_precision = 0
    global_score_recall = 0
    global_score_roc_auc = 0
    global_score_jaccard_similarity = 0
    global_score_zero_one = 0
    global_score_hamming = 0
    global_score_average = 0
    global_score_kappa = 0
    global_score_iou = 0
    global_score_mean_iou = 0
    global_conf_matrix = 0
    global_score_overall_accuracy = 0

    dataset_loader.reset()
    for scene in dataset_loader:
        scene_id, scene_data = scene
        model_spec = [
            (model_name, models_config[model_name], h5["%s/%s" % (model_name, scene_id)]) for model_name in models
        ]
        result = ensemble_handler(model_spec)
        log.info("Generating prediction representation")
        reconstructed = prediction_formater(result)
        log.info("Finished generating representation")
        if postprocessings:
            reconstructed, result = run_final_postprocessings(postprocessings, reconstructed, result)
        if output_dir is not None:
            if 'RGB' in scene_data:  # ToDo: do we need special consideration for RGB ?!
                _source = scene_data['RGB']
            else:
                _source = list(scene_data.items())[0][1]
            _source_profile = _source.profile

            crs = _source_profile.get('crs', None)
            if crs is None:
                destination_file = os.path.join(output_dir, scene_id + ".png")
            else:
                destination_file = os.path.join(output_dir, scene_id + ".tif")

            log.info("Saving output tile to %s", destination_file)

            num_out_channels = reconstructed.shape[-1]
            if crs is None:
                io.imsave(destination_file, reconstructed.astype(rasterio.uint8))
            else:
                _source_profile.update(dtype=reconstructed.dtype, count=num_out_channels, compress='lzw', nodata=0)

                with rasterio.open(destination_file, 'w', **_source_profile) as dst:
                    for idx in range(0, num_out_channels):
                        dst.write(reconstructed[:, :, idx], idx + 1)

        if args.scoring_gti and args.scoring_gti not in scene_data:
            log.error("Components %s not in available components!", args.scoring_gti)

        if args.scoring_gti and args.scoring_gti in scene_data:  # We can compute the score
            prediction_count += 1
            bin_gti = scene_data[args.scoring_gti].read(1)
            format_converter = params.get("format_converter", None)
            if format_converter is not None:
                bin_gti = format_converter(bin_gti)
            if params.get("type", "binary") == "binary":
                bin_gti = bin_gti > 0
            bin_gti = bin_gti.reshape(-1)
            flat_reconstructed = reconstructed.reshape(-1)

            log.info("Scoring %s", scene_id)

            metric_params = params.get("metrics", {})
            # get all metric options
            metric_options = metric_params.get("options", {})
            metric_usage = metric_params.get("usage", {})
            if 'average' in metric_options and metric_options['average'] == 'None':
                metric_options['average'] = None

            # get metric options to metric parameters mapping
            def get_metric_option(metric_name, metric_options, metric_usage):
                metric_params = metric_usage.get(metric_name, [])
                metric_args = {}
                for m in metric_params:
                    metric_args[m] = metric_options.get(m, None)
                return metric_args

            f1_score_params = get_metric_option("f1_score", metric_options, metric_usage)
            score_f1 = f1_score(bin_gti, flat_reconstructed, **f1_score_params)

            score_accuracy = accuracy_score(bin_gti, flat_reconstructed)

            precision_score_params = get_metric_option("precision_score", metric_options, metric_usage)
            score_precision = precision_score(bin_gti, flat_reconstructed, **precision_score_params)

            recall_score_params = get_metric_option("recall_score", metric_options, metric_usage)
            score_recall = recall_score(bin_gti, flat_reconstructed, **recall_score_params)
            # score_roc_auc = roc_auc_score(bin_gti, flat_reconstructed)
            score_jaccard_similarity = jaccard_similarity_score(bin_gti, flat_reconstructed)
            score_zero_one = zero_one_loss(bin_gti, flat_reconstructed)

            hamming_loss_params = get_metric_option("hamming_loss", metric_options, metric_usage)
            score_hamming = hamming_loss(bin_gti, flat_reconstructed, **hamming_loss_params)

            score_average = 0
            if params.get("type", "binary") == "binary":
                average_precision_score_params = get_metric_option("average_precision_score", metric_options,
                                                                   metric_usage)
                score_average = average_precision_score(bin_gti, flat_reconstructed, **average_precision_score_params)

            cohen_kappa_score_params = get_metric_option("cohen_kappa_score", metric_options, metric_usage)
            score_kappa = cohen_kappa_score(bin_gti, flat_reconstructed, **cohen_kappa_score_params)

            mean_iou_params = get_metric_option("np_mean_iou", metric_options, metric_usage)

            if params.get("type", "binary") == "binary":
                conf_matrix = confusion_matrix(bin_gti, flat_reconstructed)
                score_iou = 0
                score_mean_iou = 0
            else:
                conf_matrix, score_iou, score_mean_iou = np_mean_iou(bin_gti, flat_reconstructed, **mean_iou_params)

            score_overall_accuracy = overall_accuracy(conf_matrix)

            global_score_f1 += score_f1
            global_score_accuracy += score_accuracy
            global_score_precision += score_precision
            global_score_recall += score_recall
            # global_score_roc_auc += score_roc_auc
            global_score_jaccard_similarity += score_jaccard_similarity
            global_score_zero_one += score_zero_one
            global_score_hamming += score_hamming

            global_score_average += score_average
            global_score_kappa += score_kappa
            global_score_iou += score_iou
            global_score_mean_iou += score_mean_iou
            global_score_overall_accuracy += score_overall_accuracy

            log.info("{} Scores: F-1: {}, Accuracy: {}, Precision: {}, Recall: {}, "
                     "Jaccard Similarity: {}, Zero One Loss: {}, Hamming Loss: {}, Average Precision Score: {}, Kappa Score: {}, IoU Score: {}, Mean IoU: {}, Overall accuracy: {}".format(
                scene_id, score_f1,
                score_accuracy, score_precision,
                score_recall, score_jaccard_similarity,
                score_zero_one, score_hamming, score_average, score_kappa, score_iou, score_mean_iou,
                score_overall_accuracy))

        if output_text:
            height, width, _ = result.shape
            dfile = output_text.name + ".geojson"
            features = []
            out_scene_id = scene_id

            polys = list(to_polygons(reconstructed))
            idx = 0
            for poly in polys:
                idx += 1
                _source
                gjs = [_source.transform * i for i in poly[0]]
                crs = {
                    "type": "name",
                    "properties": {
                        "name": _source.crs["init"]
                    }
                }
                if len(gjs) < 3:
                    log.warning("Not enough points for assembling polygon")
                    continue
                features.append(Feature(geometry=Polygon(gjs)))
                pstr = "POLYGON (("
                crds = []
                for pairs in poly[0]:
                    crds.append("%.3f %.3f 0" % pairs)
                crds.append(crds[0])
                pstr += ",".join(crds)
                pstr += "))"
                output_text.write("%s,%d" % (out_scene_id, idx))
                output_text.write(',"%s",%d' % (pstr, 1))
                output_text.write("\n")
                output_text.flush()
            feature_collection = FeatureCollection(features, crs=crs)
            jsdump(feature_collection, open(dfile, "w"))

    if prediction_count > 0:
        global_score_f1 = global_score_f1 / prediction_count
        global_score_accuracy = global_score_accuracy / prediction_count
        global_score_precision = global_score_precision / prediction_count
        global_score_recall = global_score_recall / prediction_count
        global_score_roc_auc = global_score_roc_auc / prediction_count
        global_score_jaccard_similarity = global_score_jaccard_similarity / prediction_count
        global_score_zero_one = global_score_zero_one / prediction_count
        global_score_hamming = global_score_hamming / prediction_count
        global_score_average = global_score_average / prediction_count
        global_score_kappa = global_score_kappa / prediction_count
        global_score_iou = global_score_iou / prediction_count
        global_score_mean_iou = global_score_mean_iou / prediction_count
        global_score_overall_accuracy = global_score_overall_accuracy / prediction_count

    else:
        global_score_f1, global_score_accuracy, global_score_precision, global_score_recall, global_score_roc_auc, global_score_jaccard_similarity, global_score_zero_one, global_score_hamming, global_score_average, global_score_kappa, global_score_iou, global_score_mean_iou, global_score_overall_accuracy = (
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    log.info("Average Global Score => F-1: {}, Accuracy: {}, Precision: {}, Recall: {}, "
             "Jaccard Similarity: {}, Zero One Loss: {}, Hamming Loss: {}, Average Precision Score: {}, Kappa Score: {}, IoU Score: {}, Mean IoU: {}, Overall accuracy: {} ".format(
        global_score_f1, global_score_accuracy,
        global_score_precision, global_score_recall,
        global_score_jaccard_similarity,
        global_score_zero_one, global_score_hamming, global_score_average, global_score_kappa, global_score_iou,
        global_score_mean_iou, global_score_overall_accuracy))

    log.info("Finished")
