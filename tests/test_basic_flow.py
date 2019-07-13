import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest
import rasterio
import numpy as np

from hugin.engine.core import IdentityModel, AverageMerger, NullMerger
from hugin.engine.scene import RasterSceneTrainer, AvgEnsembleScenePredictor, RasterScenePredictor, \
    RasterIOSceneExporter
from hugin.io.loader import BinaryCategoricalConverter
from tests.conftest import generate_filesystem_loader


#@pytest.fixture
#def small_generated_filesystem_loader():
#    return generate_filesystem_loader(num_images=4, width=500, height=510)

@pytest.fixture
def mapping():
    mapping_conf = {
        'inputs': {
            'input_1': {
                'primary': True,
                'channels': [
                    ["RGB", 1],
                    ["RGB", 2],
                    ["RGB", 3]
                ]
            }
        },
        'target': {
            'output_1': {
                'channels': [
                    ["GTI", 1]
                ],
                'window_shape': (256, 256),
                'stride': 256,
                'preprocessing': [
                    BinaryCategoricalConverter(do_categorical=False)
                ]
            }
        }
    }
    return mapping_conf

"""
@pytest.fixture
def raster_predictors(mapping):
    identity_model = IdentityModel(name="dummy_identity_model", num_loops=3)
    raster_predictor = RasterScenePredictor(
            name="simple_raster_scene_predictor",
            model=identity_model,
            stride_size=256,
            window_size=(256, 256),
            mapping=mapping,
            #prediction_merger=AverageMerger,
            prediction_merger=NullMerger,
            post_processors=[]
    )

    return raster_predictor
"""

# @pytest.mark.skipif(not runningInCI(), reason="Skipping running locally as it might be too slow")
def test_identity_complete_flow(generated_filesystem_loader, mapping):
    _test_identity_training(generated_filesystem_loader, IdentityModel, mapping)
    new_mapping = mapping.copy()
    del new_mapping['target']
    _test_identity_prediction(generated_filesystem_loader, IdentityModel, new_mapping)
    _test_identity_prediction_avgmerger(generated_filesystem_loader, IdentityModel, new_mapping)
    _test_identity_avg_prediction(generated_filesystem_loader, IdentityModel, new_mapping)


def _test_identity_training(loader, model, mapping):
    with NamedTemporaryFile(delete=False) as named_temporary_file:
        named_tmp = named_temporary_file.name
        os.remove(named_temporary_file.name)
        identity_model = model(name="dummy_identity_model", num_loops=3)
        trainer = RasterSceneTrainer(name="test_raster_trainer",
                                     stride_size=256,
                                     window_size=(256, 256),
                                     model=identity_model,
                                     mapping=mapping,
                                     destination=named_tmp)

        dataset_loader, validation_loader = loader.get_dataset_loaders()
        loop_dataset_loader_old = dataset_loader.loop
        loop_validation_loader_old = validation_loader.loop

        try:
            dataset_loader.loop = True
            validation_loader.loop = True
            print("Training on %d datasets" % len(dataset_loader))
            print("Using %d datasets for validation" % len(validation_loader))

            trainer.train_scenes(dataset_loader, validation_scenes=validation_loader)
            trainer.save()

            assert os.path.exists(named_tmp)
            assert os.path.getsize(named_tmp) > 0
        finally:
            dataset_loader.loop = loop_dataset_loader_old
            validation_loader.loop = loop_validation_loader_old


def _get_input_and_prediction_data(loader, dest_tmpdir):
    for scene in loader:
        input_file = scene[1]['RGB']
        input_data = input_file.read()
        prediction = os.path.join(dest_tmpdir,
                                  os.path.split(input_file.name)[-1].replace('_RGB', ''))

        with rasterio.open(prediction) as prediction_file:
            prediction_data = prediction_file.read()

        yield (input_data, prediction_data)


def _test_identity_prediction(loader, model, mapping):
        dataset_loader, validation_loader = loader.get_dataset_loaders()
        identity_model = model(name="dummy_identity_model", num_loops=3)

        raster_predictor = RasterScenePredictor(
            name="simple_raster_scene_predictor",
            model=identity_model,
            stride_size=256,
            window_size=(256, 256),
            mapping=mapping,
            prediction_merger=NullMerger,
            post_processors=[]
        )

        with TemporaryDirectory() as dest_tmpdir:
            raster_saver = RasterIOSceneExporter(destination=dest_tmpdir,
                                                 srs_source_component="RGB",
                                                 filename_pattern="{scene_id}.tiff",
                                                 rasterio_creation_options={
                                                     'blockxsize': 256,
                                                     'blockysize': 256
                                                 }
            )

            dataset_loader.reset()
            raster_saver.flow_prediction_from_source(dataset_loader, raster_predictor)

            dataset_loader.reset()
            for input_data, prediction_data in _get_input_and_prediction_data(dataset_loader, dest_tmpdir):
                np.testing.assert_array_equal(input_data, prediction_data)


def _test_identity_prediction_avgmerger(loader, model, mapping):
        dataset_loader, validation_loader = loader.get_dataset_loaders()
        identity_model = model(name="dummy_identity_model", num_loops=3)

        raster_predictor = RasterScenePredictor(
            name="simple_raster_scene_predictor",
            model=identity_model,
            stride_size=256,
            window_size=(256, 256),
            mapping=mapping,
            prediction_merger=AverageMerger,
            post_processors=[]
        )

        with TemporaryDirectory() as dest_tmpdir:
            raster_saver = RasterIOSceneExporter(destination=dest_tmpdir,
                                                 srs_source_component="RGB",
                                                 filename_pattern="{scene_id}.tiff",
                                                 rasterio_creation_options={
                                                     'blockxsize': 256,
                                                     'blockysize': 256
                                                 }
            )

            dataset_loader.reset()
            raster_saver.flow_prediction_from_source(dataset_loader, raster_predictor)

            dataset_loader.reset()
            for input_data, prediction_data in _get_input_and_prediction_data(dataset_loader, dest_tmpdir):
                #np.allclose(input_data, prediction_data, 1e-05, 1e-06)
                np.allclose(input_data, prediction_data)


def _test_identity_avg_prediction(loader, model, mapping):
        dataset_loader, validation_loader = loader.get_dataset_loaders()
        identity_model = model(name="dummy_identity_model", num_loops=3)

        raster_predictor = RasterScenePredictor(
            name="simple_raster_scene_predictor",
            model=identity_model,
            stride_size=256,
            window_size=(256, 256),
            mapping=mapping,
            #prediction_merger=AverageMerger,
            prediction_merger=NullMerger,
            post_processors=[],
        )

        avg_predictor = AvgEnsembleScenePredictor(
            name="simple_avg_ensable",
            predictors=[
                {
                    'predictor': raster_predictor,
                    'weight': 1
                },
                {
                    'predictor': raster_predictor,
                    'weight': 1
                }
            ])

        with TemporaryDirectory() as dest_tmpdir:
            raster_saver = RasterIOSceneExporter(destination=dest_tmpdir,
                                                 srs_source_component="RGB",
                                                 filename_pattern="{scene_id}.tiff",
                                                 rasterio_creation_options={
                                                     'blockxsize': 256,
                                                     'blockysize': 256
                                                 }
            )

            dataset_loader.reset()
            raster_saver.flow_prediction_from_source(dataset_loader, avg_predictor)

            dataset_loader.reset()
            for input_data, prediction_data in _get_input_and_prediction_data(dataset_loader, dest_tmpdir):
                np.testing.assert_array_equal((input_data + input_data) / 2.0,
                                              prediction_data)