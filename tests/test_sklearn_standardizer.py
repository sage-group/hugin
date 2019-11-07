import pytest
from tempfile import TemporaryFile

from hugin.engine.core import SkLearnStandardizer
from hugin.engine.scene import RasterSceneTrainer, RasterScenePredictor, RasterIOSceneExporter, MultipleFormatExporter
from hugin.io.loader import BinaryCategoricalConverter
from tests.conftest import generate_filesystem_loader

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
                    ["GTI", 1],
                ],
                'preprocessing': [
                    BinaryCategoricalConverter(do_categorical=False)
                ]
            }
        }
    }
    return mapping_conf

def test_sklearn_standardizer(generated_filesystem_loader, mapping):
    dataset_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()

    with TemporaryFile() as tmp_dest:
        scaler_model = SkLearnStandardizer(tmp_dest)

        raster_predictor = RasterScenePredictor(
            name="simple_raster_scene_predictor",
            model=scaler_model,
            stride_size=256,
            window_size=(256, 256),
            mapping=mapping,
        )

        exporter = MultipleFormatExporter(destination=tmp_dest)
        exporter.flow_prediction_from_source(dataset_loader, raster_predictor)
        #raster_predictor.predict_scenes_proba(dataset_loader)