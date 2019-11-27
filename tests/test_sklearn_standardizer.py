import os
import pytest
from tempfile import TemporaryDirectory

from hugin.engine.core import SkLearnStandardizer
from hugin.engine.scene import RasterSceneTrainer
from hugin.io.loader import BinaryCategoricalConverter

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
            },
            'input_2': {
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


def _create_sklearn_standardizer_instance(generated_filesystem_loader, new_mapping, tmp_dest):
    scaler_model = SkLearnStandardizer(tmp_dest, name="skstandardizer")

    trainer = RasterSceneTrainer(name="test_raster_trainer",
                                 stride_size=256,
                                 window_size=(256, 256),
                                 model=scaler_model,
                                 mapping=new_mapping)

    dataset_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()

    trainer.train_scenes(dataset_loader)
    print(f"Saving standardizer to {tmp_dest}")
    trainer.save(tmp_dest)


def _test_sklearn_standardizer_deserialization(standardizer_path):
    import pickle

    standardizer_path = os.path.join(standardizer_path, 'skstandardizer', 'input_1.pkl')
    with open(standardizer_path, 'rb') as f:
        standardizer = pickle.loads(f.read())
        print(standardizer)


def test_sklearn_standardizer_multiple_input(generated_filesystem_loader, mapping):
    with TemporaryDirectory() as tmp_dest:
        _create_sklearn_standardizer_instance(generated_filesystem_loader, mapping, tmp_dest)

        assert sorted(os.listdir(os.path.join(tmp_dest, 'skstandardizer'))) \
               == ['input_1.pkl', 'input_1_gti.pkl', 'input_2.pkl', 'input_2_gti.pkl']

        # _test_sklearn_standardizer_deserialization(tmp_dest)


def test_sklearn_standardizer_single_input(generated_filesystem_loader, mapping):
    del mapping['inputs']['input_2']

    with TemporaryDirectory() as tmp_dest:
        _create_sklearn_standardizer_instance(generated_filesystem_loader, mapping, tmp_dest)

        assert len(os.listdir(os.path.join(tmp_dest, 'skstandardizer')))  == 1152
