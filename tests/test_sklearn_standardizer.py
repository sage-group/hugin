import pytest
from tempfile import TemporaryDirectory

from hugin.engine.core import SkLearnStandardizer
from tests.conftest import generate_filesystem_loader

def test_sklearn_standardizer(generated_filesystem_loader):
    dataset_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()

    # for scene in dataset_loader:
    #     print(scene[1].items())

    with TemporaryDirectory() as temp_dir:
        standardizer = SkLearnStandardizer(temp_dir)
        standardizer.fit_generator(dataset_loader)
