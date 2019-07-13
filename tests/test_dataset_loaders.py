import os
import glob
import pytest

from rasterio.io import DatasetReader
from hugin.io import FileLoader, FileSystemLoader, DataGenerator
from tempfile import NamedTemporaryFile

from tests import runningInCI

basedir = os.path.join(os.path.dirname(__file__), "data", "scanner_examples")

class TestLoaders(object):
    def setup(self):
        self.pattern = basedir + os.sep + "*.txt"
        self.files = glob.glob(self.pattern)
        self.tempf = NamedTemporaryFile(mode="w+", delete=True)
        self.base_kwargs = {
            'data_pattern': r"(?P<name>[0-9A-Za-z]+)_(?P<SECOND>[A-Za-z0-9]+)_(?P<THIRD>[A-Za-z0-9]+)\.txt$",
            'type_format': "{THIRD}",
            'id_format': "{name}-{SECOND}",
            'input_source': self.tempf.name
        }

        for fle in self.files:
            self.tempf.write("%s\n" % fle)
        self.tempf.flush()

    def teardown(self):
        self.tempf.close()

    def test_detected_from_input_file(self):
        kwargs = self.base_kwargs.copy()
        kwargs['input_source'] = self.tempf.name
        fs1 = FileLoader(**kwargs)
        assert len(fs1) == 3

    def test_detected_from_input_descriptor(self):
        self.tempf.seek(0)
        kwargs = self.base_kwargs.copy()
        kwargs['input_source'] = self.tempf
        fs2 = FileLoader(**kwargs)
        assert len(fs2) == 3

    def test_training_validation_split_working(self):
        kwargs = self.base_kwargs.copy()
        kwargs['validation_percent'] = 0.4
        fs3 = FileLoader(**kwargs)
        assert len(fs3) == 3
        assert len(fs3.get_training_datasets()) == 2
        assert len(fs3.get_validation_datasets()) == 1

    def test_filter(self):
        kwargs4 = self.base_kwargs.copy()
        kwargs4['filter'] = lambda dataset_id, match_components, dataset: not dataset_id.startswith("C-")
        fs4 = FileLoader(**kwargs4)
        assert (len(fs4)) == 2

    def test_missing_pattern_raises_error(self):
        kwargs5 = self.base_kwargs.copy()
        del kwargs5['data_pattern']
        with pytest.raises(ValueError) as excinfo:
            FileLoader(**kwargs5)

    def test_support_for_custom_attributes(self):
        kwargs6 = self.base_kwargs.copy()
        kwargs6['custom_attributes'] = {
            'foo': lambda **kw: int(kw['THIRD']) * 10
        }
        kwargs6['id_format'] = kwargs6['id_format']+"-"+"{foo}"

        fs6 = FileLoader(**kwargs6)
        assert 'C-XYZ-10' in fs6.get_full_datasets()

    def test_pattern_partially_matching_input(self):
        kwargs7 = self.base_kwargs.copy()
        kwargs7['data_pattern'] = r'(?P<name>[0-9A-Ba-z]+)_(?P<SECOND>[A-Za-z0-9]+)_(?P<THIRD>[A-Za-z0-9]+)\.txt$'
        fs7 = FileLoader(**kwargs7)
        assert len(fs7) == 2

    def test_training_validation_loader_retrieval(self):

        kwargs8 = self.base_kwargs.copy()
        kwargs8['validation_percent'] = 0.4
        fs8 = FileLoader(**kwargs8)
        training_loader, validation_loader = fs8.get_dataset_loaders()

        assert len(training_loader) == 2
        assert len(validation_loader) == 1

class TestFileSystemLoader(object):
    def setup(self):
        self.base_kwargs = {
            'data_pattern': r"(?P<name>[0-9A-Za-z]+)_(?P<SECOND>[A-Za-z0-9]+)_(?P<THIRD>[A-Za-z0-9]+)\.txt$",
            'type_format': "{THIRD}",
            'id_format': "{name}-{SECOND}",
            'input_source': basedir
        }

    def test_detected_from_input_directory(self):
        kwargs = self.base_kwargs.copy()
        ds = FileSystemLoader(**kwargs)
        assert len(ds) == 3

class TestDatasetGenerator(object):
    def test_loader_loops(self, generated_filesystem_loader):
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
        assert len(training_loader) == 8
        assert len(validation_loader) == 2

        def _loop_data():
            for i in range(0, len(training_loader) + 10):
                next(training_loader)

            for i in range(0, len(validation_loader) + 1):
                next(validation_loader)

        training_loader.loop = False
        validation_loader.loop = False

        assert not training_loader.loop

        with pytest.raises(StopIteration) as e_info:
            _loop_data()

        training_loader.loop = True
        validation_loader.loop = True

        assert training_loader  # Should return True

        _loop_data()

    def test_loader_datasets(self, generated_filesystem_loader):
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()

        dataset_name, dataset_parts = next(training_loader)
        assert isinstance(dataset_name, str)
        assert 'RGB' in dataset_parts

        rgb_dset = dataset_parts['RGB']
        isinstance(rgb_dset, DatasetReader)

        assert rgb_dset.count == 3


class TestDataGenerator(object):
    @pytest.mark.skipif(not runningInCI(), reason="Skipping running locally as it might be too slow")
    def test_number_of_tiles(self, generated_filesystem_loader):
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
        training_loader.loop = True
        validation_loader.loop = True


        train_data = DataGenerator(training_loader,
                                   batch_size=None,
                                   input_mapping={
                                       'input_1': {
                                           'primary': True,
                                           'window_shape': (512, 512),
                                           'stride': 512,
                                           'channels': [
                                               [ "RGB", 1 ],
                                               [ "RGB", 2 ],
                                               [ "RGB", 3 ],
                                           ],
                                       }
                                   },
                                   output_mapping={})
        validation_data = DataGenerator(validation_loader,
                                   batch_size=None,
                                   input_mapping={
                                       'input_1': {
                                           'primary': True,
                                           'window_shape': (512, 512),
                                           'stride': 512,
                                           'channels': [
                                               ["RGB", 1],
                                               ["RGB", 2],
                                               ["RGB", 3],
                                           ],
                                       }
                                   },
                                   output_mapping={
                                       'output_1': {
                                           'channels': [
                                               ["RGB"]
                                           ]
                                       }
                                   })

        assert len(train_data) == 160
        assert len(validation_data) == 40
        for i in range(len(train_data)):
            tile =  next(train_data)


    @pytest.mark.skipif(not runningInCI(), reason="Skipping running locally as it might be too slow")
    def test_number_of_tiles_clasic(self, generated_filesystem_loader):
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
        training_loader.loop = True
        validation_loader.loop = True


        train_data = DataGenerator(training_loader,
                                   batch_size=None,
                                   default_window_size=(256, 256),
                                   input_mapping=[
                                       ("RGB", 1),
                                       ("RGB", 2),
                                       ("RGB", 3)
                                   ],
                                   output_mapping={})


        assert len(train_data) == 576
        for i in range(len(train_data)):
            tile = next(train_data)


    # def test_image_reassembly(self, generated_filesystem_loader):
    #     training_loader, validation_loader = generated_filesystem_loader.get_dataset_loaders()
    #     training_loader.loop = True
    #     validation_loader.loop = True
    #     training_loader._datasets = training_loader._datasets[:1]
    #
    #     train_data = DataGenerator(training_loader,
    #                                batch_size=None,
    #                                default_window_size=(256, 256),
    #                                input_mapping=[
    #                                    ("RGB", 1),
    #                                    ("RGB", 2),
    #                                    ("RGB", 3)
    #                                ],
    #                                output_mapping={})
    #
    #     img_path = training_loader._datasets[0][1]['RGB']
    #     import rasterio
    #     dset = rasterio.open(img_path)
    #     img = dset.read()
    #     assert len(train_data) == 72


