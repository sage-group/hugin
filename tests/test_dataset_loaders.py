import os

import glob

import random
import pytest
import numpy as np
from PIL import Image,ImageDraw
from rasterio.io import DatasetReader

from hugin.io import FileLoader, FileSystemLoader, DataGenerator
import rasterio
from rasterio.transform import from_origin

from tempfile import TemporaryDirectory, NamedTemporaryFile


basedir = os.path.join(os.path.dirname(__file__), "data", "scanner_examples")

@pytest.fixture
def generated_filesystem_loader():
    width = 2131
    height = 1979
    size = 35
    tempdir = TemporaryDirectory("-hugin")
    random.seed(42)

    match_color = "red"

    NUM_IMAGES = 10

    for imgidx in range(0, NUM_IMAGES):
        data = np.zeros((height, width, 3), dtype=np.uint8)
        data_mask = np.zeros((height, width), dtype=np.uint8)
        im = Image.fromarray(data)
        mask = Image.fromarray(data_mask, mode="L")

        draw = ImageDraw.Draw(im)
        mdraw = ImageDraw.Draw(mask)
        colors = ["red", "blue", "green", "white"]
        num_rect = 5
        num_circle = 4

        for i in range(0, num_rect):
            startx = random.randint(0, width-50)
            starty = random.randint(0, height-50)
            endx = random.randint(startx+10, startx+random.randint(startx+10, startx+size))
            endy = random.randint(starty+10, starty+random.randint(starty+10, starty+size))
            color = random.choice(colors)
            draw.rectangle([(startx, starty), (endx, endy)], fill=color)
            if match_color == color:
                mdraw.rectangle([(startx, starty), (endx, endy)], fill=color)

        for i in range(0, num_circle):
            startx = random.randint(0, width-50)
            starty = random.randint(0, height-50)
            endx = random.randint(startx+20, startx+random.randint(startx+10, startx+size))
            endy = random.randint(starty+20, starty+random.randint(starty+10, starty+size))
            color = random.choice(colors[:-1])
            draw.ellipse([(startx, starty), (endx, endy)], fill=color)
            if match_color == color:
                mdraw.ellipse([(startx, starty), (endx, endy)], fill=color)

        fname = os.path.join(tempdir.name, "TEST_RGB_{}.tiff".format(imgidx+1))
        mname = os.path.join(tempdir.name, "TEST_RGB_{}.tiff".format(imgidx+1))

        res = 0.5
        transform = from_origin(21.2086, 45.7488, res, res)
        rgb_image = rasterio.open(
            fname,
            'w',
            height=data.shape[0],
            width=data.shape[1],
            count=3,
            driver='GTiff',
            dtype=data.dtype,
            transform=transform,
            compress='lzw',
            crs={'init': 'epsg:3857'},
        )
        gti_image = rasterio.open(
            mname,
            'w',
            height=data_mask.shape[0],
            width=data_mask.shape[1],
            count=1,
            driver='GTiff',
            dtype=data_mask.dtype,
            transform=transform,
            compress='lzw',
            crs={'init': 'epsg:3857'},
        )
        for i in range(0, data.shape[-1]):
            rgb_image.write(data[:,:,i], i+1)
        gti_image.write(data_mask, 1)
        gti_image.close()
        rgb_image.close()

    base_kwargs = {
        'data_pattern': r"(?P<name>[0-9A-Za-z]+)_(?P<SECOND>[A-Za-z0-9]+)_(?P<THIRD>[A-Za-z0-9]+)\.tiff$",
        'type_format': "{SECOND}",
        'id_format': "{name}-{THIRD}",
        'input_source': tempdir.name,
        'validation_percent': 0.4
    }

    loader = FileSystemLoader(**base_kwargs)
    loader.__temp_input_directory = tempdir

    return loader

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
        training_loader, validation_loader = fs8.get_dataset_loader()

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
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loader()
        assert len(training_loader) == 6
        assert len(validation_loader) == 4

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
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loader()

        dataset_name, dataset_parts = next(training_loader)
        assert isinstance(dataset_name, str)
        assert 'RGB' in dataset_parts

        rgb_dset = dataset_parts['RGB']
        isinstance(rgb_dset, DatasetReader)

        assert rgb_dset.count == 3


class TestTileGenerator(object):

    def test_number_of_tiles(self, generated_filesystem_loader):
        training_loader, validation_loader = generated_filesystem_loader.get_dataset_loader()
        training_loader.loop = True
        validation_loader.loop = True


        train_data = DataGenerator(training_loader,
                                   batch_size=None,
                                   input_mapping={
                                       'input_1': {
                                           'primary': True,
                                           'window_shape': (256, 256),
                                           'stride': 256,
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
                                           'window_shape': (256, 256),
                                           'stride': 256,
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

        assert len(train_data) == 432
        assert len(validation_data) == 288
