import json
import os
import random
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import rasterio
import rasterio.features
import rasterio.warp

from rasterio.transform import from_origin
from skimage.draw import rectangle, circle

from hugin.io import FileSystemLoader
from hugin.preprocessing.rasterize import RasterFromShapesGenerator


class GenerateFileSystemLoader():
    __loader = None

    def __init__(self, multi_json=True):
        self._multi_json = multi_json
        pass

    def __call__(self, width=2131, height=1979, size=35, num_images=10):
        if self.__loader is None:
            self.__loader = self._generate_filesystem_loader(width, height, size, num_images)
        return self.__loader

    def _generate_filesystem_loader(self, width=2131, height=1979, size=35, num_images=10):
        tempdir = TemporaryDirectory("-hugin")
        tempdir_name = tempdir.name
        random.seed(42)
        coords = (21.2086, 45.7488)
        num_rect = 5
        num_circle = 4
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
        match_color = colors[0]

        single_geojson = features_col = {'type': 'FeatureCollection', 'features': []}
        sgname = os.path.join(tempdir_name, "single_GT.geojson")

        for imgidx in range(0, num_images):
            data = np.zeros((height, width, 3), dtype=np.uint8)
            data_mask = np.zeros((height, width), dtype=np.uint8)

            for i in range(0, num_rect):
                color = random.choice(colors)
                startx = random.randint(1, width - size - 5)
                starty = random.randint(1, height - size - 5)

                endx = startx + size
                endy = starty + size
                rr, cc = rectangle((startx, endx), (starty, endy), shape=data_mask.shape)
                data[rr, cc] = color
                if color == match_color:
                    data_mask[rr, cc] = 1

            for i in range(0, num_circle):
                color = random.choice(colors)
                startx = random.randint(1, width - size - 5)
                starty = random.randint(1, height - size - 5)
                radius = random.randint(1, 2 * size)
                rr, cc = circle(startx, starty, radius, shape=data_mask.shape)
                data[rr, cc] = color
                if color == match_color:
                    data_mask[rr, cc] = 1

            fname = os.path.join(tempdir_name, f"TEST_RGB_{imgidx + 1}.tiff")
            mname = os.path.join(tempdir_name, f"TEST_GTI_{imgidx + 1}.tiff")
            gname = os.path.join(tempdir_name, f"TEST_GT_{imgidx + 1}.geojson")

            res = 0.5
            transform = from_origin(coords[0], coords[1], res, res)
            coords = (coords[0] + (res * width), coords[1])
            rgb_image = rasterio.open(
                fname,
                'w',
                height=data.shape[0],
                width=data.shape[1],
                count=3,
                driver='GTiff',
                dtype=data.dtype,
                transform=transform,
                crs={'init': 'epsg:4326'},
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
                compress='none',
                tiled=True,
                crs={'init': 'epsg:4326'},
            )

            for i in range(0, data.shape[-1]):
                rgb_image.write(data[:, :, i], i + 1)

            gti_image.write(data_mask, 1)
            gti_image.close()
            rgb_image.close()

            with rasterio.open(mname) as gti:
                src = gti.read()
                src[src > 0] = 1

                features_col = {'type': 'FeatureCollection', 'features': []}
                for geom, _ in rasterio.features.shapes(src, mask=src, transform=transform):
                    features_col['features'].append({'geometry': geom})
                    single_geojson['features'].append({'geometry': geom})

                with open(gname, 'w') as dst_geojson:
                    dst_geojson.write(json.dumps(features_col))

        for file in filter(lambda x: 'RGB' in x, os.listdir(tempdir_name)):
            _outf = file.replace('.tiff', '.nc').replace('RGB', 'NCDF')

            subprocess.Popen(['gdal_translate', '-of', 'netCDF',
                              os.path.join(tempdir_name, file),
                              os.path.join(tempdir_name, _outf)]).wait()

        with open(sgname, 'w') as dst_geojson:
            dst_geojson.write(json.dumps(single_geojson))

        base_kwargs = {
            'data_pattern': r"(?P<name>[0-9A-Za-z]+)_(?P<SECOND>[A-Za-z0-9]+)_(?P<THIRD>[A-Za-z0-9]+)\.(tiff|geojson|nc)$",
            'id_format': "{name}_{THIRD}",
            'type_format': "{SECOND}",
            'input_source': tempdir_name,
            'validation_percent': 0.2,
        }

        if self._multi_json == True:
            base_kwargs['dynamic_types'] = {'GENERATED_GROUNDTRUTH': RasterFromShapesGenerator(base_component='RGB',
                                                                                               shape_input='GT')}
        else:
            base_kwargs['dynamic_types'] = {'GENERATED_GROUNDTRUTH': RasterFromShapesGenerator(base_component='RGB',
                                                                                               shape_input=sgname)}

        loader = FileSystemLoader(**base_kwargs)
        loader.__temp_input_directory = tempdir

        return loader


@pytest.fixture
def generated_filesystem_loader(multiple_json=True):
    loader = GenerateFileSystemLoader(multiple_json)
    return loader()
