import pytest
from tempfile import TemporaryDirectory
from PIL import Image,ImageDraw
import numpy as np
import rasterio
from rasterio.transform import from_origin
from hugin.io import FileSystemLoader
import os

import random

def generate_filesystem_loader(width=2131, height=1979, size=35, num_images=10):
    tempdir = TemporaryDirectory("-hugin")
    tempdir_name = tempdir.name
    random.seed(42)

    match_color = "red"

    for imgidx in range(0, num_images):
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

        fname = os.path.join(tempdir_name, "TEST_RGB_{}.tiff".format(imgidx+1))
        mname = os.path.join(tempdir_name, "TEST_GTI_{}.tiff".format(imgidx+1))

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
            compress='none',
            tiled=True,
            crs={'init': 'epsg:3857'},
        )
        data = np.array(im.getdata()).reshape(data.shape).astype(np.uint8)
        new_data_mask = np.array(mask.getdata())
        data_mask = new_data_mask.reshape(data_mask.shape)
        data_mask = data_mask.astype(np.uint8)
        for i in range(0, data.shape[-1]):
            rgb_image.write(data[:,:,i], i+1)
        gti_image.write(data_mask, 1)
        gti_image.close()
        rgb_image.close()

    base_kwargs = {
        'data_pattern': r"(?P<name>[0-9A-Za-z]+)_(?P<SECOND>[A-Za-z0-9]+)_(?P<THIRD>[A-Za-z0-9]+)\.tiff$",
        'type_format': "{SECOND}",
        'id_format': "{name}_{THIRD}",
        'input_source': tempdir_name,
        'validation_percent': 0.2
    }

    loader = FileSystemLoader(**base_kwargs)
    loader.__temp_input_directory = tempdir

    return loader

@pytest.fixture
def generated_filesystem_loader():
    return generate_filesystem_loader(num_images=10)