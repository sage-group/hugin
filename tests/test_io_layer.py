import os
from random import randint
from copy import deepcopy

import numpy as np
from rasterio.windows import Window

from hugin import io
from hugin.io import FileSystemLoader


def test_io_functionality(generated_filesystem_loader):
    cf = io.ComponentFactory()
    files = os.listdir(generated_filesystem_loader.input_source)
    netcdf_files = map(lambda f: os.path.join(generated_filesystem_loader.input_source, f),
                       filter(lambda x: x if x.endswith('.nc') else None, files))
    tiff_files = map(lambda f: os.path.join(generated_filesystem_loader.input_source, f),
                     filter(lambda x: x if 'RGB' in x else None, files))
    geojson_files = map(lambda f: os.path.join(generated_filesystem_loader.input_source, f),
                        filter(lambda x: x if x.endswith('.geojson') else None, files))

    _test_NetCDFComponent(deepcopy(netcdf_files), cf)
    _test_RasterComponent(deepcopy(tiff_files), cf)
    _test_ncdf_vs_raster(deepcopy(tiff_files), deepcopy(netcdf_files), cf)


def _test_NetCDFComponent(netcdf_files, factory):
    netcdf_components = []
    win = Window(0, 0, 256, 256)
    
    for file in netcdf_files:
        netcdf_components.append(factory.get_new_object(file))

    assert len(netcdf_components) == 10
    assert map(lambda x: isinstance(x, io.NetCDFComponent), netcdf_components)
    assert all([c.read().shape == (3, 1979, 2131) for c in netcdf_components])
    assert all([c.read('Band1').shape == (1979, 2131) for c in netcdf_components])
    assert all([c.read('Band2', window=win).shape == (256, 256) for c in netcdf_components])
    assert all([c.read(window=win).shape == (3, 256, 256) for c in netcdf_components])


def _test_RasterComponent(gtiff_files, factory):
    gtiff_components = []
    win = Window(0, 0, 256, 256)

    for file in gtiff_files:
        gtiff_components.append(factory.get_new_object(file))

    assert len(gtiff_components) == 10
    assert map(lambda x: isinstance(x, io.gtiffComponent), gtiff_components)
    assert all([c.read().shape == (3, 1979, 2131) for c in gtiff_components])
    assert all([c.read(1).shape == (1979, 2131) for c in gtiff_components])
    assert all([c.read(2, window=win).shape == (256, 256) for c in gtiff_components])
    assert all([c.read(window=win).shape == (3, 256, 256) for c in gtiff_components])


def randint_twosmultiple():
    x = randint(0, 255)
    while x % 2 != 0:
        x = randint(0, 255)
    return x


def _test_ncdf_vs_raster(gtiff_components, netcdf_components, factory):
    gtiff_files = []
    netcdf_files = []

    for file in gtiff_components:
        gtiff_files.append(factory.get_new_object(file))

    for file in netcdf_components:
        netcdf_files.append(factory.get_new_object(file))

    gtiff_files.sort(key=lambda x: x.name)
    netcdf_files.sort(key=lambda x: x.name)

    assert len(gtiff_files) == 10 and len(netcdf_files) == 10
    assert len(gtiff_files) == len(netcdf_files)
    assert all([gtiff_files[i].read().shape == netcdf_files[i].read().shape
                for i in range(0, len(gtiff_files))])


    assert all([np.array_equal(gtiff_files[i].read(), netcdf_files[i].read())
                for i in range(0, len(gtiff_files))])

    for i in range(len(gtiff_files)):
        for shift in range(0, gtiff_files[i].read().shape[1], 256):
            win = Window(shift, shift, 256, 256)
            for x in range(1, 4):
                assert all([np.array_equal(netcdf_files[i].read(f'Band{x}'),
                                           gtiff_files[i].read(x))])
                assert all([np.array_equal(netcdf_files[i].read(f'Band{x}', window=win),
                                           gtiff_files[i].read(x, window=win))])

        for shift in range(0, gtiff_files[i].read().shape[1], randint(0, 256)):
            shp = randint_twosmultiple()
            win = Window(shift, shift, shp, shp)
            for x in range(1, 4):
                assert all([np.array_equal(netcdf_files[i].read(f'Band{x}', window=win),
                                           gtiff_files[i].read(x, window=win))])



def _testGeoJSONComponent(loader, factory):
    pass

