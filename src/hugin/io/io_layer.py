import os
import inspect
from abc import ABC, abstractmethod

import numpy as np

import rasterio as rio
import rioxarray as riox
import geopandas as gp
from fiona.errors import DriverError
from rasterio.errors import RasterioIOError


class Component(ABC):
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @abstractmethod
    def read(self, band=None):
        raise NotImplementedError
    
    @abstractmethod
    def write(self):
        raise NotImplementedError

    @abstractmethod
    def open(self, path_to_a_file):
        raise NotImplementedError

    @abstractmethod
    def get_new_object(self, path_to_component):
        raise NotImplementedError


class ComponentFactory:
    def __init__(self):
        self._component_types = {'tiff': RasterComponent,
                                 'tif': RasterComponent,
                                 'jpg': RasterComponent,
                                 'nc': NetCDFComponent,
                                 'shp': GeoJSONComponent,
                                 'geojson': GeoJSONComponent}

    def get_new_object(self, path_to_component):
        try:
            ext = path_to_component.split('.')[1]
            return self._component_types[ext]().get_new_object(path_to_component)
        except IndexError:
            pass
            # We somehow get the path to the directory here as well?


class GeoJSONComponent(Component):
    def __init__(self):
        pass

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.data[item]

    def read(self, band=None):
        pass

    def write(self):
        pass

    def open(self, path_to_a_file):
        try:
            self.data = gp.read_file(path_to_a_file)
        except DriverError as e:
            df = gp.GeoDataFrame()
            df['geometry'] = ''
            self.data = df

    def get_new_object(self, path_to_component):
        self.open(path_to_component)
        return self


class NetCDFComponent(Component):
    def __init__(self):
        pass

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def read(self, band=None, window=None):
        if band is not None:
            if window is not None:
                x = self.data.rio.isel_window(window)[band].data
            else:
                x = self.data[band].data
        else:
            if window is not None:
                x = self.data.rio.isel_window(window).to_array().data
            else:
                x = self.data.to_array().data
        data = np.squeeze(x)

        return data

    def write(self):
        raise NotImplementedError

    def open(self, path_to_a_file):
        self.name = path_to_a_file
        self.data = riox.open_rasterio(path_to_a_file)
        self.shape = self.data.rio.shape
        self.width, self.height = self.data.rio.width, self.data.rio.height

    def get_new_object(self, path_to_component):
        self.open(path_to_component)
        return self


class RasterComponent(Component):
    def __init__(self):
        pass

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def read(self, band=None, window=None):
        data = self.data.read(band, window=window)
        return data

    def write(self, path_to_file):
        pass

    def open(self, path_to_a_file):
        self.data = rio.open(path_to_a_file)
        [setattr(self, m[0], m[1]) for m in inspect.getmembers(self.data) if not m[0].startswith('_')]

    def get_new_object(self, path_to_component):
        self.open(path_to_component)
        return self
