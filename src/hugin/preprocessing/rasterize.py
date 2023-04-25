# -*- coding: utf-8 -*-
__license__ = """Copyright 2023 West University of Timisoara

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
    """

import tempfile

import geopandas as gpd
import numpy as np
import rasterio.features

from hugin.engine.core import RasterGenerator


class ComponentGenerator(RasterGenerator):
    def __init__(self, base_component):
        """
        Args:
            base_component: the component we should use for detecting the size
        """
        self._base_component = base_component

    def _create_component(self, scene_components):
        raise NotImplementedError  # this should not be implemented here

    def __call__(self):
        raise NotImplementedError  # this should not be implemented here


class RasterFromShapesGenerator(ComponentGenerator):
    """
    Generator creating ground truth components from ESRI shapefiles or geojsons
    """

    def __init__(
        self,
        base_component,
        shape_input,
        growth_factor=0,
        default_value=1,
        all_touched=False,
    ):
        """
        Args:
            shape_input: path of a geojson or shapefile whose \
              geometries we burn in the GTI or an existing component name
            growth_factor: how much we want to increase the width of the geometries
            default_value: value to write in raster where geometries overlap
            all_touched: If True, all pixels touched by geometries will be burned in. \
            If false, only pixels whose center is within the polygon will be burned in.
        """
        super().__init__(base_component)
        self._shape_input = shape_input
        self._growth_factor = growth_factor
        self._default_value = default_value
        self._all_touched = all_touched

    def _create_component(self, scene_components):
        if self._shape_input in scene_components.keys():
            _shape_input = scene_components[self._shape_input]
        else:
            _shape_input = gpd.read_file(self._shape_input)

        base_raster = scene_components[self._base_component]
        gt_component = _shape_input

        shapes = gt_component["geometry"]
        profile = base_raster.profile
        profile.update(count=1, nodata=0)
        gti_component = np.zeros(base_raster.shape, dtype=profile["dtype"])

        if not shapes.empty:
            if self._growth_factor != 0:
                shapes = shapes.buffer(self._growth_factor)

            rasterio.features.rasterize(
                shapes,
                out_shape=gti_component.shape,
                default_value=self._default_value,
                all_touched=self._all_touched,
                transform=base_raster.transform,
                out=gti_component,
            )

        with tempfile.NamedTemporaryFile() as tmp_file:
            allowed_profile_options = {
                "driver",
                "dtype",
                "nodata",
                "width",
                "height",
                "count",
                "crs",
                "transform",
            }
            new_profile = {
                k: v for k, v in profile.items() if k in allowed_profile_options
            }
            with rasterio.open(tmp_file.name, "w", **new_profile) as dst:
                dst.write(gti_component, 1)
            gti_file = rasterio.open(tmp_file.name, "r")

        return gti_file

    def __call__(self, scene_components):
        return self._create_component(scene_components)
