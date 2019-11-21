import numpy as np
import geopandas as gpd
import rasterio as rio
import rasterio.features

from hugin.engine.core import RasterGenerator


class GroundTruthComponentGenerator(RasterGenerator):
    """
    Generator generating ground truth components using shapefiles/geojsons
    """
    def __init__(self, base_component, shape_input, growth_factor=0, default_value=1, all_touched=True):
        """
        Args:
            base_component: the component we should use for detecting the size, etc
            shape_input: path of a geojson or shapefile whose geometries we burn in the GTI \
            or an existing component type's name from the FileSystemLoader
            growth_factor: how much we want to increase the width of the geometries
            default_value: value to write in raster where geometries overlap
            all_touched: If True, all pixels touched by geometries will be burned in. \
            If false, only pixels whose center is within the polygon will be burned in.
        """
        self._base_component = base_component
        self._shape_input = shape_input
        self._growth_factor = growth_factor
        self._default_value = default_value
        self._all_touched = all_touched

    def _create_gti(self, scene_components):
        # this means shape_input is a component in our scene_components already found by FSLoader
        if self._shape_input in scene_components.keys():
            self._shape_input = scene_components[self._shape_input]

        with rio.open(scene_components[self._base_component]) as base_raster:
            gt_component = gpd.read_file(self._shape_input)
            shapes = gt_component['geometry']
            gti_component = np.zeros(base_raster.shape, dtype=np.uint16)
            profile = base_raster.profile
            profile.update(count=1, nodata=0)

            if self._growth_factor != 0:
                shapes = shapes.buffer(self._growth_factor)

            rasterio.features.rasterize(shapes,
                                        out_shape=gti_component.shape,
                                        default_value=self._default_value,
                                        all_touched=self._all_touched,
                                        transform=base_raster.transform,
                                        out=gti_component)

            # works for the first one:
            fname = scene_components[self._base_component].split('.')[0] + "_gti.tif"
            with rasterio.open(fname, 'w', **profile) as dst:
                dst.write(gti_component.astype(np.uint8), 1)

            return rio.io.MemoryFile(bytes(gti_component))

    def __call__(self, scene_components):
        return self._create_gti(scene_components)
