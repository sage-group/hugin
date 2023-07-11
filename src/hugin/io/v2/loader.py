import pystac
import pystac_client as pyc
import mimetypes
import rasterio
import math
import numpy as np
import shapely.geometry
import pyproj
from rasterio.errors import RasterioIOError
from shapely.geometry import MultiPolygon
from shapely.ops import transform
from rasterio.plot import show
import tqdm
from typing import Union
import backoff
import fiona


from rasterio import DatasetReader
from rasterio.windows import Window, from_bounds

_WGS84_PROJECT = pyproj.CRS('EPSG:4326')

RASTERIO_MIME_TYPES = {
    'image/tiff; application=geotiff; profile=cloud-optimized',
    'image/tiff; application=geotiff'
}

FIONA_MIME_TYPES = {
    "application/vnd.flatgeobuf",
    "application/geopackage+sqlite3",
    "application/geo+json"
}


#def get_patches(rio: DatasetReader, width, height, stride, crs=None):
def get_patches(rio: DatasetReader, window_width, window_height, stride, crs=None):
    image_height, image_width = rio.shape
    if window_width != stride:
        xtiles = math.ceil((image_width - window_width) / float(stride) + 2)
    else:
        xtiles = math.ceil((image_width - window_height) / float(stride) + 1)

    if window_height != stride:
        ytiles = math.ceil((image_height - window_height) / float(stride) + 2)
    else:
        ytiles = math.ceil((image_height - window_height) / float(stride) + 1)

    ytile = 0
    while ytile < ytiles:
        y_start = ytile * stride
        y_end = y_start + window_height
        if y_end > image_height:
            y_start = image_height - window_height
            y_end = y_start + window_height

        ytile += 1
        xtile = 0
        while xtile < xtiles:
            x_start = xtile * stride
            x_end = x_start + window_width
            if x_end > image_width:
                x_start = image_width - window_width
                x_end = x_start + window_width

            xtile += 1

            window = Window(x_start, y_start, window_width, window_height)
            patch = rio.window_bounds(window)
            yield window, patch


class BaseRasterTile:
    def get_value(self, channel: int = None) -> np.ndarray:
        raise NotImplementedError()


class RasterTile(BaseRasterTile):
    rio: rasterio.DatasetReader
    window: Window
    def __init__(self, rio: rasterio.DatasetReader, bounds=tuple, project=None):
        self.rio = rio
        self.window = from_bounds(*bounds, transform=rio.transform)
        self.__geo_interface = None
        self.project = project
        self.bounds = bounds


        self.value = None

    @property
    def __geo_interface__(self):
        if self.__geo_interface is not None:
            return self.__geo_interface

        geom = shapely.geometry.box(*self.bounds)
        if self.project is not None:
            geom = transform(self.project, geom)
        geom = shapely.geometry.mapping(geom)
        self.__geo_interface = geom
        return geom




    @backoff.on_exception(backoff.expo, RasterioIOError, max_tries=8)
    def get_value(self, channel: int = None):
        if self.value is not None:
            return self.value

        if channel is None:
            data = self.rio.read(window=self.window)  # ToDo: Boundless ?
            data = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)
            value = data
        else:
            value = self.rio.read(channel, window=self.window)
        self.value = value
        return self.value

    def plot(self):
        show(self.get_value(), transform=self.rio.window_transform(self.window))


class LazzyRasterProcessor:
    pass

class LazzyRasterTileProcessor(LazzyRasterProcessor, BaseRasterTile):
    tile: BaseRasterTile

    def __init__(self, tile: BaseRasterTile, callback):
        self.tile = tile
        self.callback = callback
        self.__geo_interface = None

    @property
    def __geo_interface__(self):
        if self.__geo_interface is not None:
            return self.__geo_interface
        if isinstance(self.tile, BaseRasterTile):
            self.__geo_interface__ = self.tile.__geo_interface__.copy()
        else:
            shapes = [shapely.geometry.shape(v) for v in self.tile]
            shape = MultiPolygon(shapes)
            self.__geo_interface__ = shapely.geometry.mapping(shape)


    def get_value(self, channel: int = None):
        value = self.callback(self.tile)
        return value

    def _plot(self, tile:(BaseRasterTile, list)):
        if isinstance(tile, BaseRasterTile):
            tile.plot()
        else:
            for entry in tile:
                self._plot(entry)

    def plot(self):
        if isinstance(self.tile, (BaseRasterTile, list, tuple)):
            self._plot(self.tile)
        else:
            raise NotImplementedError()




class Scene:
    """
    Used for representing a single scene with all corresponding bands and assets
    """

    def __init__(self, raster_assets: dict[str, dict], vector_assets: dict[str, dict],
                 patches: tuple[float, float, float, float], crs: int):
        self.raster_assets = {}
        self.project = None
        self.crs = None
        self._cache = {}
        self.rios: dict[str, DatasetReader] = {}
        for asset_name, asset_value in raster_assets.items():
            new_value = asset_value.copy()

            with rasterio.open(asset_value['url']) as rio:
                scene_crs = rio.crs
                if self.crs is None:
                    self.crs = scene_crs
                else:
                    if self.crs != rio.crs:
                        raise ValueError(
                            f"Mismatched CRS in scene {self.crs} != {scene_crs}")
                if self.project is None:
                    scene_projection = pyproj.CRS(rio.crs.data['init'])
                    self.project = pyproj.Transformer.from_crs(scene_projection,
                                                               _WGS84_PROJECT,
                                                               always_xy=True).transform
            self.raster_assets[asset_name] = new_value
        self.vector_assets = {}
        for asset_name, asset_value in vector_assets.items():
            new_value = asset_value.copy()
#            raise NotImplementedError()
            self.raster_assets[asset_name] = new_value

        self.patches = patches
        self.crs = crs

    def __len__(self):
        return len(self.patches)

    def get_item(self, item, assets: Union[set, tuple, list] = None):
        try:
            patch = self.patches[item]
        except IndexError as e:
            raise e
        result = {}
        for raster_asset_name, raster_asset_value in self.raster_assets.items():
            if assets is not None:
                if raster_asset_name not in assets:
                    continue
            if raster_asset_name not in self.rios:
                self.rios[raster_asset_name] = None

            if self.rios[raster_asset_name] is None:
                self.rios[raster_asset_name] = rasterio.open(raster_asset_value["url"])

            raster_tile = RasterTile(self.rios[raster_asset_name], patch, self.project)
            # window = from_bounds(*patch, transform=rio.transform)
            # rez = rio.read(1, window=window) #ToDo: Boundless ?
            result[raster_asset_name] = raster_tile
        return result

    def __getitem__(self, item):
        return self.get_item(item)


class BasePatchLoader(object):
    """
    Base class for all data `loaders`
    """

    def __len__(self):
        """
        :return: number of patches
        """
        raise NotImplementedError()

    def __getitem__(self, item: int):
        raise NotImplementedError()

    def __iter__(self):
        for idx in range(0, len(self)):
            yield self[idx]

def explode(coords):
    """Explode a GeoJSON geometry's coordinates object and yield coordinate tuples.
    As long as the input is conforming, the type of the geometry doesn't matter.
    From: https://gis.stackexchange.com/questions/90553/getting-extent-bounds-of-each-feature-using-fiona
    """
    for e in coords:
        if isinstance(e, (float, int)):
            yield coords
            break
        else:
            for f in explode(e):
                yield f

class STACPatchLoader(BasePatchLoader):
    collection: Union[pyc.CollectionClient, pystac.Collection]
    scenes: list
    scenes_count: list[tuple[Scene, int]]
    items: list

    def __init__(self,
                 stac_items: Union[
                     pystac.Collection,
                     pyc.CollectionClient,
                     pyc.ItemSearch
                 ],
                 assets=[], patch_source_asset=None):
        self.scenes = []
        self.scenes_count = []
        if isinstance(stac_items, pyc.ItemSearch):
            self.items = [item for item in stac_items.items()]
        elif isinstance(stac_items, pyc.CollectionClient):
            self.items = [item for item in stac_items.get_items()]
        elif isinstance(stac_items, (pystac.Collection, pystac.Catalog)):
            self.items = [item for item in stac_items.get_items()]
        else:
            raise NotImplementedError()

        if assets is None:
            self.assets = []
        else:
            self.assets = assets.copy()
        last_count = 0
        for stac_item in tqdm.tqdm(self.items):
            properties = stac_item.properties
            raster_item_assets = {}
            vector_item_assets = {}
            for stac_item_asset_name, stac_item_asset_value in stac_item.assets.items():
                if assets is not None:
                    if stac_item_asset_name not in assets:
                        continue
                    else:
                        self.assets.append(stac_item_asset_name)
                else:
                    self.assets.append(stac_item_asset_name)
                if stac_item_asset_value.media_type is None:
                    mediatype = mimetypes.guess_type(stac_item_asset_value.href)
                else:
                    mediatype = stac_item_asset_value.media_type
                if mediatype in RASTERIO_MIME_TYPES:
                    raster_item_assets[stac_item_asset_name] = {
                        'url': stac_item_asset_value.href,
                        'mediatype': mediatype
                    }
                if mediatype in FIONA_MIME_TYPES:
                    vector_item_assets[stac_item_asset_name] = {
                        'url': stac_item_asset_value.href,
                        'mediatype': mediatype
                    }
            # Prepare the patches
            if patch_source_asset is None:
                # No patch source specified, what should we do #ToDo
                raise NotImplementedError
            else:
                patch_source_asset_name = patch_source_asset['asset']

                if patch_source_asset_name in raster_item_assets:
                    patch_source_shape = patch_source_asset.get('shape', (256, 256))
                    window_width, window_height = patch_source_shape
                    patch_source_stride = patch_source_asset.get('stride',
                                                                 patch_source_shape[0])
                    with rasterio.open(
                            raster_item_assets[patch_source_asset_name]["url"],
                            "r") as rio:
                        epsg = properties.get('proj:epsg', rio.crs)
                        patches = [patch[1] for patch in
                                   get_patches(rio, window_width, window_height,
                                               patch_source_stride, epsg)]
                elif patch_source_asset_name in vector_item_assets:
                    with fiona.open(vector_item_assets[patch_source_asset_name]['url'],
                                    "r") as fio:
                        epsg = fio.crs
                        patches = []
                        for entry in fio:
                            x, y = zip(*list(
                                explode(
                                    entry['geometry']['coordinates']
                                ))
                            )
                            patches.append((min(x), min(y), max(x), max(y)))
                else:
                    raise ValueError(f"{patch_source_asset_name} not known in {stac_item.id}: {raster_item_assets}") # noqa: E501
            scene = Scene(raster_assets=raster_item_assets,
                          vector_assets=vector_item_assets, patches=patches, crs=epsg)
            self.scenes.append(scene)
            patches_count = len(scene)
            self.scenes_count.append((scene, last_count + patches_count))
            last_count += patches_count
        self._count = last_count

    def get_single_item(self, item: int, assets=None):
        low = 0
        high = len(self.scenes_count)
        mid = 0
        while low <= high:
            mid = (high + low) // 2
            current_scene, current_idx_cumulative_count = self.scenes_count[mid]
            current_scene_length = len(current_scene)

            current_bin_start = current_idx_cumulative_count - current_scene_length
            current_bin_end = current_idx_cumulative_count
            if item > current_bin_end:
                low = mid + 1
            elif item < current_bin_start:
                high = mid - 1
            else:
                idx = item - current_bin_start - 1  # Todo: why -1 ?
                try:
                    return self.scenes[mid].get_item(idx, assets)
                except IndexError as e:
                    raise e
        return IndexError(f"Could not find {item}")

    def get_item(self, item, assets):
        indices = None
        if isinstance(item, slice):
            indices = range(item.start, item.stop,
                            item.step if item.step is not None else 1)
        elif isinstance(item, tuple) or isinstance(item, tuple):
            indices = item

        if indices is not None:
            result = {}
            for i in indices:
                item_data = self.get_single_item(i, assets)
                for item_data_key, item_data_value in item_data.items():
                    if item_data_key not in result:
                        result[item_data_key] = []
                    result[item_data_key].append(item_data_value)
            return result
        else:
            rez = self.get_single_item(item, assets)
            for k, v in rez.items():
                rez[k] = [v]
            return rez

    def __getitem__(self, item):
        return self.get_item(item, self.assets)

    def __len__(self):
        return self.scenes_count[-1][1]

class PatchLoaderView(BasePatchLoader):
    def __init__(self, idx, loader):
        self._loader = loader
        self._idx = idx

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, item):
        if isinstance(item, slice):
            idx = self._idx[item]
            self._loader[idx]
            raise NotImplementedError()
        else:
            return self._loader[self._idx[item]]


class Concat():
    def __init__(self, assets, axis):
        self.assets = assets
        self.axis = axis

    def __call__(self, loader: BasePatchLoader, item: Union[slice, list, tuple]):
        def concat(x):
            vals = []
            for v in x:
                if isinstance(v, np.ndarray):
                    v_ = v
                else:
                    v_ = v.get_value()
                vals.append(v_)
            return np.concatenate(vals, axis=self.axis)
        value = loader.get_item(item, self.assets)
        new_values = []
        length = len(value[self.assets[0]])
        for i in range(0, length):
            values = []
            for asset_name in self.assets:
                values.append(value[asset_name][i])
            new_values.append(
                LazzyRasterTileProcessor(
                    values,
                    concat
                )
            )
        return new_values



class Apply(BasePatchLoader):
    loader: BasePatchLoader

    def __init__(self, loader: BasePatchLoader, **asset_callable_mapping):
        self.asset_callable_mapping = asset_callable_mapping
        self.loader = loader

    def get_item(self, item, asset_names=None):
        result = {}
        original = self.loader.get_item(item, asset_names)
        if asset_names is not None:
            valid_asset_names = asset_names
        else:
            valid_asset_names = set(original.keys())
        for asset_name in valid_asset_names:
            original_values_ = original[asset_name]
            if not isinstance(original_values_, (tuple, list)):
                original_values = [original_values_, ]
            else:
                original_values = original_values_
            if asset_name not in self.asset_callable_mapping:
                result[asset_name] = original_values
                continue
            new_values = []
            callback = self.asset_callable_mapping[asset_name]
            for value in original_values:
                new_values.append(LazzyRasterTileProcessor(value, callback))
            if not isinstance(original_values_, (tuple, list)):
                new_values = new_values[0]

            result[asset_name] = new_values
        return result

    def __getitem__(self, item):
        return self.get_item(item)

    def __len__(self):
        return len(self.loader)



class PatchLoaderAssetView(BasePatchLoader):
    def __init__(self, loader, **mapping):
        self.loader = loader
        self.mapping = mapping
        self.asset_names = set(self.mapping.keys())

    def __len__(self):
        return len(self.loader)

    def get_item(self, item, asset_names=[]):
        if not asset_names:
            asset_names = self.asset_names
        result = {}
        for asset_name in asset_names:
            if asset_name not in self.mapping:
                raise KeyError(f"No such asset {asset_name}")
            hndlr = self.mapping[asset_name]
            if callable(hndlr):
                value = hndlr(self.loader, item)
            elif isinstance(hndlr, str):
                value = self.loader.get_item(item, [hndlr, ])[hndlr]
            else:
                raise NotImplementedError()
            result[asset_name] = value
        return result

    def __getitem__(self, item):
        return self.get_item(item)


class STACAPISearchPatchLoader(STACPatchLoader):
    api: pyc.Client

    def __init__(self, endpoint: str, collection_id: str, max_items: int, assets,
                 bbox=None, intersects=None, datetime=None, query=None, filter=None,
                 filter_lang=None, sortby=None, patch_source_asset=None):
        self.stac_endpoint = endpoint
        self.api = pyc.Client.open(self.stac_endpoint)
        stac_collection = self.api.search(collections=[collection_id, ],
                                          max_items=max_items, bbox=bbox,
                                          intersects=intersects, datetime=datetime,
                                          query=query, filter=filter,
                                          filter_lang=filter_lang, sortby=sortby)
        super().__init__(stac_collection, assets, patch_source_asset=patch_source_asset)


class Dataset(object):
    def __init__(self, loader: BasePatchLoader,
                 inputs: dict,
                 outputs: dict):
        self.loader = loader
        self.inputs = inputs
        self.outputs = outputs

        self.asset_names = set(self.inputs).union(self.outputs)

    def get_item(self, item):
        x = {}
        y = {}
        rez = self.loader.get_item(item, self.asset_names)

        for input_name in self.inputs:
            if input_name not in x:
                x[input_name] = []
            x[input_name].append(rez[input_name])
        for output_name in self.outputs:
            if output_name not in y:
                y[output_name] = []
            y[output_name].append(rez[output_name])
        return x, y

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, item):
        return self.get_item(item)

    def _postprocess(self, entry, spec):
        new_entry = entry
        return new_entry

    def _get_required_assets(self, input_spec: dict):
        pass


if __name__ == "__main__":

    from stactools.core.io import FsspecStacIO
    import pystac

    if True:
        stac_io = FsspecStacIO()
        catalog_s3 = pystac.Catalog.from_file(
            "s3://sage/public/datasets/paduri/paduri-worldcover2021/catalog/train/catalog.json",
            stac_io=stac_io)
        loader = STACPatchLoader(
            stac_items=catalog_s3,
            assets=["B02", "B03", "B04", "B08"],
            patch_source_asset={
                "asset": "B02",
                "shape": (256, 256),
                "stride": 190
            }
        )
    1/0


    loader = STACAPISearchPatchLoader(
        endpoint="https://stac.sage.uvt.ro/",
        collection_id="sentinel-2-l1c",
        assets=None,  # ["B04", "B02"],
        max_items=2,
        patch_source_asset={
            "asset": "B02",
            "shape": (256, 256),
            "stride": 190
        }
    )
    # rez = loader.get_item(0, assets=['B04', 'B02'])
    # value = rez['B02'].get_value()
    # print (value)

    rez = loader[3481]
    print(rez)

    # print(loader[3481])
    # print(loader[6961])
    # print(loader[0:10])
    # dataset = Dataset(loader=loader,
    #                   x_assets={
    #                       "B02": {
    #                           "channel": 1,
    #                           "preprocessing": [
    #
    #                           ]
    #                       }
    #                   },
    #                   y_assets={
    #                       "B08": {
    #
    #                       }
    #                   }
    #                   )
    # result = dataset[0:2]
    # print(result)
