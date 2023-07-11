import tensorflow as tf
import math
import numpy as np
import random
import threading

import tqdm

from hugin.preprocessing.standardize import SkLearnStandardizer

from .loader import BasePatchLoader, STACAPISearchPatchLoader, Dataset, \
    PatchLoaderAssetView, Concat, BaseRasterTile, Apply


class TFLoader(Dataset, tf.keras.utils.Sequence):
    loader: BasePatchLoader
    batch_size: int

    def __init__(self, dataset: BasePatchLoader, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.dataset))
        batch_x_, batch_y_ = self.dataset[low:high]
        batch_x = {}
        batch_y = {}
        for input_name, input_value in batch_x_.items():
            entries = []
            for entry in batch_x_[input_name]:
                for patch in entry:
                    if isinstance(patch, BaseRasterTile):
                        entries.append(patch.get_value())
                    else:
                        entries.append(patch)
            batch_x[input_name] = np.array(entries)
        for output_name, output_value in batch_y_.items():
            entries = []
            for entry in batch_y_[output_name]:
                for patch in entry:
                    if isinstance(patch, BaseRasterTile):
                        entries.append(patch.get_value())
                    else:
                        entries.append(patch)
            batch_y[output_name] = np.array(entries)
        return batch_x, batch_y

    def get_input_shapes(self):
        rez = {}
        first_batch, _ = self[0]
        for k, v in first_batch.items():
            rez[k] = v.shape[1:]  # ignore batch
        return rez

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class CachedTFLoader(tf.keras.utils.Sequence):
    def __init__(self, loader):
        self.loader = loader
        self.__lock = threading.Lock()
        self.__cache = {}

    def __getitem__(self, idx):
        with self.__lock:
            if idx not in self.__cache:
                val = self.loader[idx]
                self.__cache[idx] = val
            return self.__cache[idx]

    def get_input_shapes(self):
        return self.loader.get_input_shapes()
    def __len__(self):
        return len(self.loader)


if __name__ == '__main__':
    loader = STACAPISearchPatchLoader(
        endpoint="https://stac.sage.uvt.ro/",
        collection_id="sentinel-2-l1c",
        assets=None,  # ["B04", "B02"],
        max_items=6,
        patch_source_asset={
            "asset": "B02",
            "shape": (256, 256),
            "stride": 190
        }
    )

    b08_scaler = SkLearnStandardizer(
        path="s3://sage/public/datasets/paduri/dset-paduri-worldcover2021/scalers/B08_standardizer.pickle",
        fsspec_storage_options={
            "endpoint_url": 'https://storage.info.uvt.ro',
            "anon": False
        }
    )
    b02_scaler = SkLearnStandardizer(
        path="s3://sage/public/datasets/paduri/dset-paduri-worldcover2021/scalers/B02_standardizer.pickle",
        fsspec_storage_options={
            "endpoint_url": 'https://storage.info.uvt.ro',
            "anon": False
        }
    )
    b03_scaler = SkLearnStandardizer(
        path="s3://sage/public/datasets/paduri/dset-paduri-worldcover2021/scalers/B03_standardizer.pickle",
        fsspec_storage_options={
            "endpoint_url": 'https://storage.info.uvt.ro',
            "anon": False
        }
    )
    b04_scaler = SkLearnStandardizer(
        path="s3://sage/public/datasets/paduri/dset-paduri-worldcover2021/scalers/B04_standardizer.pickle",
        fsspec_storage_options={
            "endpoint_url": 'https://storage.info.uvt.ro',
            "anon": False
        }
    )

    loader = Apply(loader,
                   B02=b02_scaler,
                   B03=b03_scaler,
                   B04=b04_scaler,
                   B08=b08_scaler)

    loader = PatchLoaderAssetView(loader,
                                         rgbnir=Concat(["B02", "B03", "B04", "B08"],
                                                       axis=2)
                                         )
    loader = PatchLoaderAssetView(loader,
                                    input_1="rgbnir",
                                    output_1="rgbnir")

    def deteriorate(inp):
        if isinstance(inp, BaseRasterTile):
            inp = inp.get_value()
        width = inp.shape[0]
        height = inp.shape[1]
        patch_width = random.randint(2, width/4)
        patch_height = random.randint(2, height/4)
        start_w = random.randint(0, width-patch_width)
        end_w = start_w + patch_width
        start_h = random.randint(0, height-patch_height)
        end_h = start_h + patch_height

        inp[start_w:end_w, start_h:end_h] = 0
        return inp
    augmented = Apply(loader, input_1=deteriorate)

    #out = augmented[0]
    #
    #print (augmented[0]['input_1'][0].get_value())
    print (len(augmented))
    for i in tqdm.tqdm(range(0, len(augmented))):
        augmented[i]['input_1'][0].get_value()

    # dataset = Dataset(augmented,
    #                   inputs=[
    #                       "input_1"
    #                   ],
    #                   outputs=[
    #                       "output_1"
    #                   ]
    #
    #                   )
    #
    # #
    # # dataset[0]
    # # # rez = dataset[0:2]
    # # # rez
    # # tf_dataset = TFLoader(dataset=dataset, batch_size=2)
    # # result = tf_dataset[0]
    # # print(result)
