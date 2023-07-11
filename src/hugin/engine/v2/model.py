import logging
import random
import tempfile

from hugin.models.unet import UNet

from hugin.engine.keras import KerasModel
from hugin.preprocessing.standardize import SkLearnStandardizer
from hugin.io.v2.loader import STACPatchLoader, PatchLoaderAssetView,\
    Dataset, Apply, Concat, BaseRasterTile
from hugin.io.v2.tf import TFLoader

import tensorflow as tf

log = logging.getLogger()

class RasterScenePatchPredictor():
    def __init__(self, model_path, model_builder):
        self.model_path = model_path
        self.model_builder = model_builder

    def predict(self, batch, batch_size=None):
        if self.model is None:
            self.__load_model(self.model_path)
        batch_size = batch_size if batch_size else self.batch_size
        prediction = self.model.predict(batch, batch_size=batch_size)
        return prediction

class RasterScenePatchTrainer():
    _base_directory = None

    def __init__(self,
                 model,
                 *args,
                 pre_processors=[], base_directory=None, **kws):
        self.model = model
        if base_directory is not None:
            self.base_directory = base_directory
        else:
            tmdir = tempfile.mkdtemp()
            log.info(f"No base directory specified. Using temporary directory: {tmdir}")
            self.base_directory = tmdir

    def train(self, training, validation=None):
        self.model.fit_generator(training, validation_data=validation)

    def predict(self, sequence):
        self.model

    @property
    def base_directory(self):
        return self._base_directory

    @base_directory.setter
    def base_directory(self, value):
        self._base_directory = value
        self.model.base_directory = value

    def save(self):
        raise NotImplementedError()


if __name__ == '__main__':
    import pystac
    loader = STACPatchLoader(
        stac_items=pystac.Catalog.from_file("/home/marian/notebooks/wc2021/train/catalog.json"),
        assets=["B02", "B03", "B04", "B08", "PATCHES"],
        patch_source_asset={
            "asset": "PATCHES",
            "shape": (256, 256),
            "stride": 256
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

    merged_rgbnir = PatchLoaderAssetView(loader,
                                         rgbnir=Concat(["B02", "B03", "B04", "B08"],
                                                       axis=2)
                                         )
    prepared = PatchLoaderAssetView(merged_rgbnir,
                                    input_1="rgbnir",
                                    output_1="rgbnir")

    def deteriorate(inp):
        if isinstance(inp, BaseRasterTile):
            inp = inp.get_value()
        width = inp.shape[0]
        height = inp.shape[1]
        patch_width = random.randint(1, width / 4)
        patch_height = random.randint(1, height/4)
        start_w = random.randint(0, width-patch_width)
        end_w = start_w + patch_width
        start_h = random.randint(0, height-patch_height)
        end_h = start_h + patch_height

        inp[start_w:end_w, start_h:end_h] = 0
        return inp
    augmented = Apply(prepared, input_1=deteriorate)


    dataset = Dataset(augmented,
                      inputs=[
                          "input_1"
                      ],
                      outputs=[
                          "output_1"
                      ]

                      )

    tf_dataset = TFLoader(dataset=dataset, batch_size=14)


    model = KerasModel(
        name="keras_model_1",
        destination="/tmp/model-marian-test",
        model_builder=UNet(
            name="foo",
            batch_normalization=True
        ),
        swap_axes=True,
        random_seed=1993,
        model_builder_options={
            "output_channels": 4
        },
        epochs=100,
        metrics=[
            "categorical_accuracy"
        ],
        loss="mse",
        use_multiprocessing=False,
        workers=1,
        max_queue_size=40,
        optimizer=tf.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        callbacks=[
        ]
    )
    trainer = RasterScenePatchTrainer(model=model,
                                      base_directory="/tmp/marian-temp-model")
    trainer.train(tf_dataset)
    # print (trainer)
    # #model.fit_generator(tf_dataset)
