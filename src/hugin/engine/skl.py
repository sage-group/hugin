import pickle

import tqdm
import os
from hugin.engine.core import RasterModel
from logging import getLogger

log = getLogger(__name__)


class SkLearnModel(RasterModel):
    def __init__(self, model=None, model_path=None, **kwargs):
        RasterModel.__init__(self, **kwargs)
        self.model_path = model_path
        self.model = model
        if self.model_path is not None and os.path.exists(self.model_path):
            log.info(f"Loading model from {self.model_path}")
            with open(self.model_path, "rb") as model_pickle:
                self.model = pickle.load(model_pickle)
            log.info("Model Loaded")
        elif self.model is None:
            log.error("No model specified")
            raise ValueError("No model specified")

        if not hasattr(self.model, "partial_fit"):
            raise ValueError("Model does not implement the SciKit-Learn `partial_fit` API")
        log.info(f"Using {self.model}")

    def fit_generator(self, train_data, validation_data=None, *_):
        log.info("Starting fitting")
        for batch in tqdm.tqdm(train_data):
            X, y = batch
            self.model.partial_fit(X, y)
        log.info("Finished fitting")

    def save(self, destination=None):
        log.info("Saving SciKit-Learn model to %s", destination)
        destination_file = os.path.join(destination, f"{self.model_name}.pickle")
        if not os.path.exists(destination):
            os.makedirs(destination)
        log.info(f"Pickling model to {destination_file}")
        with open(destination_file, "wb") as model_pickle:
            pickle.dump(self.model, model_pickle)
            log.info(f"Pickled model to {destination_file}")
