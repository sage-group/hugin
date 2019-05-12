# -*- coding: utf-8 -*-
__license__ = \
    """Copyright 2019 West University of Timisoara
    
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

from logging import getLogger

import os
import tempfile
from keras.callbacks import CSVLogger as KerasCSVLogger
from keras.callbacks import ModelCheckpoint as KerasModelCheckpoint
from tensorflow.python.lib.io import file_io

log = getLogger(__name__)


class CSVLogger(KerasCSVLogger):
    def __init__(self, *args, **kws):
        KerasCSVLogger.__init__(self, *args, **kws)
        if self.filename.startswith("gs://"):
            self.on_train_begin = self._gcp_on_train_begin

    def _gcp_on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists(self.filename):
                with file_io.FileIO(self.filename, "r" + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"

        self.csv_file = file_io.FileIO(self.filename, mode + self.file_flags)


class ModelCheckpoint(KerasModelCheckpoint):
    def __init__(self, *args, **kws):
        KerasModelCheckpoint.__init__(self, *args, **kws)
        if self.filepath.startswith("gs://"):
            self.on_epoch_end = self._gcp_on_epoch_end
            self._original_filepath = self.filepath
            self._temp_file = tempfile.NamedTemporaryFile()
            self.filepath = self._temp_file.name

    def _gcp_on_epoch_end(self, epoch, logs=None):
        # Call original checkpoint to temporary file
        KerasModelCheckpoint.on_epoch_end(self, epoch, logs=logs)

        logs = logs or {}

        # Check if file exists and not empty
        if not os.path.exists(self.filepath):
            log.warning("Checkpoint file does not seem to exists. Ignoring")
            return

        if os.path.getsize(self.filepath) == 0:
            log.warning("File empty, no checkpoint has been saved")
            return

        final_path = self._original_filepath.format(epoch=epoch + 1, **logs)

        with file_io.FileIO(self.filepath, mode='rb') as input_f:
            with file_io.FileIO(final_path, mode='w+b') as output_f:
                output_f.write(input_f.read())

        # Remove local model
        os.remove(self.filepath)
