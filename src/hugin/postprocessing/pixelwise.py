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

import numpy as np
from skimage import morphology
from skimage import restoration
from skimage.morphology import disk, binary_erosion, binary_dilation

log = getLogger(__name__)


class DummyProcessing(object):
    def __init__(self, argc=None):
        self.argc = argc

    def __call__(self, data, probabilities):
        return data, probabilities


class RemoveSmallObjects(object):
    def __init__(self, threshold=100):
        self.pixel_threshold = threshold

    def __call__(self, data, probs):
        log.debug("Removing small objects with threshold=%d", self.pixel_threshold)
        return morphology.remove_small_objects(data, self.pixel_threshold), probs


class DenoiseTVChambolle(object):
    def __init__(self, weight=1, threshold=0.4):
        self.weight = weight
        self.threshold = float(threshold)

    def __call__(self, data, probs):
        log.debug("Doing denoise_tv_chambolle with weight:%d and threshold:%f", self.weight, self.threshold)
        clean = restoration.denoise_tv_chambolle(data, weight=1)
        log.debug("Converting to binary according to threshold")
        clean = clean > self.threshold
        return clean, probs


class ZeroNoData(object):
    def __init__(self, value, input_name):
        self._value = value
        self._input_name = input_name

    def __call__(self, data, probs, scene_data):
        nodata = self._value
        rio_data = scene_data[self._input_name]
        if nodata is None:
            nodata = rio_data.nodata
        msk = rio_data.read_masks()
        shp = msk.shape
        bands = shp[0]
        for i in range(0, bands):
            arr = msk[i]
            data[np.where(arr == nodata)] = 0
        return data, probs


class MaskExpand(object):
    def __init__(self, iter=1):
        self.iter = iter

    def __call__(self, data, probs, scene_data):
        log.debug("Expanding binary mask by %d pixels...", self.iter)
        ylen, xlen = data.shape
        output = data.copy()
        for iter in range(self.iter):
            for y in range(ylen):
                for x in range(xlen):
                    if (y > 0 and data[y - 1, x]) or (y < ylen - 1 and data[y + 1, x]) or \
                            (x > 0 and data[y, x - 1]) or (x < xlen - 1 and data[y, x + 1]):
                        output[y, x] = 1
            data = output.copy()
        return output, probs


class Erode(object):
    def __init__(self, size=1):
        self.size = size

    def __call__(self, data, probs, scene_data):
        selem = disk(self.size)
        eroded = binary_erosion(data, selem)
        return eroded, probs


class Dilate(object):
    def __init__(self, size=1):
        self.size = size

    def __call__(self, data, probs, scene_data):
        selem = disk(self.size)
        dilated = binary_dilation(data, selem)
        return dilated, probs
