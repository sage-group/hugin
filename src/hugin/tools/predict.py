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

import math
import numpy as np
from skimage import measure

log = getLogger(__name__)


def create_numbered_tiles(tile_count):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('L', (256, 256), color=0)
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    msg = "%d" % tile_count
    w, h = draw.textsize(msg, font=fnt)
    draw.rectangle([(0, 0), (256, 256)], outline=1, fill=0)
    draw.text(((256 - w) / 2, (256 - h) / 2), msg, font=fnt, fill=1)

    pic = np.array([np.zeros((256, 256)), np.array(img)])
    pic = np.swapaxes(np.swapaxes(pic, 0, 1), 1, 2)
    return pic


def get_probabilities_from_tiles(model,
                                 data_generator,
                                 rows, columns,
                                 window_width,
                                 window_height,
                                 stride_size,
                                 batch_size,
                                 merge_strategy='average', _show_numbered_tiles=False):
    assert merge_strategy in ("average", "maximum", "entropy")
    log.debug("Size: (%d, %d)", rows, columns)
    tile_array = None
    tile_freq_array = np.zeros((rows, columns), dtype=np.uint8)

    if window_width != stride_size:
        num_horizontal_tiles = math.ceil((columns - window_width) / float(stride_size) + 2)
    else:
        num_horizontal_tiles = math.ceil((columns - window_width) / float(stride_size) + 1)

    xoff = 0
    yoff = 0

    tile_count = 0

    for data in data_generator:
        in_arrays, out_arrays = data
        prediction = model.predict(in_arrays, batch_size=batch_size)
        if tile_array is None:
            tile_array = np.zeros((rows, columns, prediction.shape[3]), dtype=prediction.dtype)
        for i in range(0, in_arrays.shape[0]):
            tile_count += 1
            tile_prediction = prediction[i].reshape(
                (window_width, window_height, prediction.shape[3]))

            if _show_numbered_tiles:
                tile_prediction = create_numbered_tiles(tile_count)

            xstart = xoff * stride_size
            xend = xstart + window_height
            ystart = yoff * stride_size
            yend = ystart + window_width

            if xend > rows:
                xstart = rows - window_height
                xend = rows
            if yend > columns:
                ystart = columns - window_width
                yend = columns

            log.debug("%d (%d: %d, %d: %d)" % (tile_count, ystart, yend, xstart, xend))
            if merge_strategy == "average":
                tile_array[xstart: xend, ystart: yend, :] += tile_prediction
            elif merge_strategy == "maximum":
                tile_array[xstart: xend, ystart: yend, :] = np.maximum(tile_array[xstart: xend, ystart: yend, :],
                                                                       tile_prediction)
            elif merge_strategy == "entropy":
                # ToDo: Implement me!
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            tile_freq_array[xstart: xend, ystart: yend] += 1

            if yoff == num_horizontal_tiles - 1:
                yoff = 0
                xoff += 1
            else:
                yoff += 1

    if merge_strategy == "average":
        channels = tile_array.shape[-1]
        for i in range(0, channels):
            tile_array[:, :, i] = tile_array[:, :, i] / tile_freq_array
    elif merge_strategy in ("maximum", "entropy"):
        pass
    else:
        raise NotImplementedError()

    return tile_array


def categorical_prediction(probability_array):
    prediction = np.argmax(probability_array, -1).astype(np.uint8)
    return prediction.reshape(prediction.shape + (1,))


def noop_prediction(probability_array):
    return probability_array


def to_polygons(img):
    contours = measure.find_contours(img, 0.9)
    for contour in contours:
        coords = []
        for pairs in list(contour):
            x, y = list(pairs)
            coords.append((y, x))

        yield [coords, ]  # temporary hack to allow later hole specification
