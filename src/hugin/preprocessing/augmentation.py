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

import importlib
import sys
from logging import getLogger

import keras.preprocessing.image as prep
import numpy as np
import random
import yaml
from imgaug import augmenters as iaa

log = getLogger(__name__)
has_opencv = True
try:
    import cv2
except ImportError:
    has_opencv = False
    log.warn("Disabling augmentation as cv2 is missing!")


class Augmentation(object):
    def __init__(self, operators=None, random_order=False):
        self.operators = operators
        self.random_order = random_order
        self.mod_name = 'imgaug.augmenters'

    def __call__(self, input, gti=None):
        return self.augment(input=input, gti=gti)

    def augment(self, input, gti=None):
        seq = self._sequencerv2()
        if isinstance(input, dict):
            seq_det = seq.to_deterministic()
            aug = {}
            if isinstance(gti, dict):
                in_aug = {}
                out_aug = {}
                for k, v in input.items():
                    input_aug = seq_det.augment_image(v)
                    in_aug[k] = input_aug
                for k, v in gti.items():
                    gti_aug = seq_det.augment_image(v)
                    if np.amax(gti_aug) > 1:
                        gti_aug[gti_aug > 1] = 1
                    out_aug[k] = gti_aug
                return in_aug, out_aug
            else:
                for k, v in input.items():
                    if len(v.shape) == 2:
                        segmap_aug = seq_det.augment_image(v)
                        if np.amax(segmap_aug) > 1:
                            segmap_aug[segmap_aug > 1] = 1
                        aug[k] = segmap_aug
                    else:
                        aug[k] = seq_det.augment_image(v)
        elif isinstance(input, list):
            aug = []
            seq_det = seq.to_deterministic()
            for im in input:
                print(im.shape)
                if len(im.shape) == 2:
                    seq_aug = seq_det.augment_image(im)
                    if np.argmax(seq_aug) > 1:
                        seq_aug[seq_aug > 1] = 1
                    aug.append(seq_aug)
                else:
                    aug.append(seq_det.augment_image(im))
        else:
            seq_det = seq.to_deterministic()
            aug_img = seq_det.augment_image(input)
            aug_gti = seq_det.augment_image(gti)
            aug_gti[aug_gti > 1] = 1
            aug = (aug_img, aug_gti)
        return aug

    def _sequencer(self):
        op = self.operators
        operators = []
        if 'Fliplr' in op:
            operators.append(iaa.Fliplr(op.get('Fliplr', 0.25)))
        if 'Flipud' in op:
            operators.append(iaa.Flipud(op.get('Flipud', 0.25)))
        if 'Dropout' in op:
            operators.append(iaa.Sometimes(op['Dropout'].get('prop', 0.25),
                                           iaa.Dropout(op['Dropout'].get('p', [0.05, 0.2])),
                                           deterministic=True))
        if 'Sharpen' in op:
            operators.append(iaa.Sometimes(op['Sharpen'].get('prop', 0.25),
                                           iaa.Sharpen(op['Sharpen'].get('alpha', [0.0, 1.0])),
                                           deterministic=True))
        if 'Crop' in op:
            operators.append(iaa.Sometimes(op['Crop']['prop'],
                                           iaa.Crop(percent=op['Crop']['percent']),
                                           deterministic=True))
        if 'CropAndPad' in op:
            operators.append(iaa.Sometimes(op['CropAndPad']['prop'],
                                           iaa.CropAndPad(percent=op['CropAndPad']['percent'],
                                                          pad_mode=op['CropAndPad']['pad_mode']),
                                           deterministic=True))
        if 'CoarseDropout' in op:
            operators.append(iaa.Sometimes(op['CourseDropout']['prop'],
                                           iaa.CoarseDropout(op['CourseDropout']['p'],
                                                             op['CourseDropout']['size_percent']),
                                           deterministic=True))
        if 'Affine' in op:
            operators.append(
                iaa.Sometimes(
                    op['Affine']['p'],
                    iaa.Affine(
                        scale=op['Affine']['scale'],
                        rotate=op['Affine']['rotate'],
                        translate_percent=op['Affine']['translate_percent'],
                        shear=op['Affine']['shear']
                    ), deterministic=True))
        if 'ElasticTransformation' in op:
            operators.append(iaa.Sometimes(op['ElasticTransformation']['p'],
                                           iaa.ElasticTransformation(
                                               alpha=op['ElasticTransformation']['alpha'],
                                               sigma=op['ElasticTransformation']['sigma']
                                           ), deterministic=True))
        if 'GaussianBlur' in op:
            operators.append(iaa.Sometimes(op['GaussianBlur']['p'],
                                           iaa.GaussianBlur(sigma=op['GaussianBlur']['sigma']),
                                           deterministic=True))
        if 'Multiply' in op:
            operators.append(iaa.Sometimes(op['Multiply']['p'],
                                           iaa.Multiply(op['Multiply']['percent'],
                                                        per_channel=op['Multiply']['per_channel']),
                                           deterministic=True))

        seq = iaa.Sequential(operators, random_order=self.random_order)

        return seq

    def _sequencerv2(self):
        op = self.operators
        mod = importlib.import_module(self.mod_name)
        operators = []
        for k, v in op.items():
            operator = getattr(mod, k)
            if isinstance(v, float):
                try:
                    operators.append(operator(v))
                except Exception as inst:
                    log.error("Simple operator {} failed to instantiate with {} and {}".format(k, type(inst),
                                                                                               inst.args))
                    sys.exit()
            elif 'prop' in v:
                prop = v.pop('prop', None)
                some = getattr(mod, 'Sometimes')
                try:
                    operators.append(some(prop, operator(**v), deterministic=True))
                except Exception as inst:
                    log.error("Probabilistic operator {} failed to instantiate with {} and {}".format(k,
                                                                                                      type(inst),
                                                                                                      inst.args))
                    sys.exit()
            else:
                try:
                    operators.append(operator(**v))
                except Exception as inst:
                    log.error("Operator {} failed to instantiate with {} and {}".format(k, type(inst), inst.args))
                    sys.exit()
        seq = iaa.Sequential(operators, random_order=self.random_order)
        return seq

    def legacy_aug(X,
                   y,
                   horizontal_flip=False,
                   vertical_flip=False,
                   rotate=False,
                   shift=False,
                   zoom=False,
                   laplace=False):
        '''
        :param X: input image
        :param y: ground truth
        :param horizontal_flip: flip image and ground truth horizontally
                                - False -> skip
                                - probability in float (i.e. 0.5)
        :param vertical_flip: flip image and ground truth vertically
                              - False -> skip
                              - probability in float (i.e. 0.5)
        :param rotate: roatate image
                       - False -> skip
                       - {"prob": 0.5, "angle": 45}
        :param shift: shift image to location by columns, rows
                       - False -> skip
                       - {"prob": 0.05, "rcol": 0.1, "rrow": 0.1}
        :param zoom: zoom image to a certain range
                     - False -> skip
                     - {"prob": 0.05, "zoom_rg": (1, 1)}
        :param laplace: apply Laplacian gradient filter with definable kernel size
                        - False -> skip
                        - {"prob": 0.05, "ksize": 7}
        :return: tuple with transformed input and ground truth
        '''

        if not has_opencv:
            return X, y

        def _horizontal_flip(img, ground_truth, u=0.5, v=1.0):
            if v < u:
                img = cv2.flip(img, 1)
                ground_truth = cv2.flip(ground_truth, 1)

            return img, ground_truth

        def _vertical_flip(img, ground_truth, u=0.5, v=1.0):
            if v < u:
                img = cv2.flip(img, 0)
                ground_truth = cv2.flip(ground_truth, 0)

            return img, ground_truth

        def _rotate90(img, ground_truth, u=0.5, v=1.0):
            if v < u:
                angle = 90
                img_rows, img_cols = img.shape
                img_m = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), angle, 1)
                img = cv2.warpAffine(img, img_m, (img_cols, img_rows))
                ground_truth_rows, ground_truth_cols = ground_truth.shape
                ground_truth_m = cv2.getRotationMatrix2D((ground_truth_cols / 2, ground_truth_rows / 2), angle, 1)
                ground_truth = cv2.warpAffine(ground_truth, ground_truth_m, (ground_truth_cols, ground_truth_rows))
            return img, ground_truth

        def _rotate(img, ground_truth, angle=45, u=0.5, v=1.0):
            if v < u:
                # test = "augment image shape: {}".format(img.shape)
                # log.info(test)
                img_rows, img_cols, channels = img.shape
                img_m = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), angle, 1)
                img = cv2.warpAffine(img, img_m, (img_cols, img_rows))

                ground_truth_rows, ground_truth_cols, channels_gt = ground_truth.shape
                ground_truth_m = cv2.getRotationMatrix2D((ground_truth_cols / 2, ground_truth_rows / 2), angle, 1)
                ground_truth = cv2.warpAffine(ground_truth, ground_truth_m, (ground_truth_cols, ground_truth_rows))

            return img, ground_truth

        def _shift(img, ground_truth, rcol=0.1, rrow=0.1, u=0.5, v=1.0):
            if v < u:
                img_rows, img_cols = img.shape
                img_m = np.float32([[1, 0, img_rows * rrow], [0, 1, img_cols * rcol]])
                img = cv2.warpAffine(img, img_m, (img_cols, img_rows))

                ground_truth_rows, ground_truth_cols = ground_truth.shape
                ground_truth_m = np.float32([[1, 0, ground_truth_rows * rrow], [0, 1, ground_truth_cols * rcol]])
                ground_truth = cv2.warpAffine(ground_truth, ground_truth_m, (ground_truth_cols, ground_truth_rows))
            return img, ground_truth

        def _laplace(img, ground_truth, ksize=3, u=0.5, v=1.0):
            if v < u:
                img = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
                ground_truth = cv2.Laplacian(ground_truth, cv2.CV_64F, ksize=ksize)

            return img, ground_truth

        def _zoom(img, ground_truth, zoom_rg=(0.1, 0.1), u=0.5, v=1.0):
            # Current version of zoom re-sizes image. TODO add padding
            if v < u:
                img = prep.random_zoom(img, zoom_range=zoom_rg,
                                       row_axis=0, col_axis=1, channel_axis=2)
                ground_truth = prep.random_zoom(ground_truth, zoom_range=zoom_rg,
                                                row_axis=0, col_axis=1, channel_axis=2)
            return img, ground_truth

        if rotate:
            angle = rotate["angle"] * random.sample([0, 1, 2, 3, 4], 1)[0]  ### Hack
            X, y = _rotate(X, y, angle=angle, u=rotate["prob"], v=np.random.random())

        if shift:
            X, y = _shift(X, y, u=shift["prob"], v=np.random.random())

        if zoom:
            X, y = _zoom(X, y, zoom_rg=zoom["zoom_rg"], u=zoom["prob"], v=np.random.random())

        if laplace:
            X, y = _laplace(X, y, ksize=laplace["ksize"], u=laplace["prob"], v=np.random.random())

        if horizontal_flip:
            X, y = _horizontal_flip(X, y, u=horizontal_flip, v=np.random.random())

        if vertical_flip:
            X, y = _vertical_flip(X, y, u=vertical_flip, v=np.random.random())

        return (X, y)


if __name__ == '__main__':
    example_cfg_loc = "/Users/Gabriel/Documents/workspaces/hugin/etc/aug_test.yaml"

    with open(example_cfg_loc, 'r') as f:
        ensemble_config = yaml.load(f)

    print(ensemble_config.keys())
    # print(ensemble_config['model'])
    # print(ensemble_config['model'].get('model_builders', []))
    # print(ensemble_config['augment']['custom'])
    print(ensemble_config['augment']['operators'])
    aug = Augmentation(ensemble_config)
    test = aug._sequencerv2()
    print(len(test))
    # print(type(test))
