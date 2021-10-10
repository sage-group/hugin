from hugin.preprocessing.augmentation import Augmentation
import numpy as np
import os
import yaml
from scipy.misc import imshow
from matplotlib import pyplot as plt


conf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests/config'))

aug_conf = os.path.join(conf_dir, 'aug_test.yaml')

with open(aug_conf, 'r') as f:
    aug_conf_d = yaml.load(f)

print(aug_conf_d)
print(aug_conf_d['augment']['operators'])
aug = Augmentation(aug_conf_d)
test = aug._sequencerv2()
print(len(test))

# arr = np.ones((250, 250, 4), dtype=np.uint8) * 255
arr = np.random.randint(0, 255, size=(250, 250, 4)).astype(np.uint8)
print(arr.shape)

img_aug = aug.augment(arr, arr, aug_gti=False)
print(img_aug[0].shape)
plt.imshow(img_aug[0], interpolation='nearest')
plt.show()
