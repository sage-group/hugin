from hugin.preprocessing.augmentation import Augmentation
import os
import yaml

conf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests/config'))

aug_conf = os.path.join(conf_dir, 'aug_test.yaml')

with open(aug_conf, 'r') as f:
    aug_conf_d = yaml.load(f)

print(aug_conf_d)
print(aug_conf_d['augment']['operators'])
aug = Augmentation(aug_conf_d)
test = aug._sequencerv2()
print(len(test))


img_aug = aug.augment()