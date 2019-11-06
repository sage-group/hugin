from setuptools import find_packages
from setuptools import setup
import os
import sys


REQUIRED_PACKAGES = [
    'Keras',
    'geojson',
    'rasterio',
    'scikit-image',
    'scipy',
    'geopandas',
    'Shapely',
    'Fiona',
    'h5py',
    'backoff',
    'matplotlib',
    'scikit-learn',
]

def extra_files(directory):
    datafiles = [(d, [os.path.join(d, f) for f in files])
                 for d, folders, files in os.walk(directory)]
    return datafiles

setup(
    name='hugin',
    version='0.1.0',
    install_requires=REQUIRED_PACKAGES,
    data_files=extra_files("etc/"),
    package_dir={'': 'src'},
    packages=find_packages("src"),
    include_package_data=True,
    zip_safe=False,
    entry_points = {
        'console_scripts': ['hugin=hugin.tools.cli:main'],
    },
    description='Hugin ML4EO experimentation tool',
    extras_require = {
        'hugin_data_augmentation': ["imgaug"]
    }
)