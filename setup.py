import os

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'Keras',
    'tensorflow',
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
    'tqdm',
]


def extra_files(directory):
    datafiles = [(d, [os.path.join(d, f) for f in files])
                 for d, folders, files in os.walk(directory)]
    return datafiles


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('src/hugin/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(
    name='hugin',
    version=version,
    license='apache-2.0',
    description='HuginEO - Machine Learning for Earth Observation experimentation tool',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Marian Neagul',
    author_email='mneagul@gmail.com',
    url='https://github.com/sage-group/hugin',
    keywords=['deep learning', 'machine learning', 'earth observation'],
    install_requires=REQUIRED_PACKAGES,
    data_files=extra_files("etc/"),
    package_dir={'': 'src'},
    packages=find_packages("src"),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': ['hugin=hugin.tools.cli:main'],
    },
    extras_require={
        'hugin_data_augmentation': ["imgaug"]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.6',
)
