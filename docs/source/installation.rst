Installation
============

Prerequisites
------------

Hugin builds functionality on top of existing technology, primarily it uses Keras, SciKit-Learn, OpenCV and RasterIO.

The exact prerequisites are specified in the `requirements.txt` and `setup.py` files. Normally you package manager will handle
requirement installation automatically.

Using pip
---------

From PyPi
~~~~~~~~~

ToDo

From GitHub
~~~~~~~~~~~~

You can install Hugin using the following command:

.. code-block:: bash

   pip install git+http://github.com/sage-group/hugin#egg=hugin

From source code
----------------

When installing from source code we recommend installation inside a specially created virtual environment.

Installing from source code involves running the `setup.py` inside you python environment.

.. code-block:: bash

   python setup.py install
