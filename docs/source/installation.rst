Installation
============

Prerequisites
------------

Hugin builds functionality on top of existing technology, primarily it uses Tensorflow Keras, SciKit-Learn, OpenCV and RasterIO.

The exact prerequisites are specified in the `requirements.txt` and `setup.py` files. Normally you package manager will handle
requirement installation automatically.

Using pip
---------

From PyPi
~~~~~~~~~

.. code-block:: bash

   pip install hugin

From GitHub
~~~~~~~~~~~~

You can install Hugin using the following command:

.. code-block:: bash

   pip install git+http://github.com/sage-group/hugin#egg=hugin

From source code
----------------

When installing from source code we recommend installation inside a specially created virtual environment.

Installing from source code involves building the package

.. code-block:: bash

   python -m build

And further instalation
.. code-block:: bash

   pip install dist/hugin-0.3.0-py3-none-any.whl
