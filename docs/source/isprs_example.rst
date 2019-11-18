ISPRS Dataset Example
---------------------

The following example illustrates how to use Hugin for running training and predictions on the ISPRS benchmark dataset
for 2D semantic labeling on the city of Potsdam.

This configuration will take in account only RGB input and GTI (or label) output provided in the ISPRS dataset.
The U-Net version that comes shipped with Hugin will be used for training.

Training
~~~~~~~~

.. literalinclude:: ../../etc/isprs_benchmark_example/train_isprs.yaml
   :language: yaml
   :linenos:

After this, we can simply start training our U-Net variant with Hugin by simply running:

.. code-block:: bash

  hugin train --config ./etc/usecases/train_isprs.yaml


Predictions
~~~~~~~~~~~

After training a model, running predictions is pretty straightforward with Hugin.

.. literalinclude:: ../../etc/isprs_benchmark_example/predict_isprs.yaml
   :language: yaml
   :linenos:

Then, for running the predictions you just have to specify the path to the configuration file, and the paths from
where you want to load the data and save the predictions.

.. code-block:: bash

   hugin predict --config ./etc/usecases/predict_isprs.yaml --input-dir /mnt/ISPRS/prediction/ \
   --output-dir /home/alex/potsdam_predictions