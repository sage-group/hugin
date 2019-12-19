Spacenet Roads Example
----------------------

This example illustrates easy one can generate dynamic data types (in this case the Ground Truth Images for training)
and use them later in data mapping.

Creation of the dynamic types in Hugin is based on the `type_format` of the FileSystemLoader. Therefore
the naming convention for the spacenet geojson files has been altered (e.g. from
spacenetroads_AOI_2_Vegas_img59.geojson to spacenetroads_AOI_2_Vegas_GT_img59.geojson). This is done
purposefully for making regex matching of those files easier.


Training
~~~~~~~~

.. literalinclude:: ../../etc/spacenet_roads_example/train_spacenet.yaml
   :language: yaml
   :linenos:
