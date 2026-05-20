Contributing Datasets
=====================

Add a dataset when Event-LAB can download the raw event data and describe each
sequence in a YAML file. The Brisbane configuration is the clearest complete
example.

Dataset YAML
------------

Create a file in ``datasets/`` named after the command users should type. For
example, ``datasets/my_dataset.yaml`` is used as:

.. code-block:: bash

   pixi run eventlab sparse_event my_dataset reference_sequence query_sequence

The YAML should include:

``dataset``
    Name, version, description, source URL, camera, and resolution.

``sequences``
    One entry per traverse. Each sequence needs a data URL, plus ground truth
    and hot-pixel entries marked as available or unavailable.

``format``
    Input event data format, timestamp units, ground truth format, and hot-pixel
    format.

``other``
    Optional dataset-specific information such as per-sequence offsets.

Template
--------

Use ``datasets/dataset_template.yaml`` as the smallest starting point. Copy it
to a new dataset YAML and replace the names, URLs, formats, and sequence keys.

Ground truth
------------

If ground truth files are available, set ``ground_truth.available`` to ``true``
and provide the download URL. Event-LAB will download the files and build the
reference/query ground truth matrix during a run.

If a dataset does not provide ground truth files, leave the URL empty and set
``available`` to ``false``. Datasets that need manual timing information can use
the ``other`` section, as in ``datasets/qcr_event.yaml``.

Adding the dataset to defaults
------------------------------

After adding the dataset YAML, add an entry to the ``datasets`` list in
``config.yaml`` only if you want ``pixi run getdata config.yaml`` to prepare it
by default.
