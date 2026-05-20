Datasets
========

Event-LAB datasets are selected by name from the command line:

.. code-block:: bash

   pixi run eventlab <baseline> <dataset> <reference> <query>

For example:

.. code-block:: bash

   pixi run eventlab sparse_event brisbane_event sunset2 sunrise

This loads ``datasets/brisbane_event.yaml``, downloads the ``sunset2`` and
``sunrise`` sequences if needed, formats them into Event-LAB's standard HDF5
format, generates frames using ``config.yaml``, and prepares the ground truth
matrix for evaluation.

Available datasets
------------------

The dataset name is the YAML filename in ``datasets/`` without the extension.
The sequence names are the keys under ``sequences`` in that YAML file.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Command name
     - Example sequences
   * - Brisbane-Event-VPR
     - ``brisbane_event``
     - ``sunset1``, ``sunset2``, ``sunrise``
   * - NSAVP
     - ``nsavp``
     - ``R0_FA0``, ``R0_FS0``
   * - Fast-and-Slow
     - ``fast_slow``
     - ``r_med1``, ``q_med1``
   * - QCR-Event-VPR
     - ``qcr_event``
     - ``normal1``, ``fast1``, ``slow1``

Download data only
------------------

To download and prepare the datasets listed in ``config.yaml`` without running
a baseline:

.. code-block:: bash

   pixi run getdata config.yaml

The ``datasets`` section in ``config.yaml`` controls which dataset and sequence
pairs are prepared.

Dataset YAML files
------------------

Each dataset YAML file describes:

``dataset``
    Metadata such as the dataset name, camera, resolution, and source URL.

``sequences``
    Download links for each sequence's event data, optional ground truth, and
    optional hot-pixel file.

``format``
    Input data format and timestamp scale used during conversion.

``other``
    Dataset-specific details such as sequence offsets or manual ground truth
    timing points.

The Brisbane dataset file is the best starting point when checking how a full
dataset configuration should look.
