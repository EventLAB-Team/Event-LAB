Baselines
=========

Baselines are selected by command name:

.. code-block:: bash

   pixi run eventlab <baseline> <dataset> <reference> <query>

Event-LAB downloads or prepares the selected baseline, formats the reference
and query data, runs the method, then evaluates the saved distance or similarity
matrix against the ground truth.

Available baselines
-------------------

.. list-table::
   :header-rows: 1

   * - Baseline
     - Command name
     - Notes
   * - EventVLAD
     - ``eventvlad``
     - Uses reconstructed frames and EventVLAD weights.
   * - Ensemble-Event-VPR
     - ``ensemble``
     - Runs the ensemble baseline over prepared event frames.
   * - LENS
     - ``lens``
     - Runs the LENS baseline with its own baseline config.
   * - Sparse-Event-VPR
     - ``sparse_event``
     - Samples sparse event pixels from generated frames.
   * - VPR-Methods
     - ``vprmethods``
     - Runs image VPR methods on reconstructed frames.

Baseline configuration
----------------------

Baseline-specific settings live beside each baseline wrapper, for example
``baselines/sparse_event.yaml`` and ``baselines/eventvlad.yaml``. These files
control method settings, while ``config.yaml`` controls shared Event-LAB
settings such as time windows and frame generation.

Outputs
-------

Each run writes baseline outputs under ``output/<baseline>/`` and appends the
evaluation summary to:

.. code-block:: text

   output/eventlab_results.xlsx

For each baseline run, Event-LAB expects one or more ``.npy`` matrices in the
baseline output directory. Distance matrices use lower values for better
matches; similarity matrices use higher values for better matches. The baseline
wrapper declares which type it produces before metrics are calculated.
