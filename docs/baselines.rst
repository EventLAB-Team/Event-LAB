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
     - Compares several temporal windows; frames are rendered with EventCV.
   * - LENS
     - ``lens``
     - Runs the LENS baseline with its own baseline config.
   * - Sparse-Event-VPR
     - ``sparse_event``
     - Samples sparse event pixels from generated frames.
   * - VPR-Methods
     - ``vprmethods``
     - Runs image VPR methods over EventCV-rendered event frames.
   * - SpikeVPR
     - ``spikevpr``
     - Spiking SEW-ResNet + MixVPR over ON/OFF event frames. Brisbane resolution only.
   * - MegaEvent
     - ``megaevent``
     - DINOv2 + SALAD over EventCV ``redblue`` frames.
   * - Event-GeM
     - ``eventgem``
     - SuperEvent features with homography re-ranking; scores shortlist and re-ranked.

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

Metrics
-------

Recall@K and the precision-recall curve are computed by ``utils.metrics``. Both
are evaluated on a matrix shaped as reference places by query places; queries
with no ground-truth match are excluded, and the ground truth is
nearest-neighbour resized when its shape differs from the matrix.

Recall@K is derived without sorting. A correct match is inside the top K exactly
when fewer than K references score strictly better than it, so every K is
obtained from one pass of two reductions. Query columns are evaluated
independently, which makes the result identical for any chunk width or worker
count, and lets the work be split across cores automatically.

Ties
~~~~

Event-based similarity matrices are often heavily quantised, so a correct match
is frequently tied with incorrect references. ``tie_policy`` makes the choice
explicit:

``optimistic`` (default)
    A match tied with incorrect references counts as retrieved.

``pessimistic``
    A match tied with incorrect references counts as missed.

The two bracket every value a tie-breaking rule could produce;
``utils.metrics.recall_at_k_bracket`` returns both from the same data. Where a
matrix has no ties the policies agree exactly.

Earlier versions delegated this to ``stschubert/VPR_Tutorial``, which resolved
ties through an unstable sort. Reported recall was therefore arbitrary among the
tied candidates and could differ between runs; results for matrices containing
ties are not directly comparable with those produced before this change.

Worker count
~~~~~~~~~~~~

The number of workers is detected rather than configured, in order:
``EVENTLAB_WORKERS``, ``SLURM_CPUS_PER_TASK``, the process CPU affinity mask on
Linux, the performance-core count on macOS, then ``os.cpu_count()``. Set
``EVENTLAB_WORKERS`` to override.

Verification
~~~~~~~~~~~~

.. code-block:: console

   pixi run python utils/verify_metrics.py

Checks that recall is invariant to chunking and worker count, that the tie
policies agree exactly on tie-free matrices, and that precision/recall matches
the previous implementation.
