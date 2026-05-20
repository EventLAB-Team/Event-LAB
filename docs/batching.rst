Batching
========

Batching runs several Event-LAB experiments from ``batch_config.yaml``. It is
useful when you want to compare several baselines, queries, and frame settings
without typing each command by hand.

Run a batch
-----------

.. code-block:: bash

   pixi run batch

This command reads ``batch_config.yaml``, updates ``config.yaml`` for each
experiment, writes ``run_batch.sh``, and runs each generated Event-LAB command
in order.

Batch configuration
-------------------

.. code-block:: yaml

   batch_experiments:
     - dataset: brisbane_event
       reference: sunset2
       queries: [sunrise]
       baselines: [sparse_event, lens, eventvlad]
       config:
         frame_generator: frames
         frame_accumulator: polarity
         timewindows:
           list: [33, 66, 120, 250]

Each experiment defines one dataset, one reference sequence, one or more query
sequences, and one or more baselines. The ``config`` block contains the
Event-LAB settings that should be applied before those runs.

Baseline overrides
------------------

Use ``baseline_config`` when a batch needs to change a baseline YAML setting for
one experiment.

.. code-block:: yaml

   baseline_config:
     vprmethods:
       method: ["cosplace", "eigenplaces", "mixvpr"]

When ``method`` is a list, Event-LAB expands it into separate runs and writes
each method into the baseline config before running that command.

Generated commands
------------------

For each baseline and query pair, batching generates commands in this form:

.. code-block:: bash

   python eventlab_run.py "<baseline>" "<dataset>" "<reference>" "<query>"

The results are saved in the same output locations as normal single-command
runs.
