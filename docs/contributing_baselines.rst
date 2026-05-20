Contributing Baselines
======================

A baseline wrapper adapts an external method to Event-LAB's standard
reference/query pipeline. Start from ``baselines/baseline_template.py``, then
compare against ``baselines/sparse_event.py`` for an in-process baseline or
``baselines/eventvlad.py`` for a baseline that prepares data and model weights.

Files to add
------------

Add two files for a new baseline:

``baselines/<name>.py``
    The Python wrapper that subclasses ``EventBaseline``.

``baselines/<name>.yaml``
    Baseline-specific settings. Use
    ``baselines/baseline_config_template.yaml`` as a small starting point.

Then register the baseline in ``baselines/get_baseline.py`` and add its command
name to ``VPR-Baselines`` in ``config.yaml``.

Wrapper methods
---------------

``format_data(config, dataset_config, reference, query, timewindow)``
    Locate the generated reference and query frame stores for the current
    time window, convert or filter data as needed, and create the baseline
    output directory.

``build_execute(config, data_config, ground_truth)``
    Prepare any command, model, temporary files, or converted inputs required by
    the method.

``run()``
    Execute the method and save one or more ``.npy`` result matrices into the
    output directory.

``parse_results(GT)``
    Load the saved matrices and call the shared metric helpers. Most baselines
    can follow the pattern already used in the existing wrappers.

``cleanup()``
    Remove temporary files created by the wrapper.

Result matrices
---------------

Event-LAB evaluates two-dimensional matrices shaped as reference places by query
places. Set ``self.matrix_type`` to ``distance`` when lower values are better,
or ``similarity`` when higher values are better.

The common output directory pattern is:

.. code-block:: text

   output/<baseline>/<dataset>/<reference>_<query>/<frame_generator>_<timewindow>/

Keep the saved matrix names simple, for example ``distance_matrix.npy``.

Keep it small
-------------

Only add configuration keys that the baseline actually reads. If a setting is
shared across methods, prefer the existing ``config.yaml`` key rather than a new
baseline-specific option.
