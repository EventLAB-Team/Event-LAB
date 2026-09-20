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
    Open the reference and query recordings for the current time window, prepare
    whatever the method consumes, and create the baseline output directory.

    Events are read through EventCV, never from pre-rendered frame directories:
    ``reference.get_dataset_info()['hdf5_path']`` is the formatted recording, and
    ``ecv.open(path, dt_ms=timewindow, offset=..., repr=...)`` streams it without
    materialising it. Use ``utils.eventcv_frames.offsets_from_dataset_config`` to
    resolve the per-sequence stream offsets, and, when the method needs images on
    disk, ``utils.eventcv_frames.render_png_sequence``.

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

Baselines with conflicting dependencies
---------------------------------------

Most baselines run in the default environment. When a method needs packages that
cannot or should not sit alongside the defaults, give it its own pixi feature and
environment rather than widening the default one:

.. code-block:: toml

   [environments]
   mybaseline = {features = ["mybaseline"]}

   [feature.mybaseline.dependencies]
   some-package = "*"

Omitting ``no-default-feature`` keeps torch, numpy and eventcv shared, so only the
extra packages are solved and installed.

The wrapper then runs the method through :func:`utils.utils.pixi_run`, which
invokes ``pixi run -e <environment> bash -c ...`` and strips the inherited
``PIXI_ENVIRONMENT``/``PIXI_PROJECT_MANIFEST`` first -- without that, the nested
call can resolve straight back to the parent environment:

.. code-block:: python

   from utils.utils import pixi_run

   result = pixi_run("mybaseline", command, cwd=repo, extra_env={"PYTHONPATH": repo})
   if result.returncode != 0:
       raise RuntimeError(f"mybaseline failed with return code {result.returncode}")

Because the wrapper itself runs in the default environment, it cannot import the
method directly. Put the model-side work in a small argparse CLI under ``utils/``
that writes descriptors or matrices to disk -- see ``utils/spikevpr_descriptors.py``
and ``utils/megaevent_descriptors.py`` -- and let the wrapper load those results
and assemble the reference-by-query matrix.

Do not modify a cloned upstream repository. Where upstream needs a behavioural
change, either fork it (as ``lens`` and ``ensemble`` do) or monkeypatch from the
helper CLI, documenting why -- ``utils/eventgem_run.py`` patches a hardcoded
``num_workers`` that cannot survive macOS's spawn start method.

Caching
-------

Intermediate artefacts (descriptor banks, feature caches) must not be written into
``self.output_dir``: ``parse_results`` globs every ``*.npy`` there and would score
them as if they were result matrices. Key any cache by every parameter that
changes its contents -- at minimum the time window, and the stream offset where one
applies -- or a later run will silently reuse a stale artefact.
