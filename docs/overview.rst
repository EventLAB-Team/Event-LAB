Contributing Overview
=====================

Event-LAB contributions should keep the shared pipeline simple: datasets define
where data comes from, baselines define how a method runs, and ``config.yaml``
defines the shared experiment settings.

Before opening a pull request
-----------------------------

1. Run the smallest command that exercises your change.
2. Check that outputs are written under ``output/``.
3. Check that metrics are appended to ``output/eventlab_results.xlsx`` when a
   baseline is run.
4. Keep unrelated formatting and refactors out of the change.

Repository conventions
----------------------

``datasets/``
    Dataset YAML files and dataset loading helpers. A dataset YAML should be
    enough for Event-LAB to download, format, and generate frames for a
    sequence.

``baselines/``
    Baseline wrappers and baseline-specific YAML files. A wrapper should adapt
    one method to Event-LAB's reference/query evaluation flow.

``docs/``
    Read the Docs source files. Keep pages short, command-focused, and close to
    the current code behavior.

``config.yaml``
    The shared experiment configuration used by normal runs and batch runs.

Minimal test command
--------------------

Use a small known run when checking documentation or pipeline changes:

.. code-block:: bash

   pixi run eventlab sparse_event brisbane_event sunset2 sunrise

For documentation-only changes, also build the docs:

.. code-block:: bash

   pixi run sphinx-build -b html docs docs/_build/html
