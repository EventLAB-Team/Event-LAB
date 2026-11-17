Datasets
========

EventLAB - Localization datasets are selected by name from the command line:

.. code-block:: bash

   pixi run eventlab <baseline> <dataset> <reference> <query>

For example:

.. code-block:: bash

   pixi run eventlab sparse_event brisbane_event sunset2 sunrise

This loads ``datasets/brisbane_event.yaml``, downloads the ``sunset2`` and
``sunrise`` sequences if needed, formats them into EventLAB - Localization's standard HDF5
format (or a window manifest, for the pooled datasets below), generates frames using ``config.yaml``, and prepares the ground truth
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
   * - Springfield-Event-VPR
     - ``springfield_event``
     - ``database`` (reference), ``queries``, ``query_day``, ``query_dawn``
   * - NYC-Event-VPR
     - ``nyc_event``
     - ``database`` (reference), ``queries``

Pooled galleries: Springfield and NYC
-------------------------------------

These two datasets are not one recording per traverse, so their sequences are
*window manifests* (``<sequence>.windows.json``): explicit event windows drawn
from many recordings, each with its own GPS position. Every baseline reads them
through the same reader, and ground truth is the GPS distance between windows,
using ``gps_tolerance`` from ``config.yaml``. ``filter_places_sec`` does not thin
them, because every window is already a place.

.. code-block:: bash

   pixi run eventlab megaevent springfield_event database queries
   pixi run eventlab megaevent nyc_event database queries

``springfield_event`` (Hines et al., arXiv:2609.21219)
    ``database`` pools all seven database sessions. Each partition is cut into
    consecutive windows from its first event, keeping the final partial window,
    and windows without valid GPS at their centre are dropped: 132,569 windows
    at 50 ms. ``queries`` are the 5,557 windows in ``queries/query_slices.csv``
    (``query_day`` 3,169, ``query_dawn`` 2,388); other time windows keep each
    slice's centre. Positions come from ``derived/sync.json`` and
    ``derived/gps_interp.csv``. The full download is about 810 GiB.

``nyc_event`` (Pan et al., ICRA 2025)
    The raw Prophesee recordings are downloaded and extracted (about 400 GiB;
    each archive is deleted after extraction unless ``other.keep_archives`` is
    set). Every 1 Hz GPS fix inside a recording is a place, and each place is one
    window of the configured length centred on the fix. A seeded random 10% of
    places are ``queries`` and the other 90% the ``database``, following the
    dataset's benchmark split (``other.query_fraction``, ``other.seed``).

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
