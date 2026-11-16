Configuration
=============

Event-LAB reads its main settings from ``config.yaml`` in the project root.
Recordings are streamed through EventCV. Event-LAB does not create persistent
frame or place stores. Image-only upstream methods use temporary PNGs, cleaned
up after the run (including failed runs).

Runtime settings
----------------

``data_path``
    Directory containing downloaded dataset recordings and ground truth.

``request_input``
    Ask before downloading in the CLI.

``gps_tolerance``
    Non-negative distance in **metres** along the reference route around the
    matched place. Defaults to 25.0; fractional metres are accepted. Datasets
    without metric poses (QCR) use route-landmark matches without dilation.
    Ground truth is regenerated for the active time window and tolerance.

``timewindows``
    Positive integer durations in milliseconds. Methods run once per duration;
    ensemble combines the listed durations in a single run.

``representation``
    One of ``polarity`` (default), ``binary``, ``count``, ``voxel``, ``tsurf``,
    ``atsurf``, ``tencode``, ``redblue``, ``countmask``, ``mcts``, or ``flow``.
    The editor shows a dropdown arrow; Enter or a click
    opens the list, then use Up/Down and Enter to select. Reconstruction appears
    greyed out and cannot be selected until EventCV supports named methods.
    Sparse point sets and connected-component labels are not dense stream representations.

``representation_options``
    **EventCV representation parameters**, passed to
    ``reader.with_repr(representation, **representation_options)``. See the
    `reader.with_repr() documentation
    <https://docs.eventcv.net/en/latest/api.html#eventcv.EventReader.with_repr>`_.
    Leave ``{}`` to use EventCV's
    defaults; the editor shows this as “EventCV defaults (no overrides)”.
    For voxel, ``{bins: 5}`` divides each time window into five smaller time bins.
    For polarity, ``{normalize: false}`` keeps raw event counts. For time surfaces,
    ``{tau_ms: 50}`` makes older events fade over 50 milliseconds. The editor
    explains the setting for the currently selected representation before editing.
    Options must match that representation; EventCV reports invalid options.

``hot_pixel_filter``
    Apply EventCV's recording-level hot-pixel filtering before representation
    generation. Defaults to true.

``filter_places_sec``
    Minimum separation in seconds for the LENS, ensemble and VPR-methods image
    adapters. Zero retains every window. Direct descriptor adapters retain every
    window. This is temporal sampling, not a ground truth distance tolerance.

``reconstruction``
    Disabled for now. The editor shows this in grey and rejects edits; the CLI
    and runner also reject reconstruction. Named reconstruction methods and
    their defaults will be incorporated after EventCV supports them.

Pretrained networks require fixed input channels. Adapters average channel groups
when reducing channels and repeat channels when expanding them. Image consumers
receive uint8 grayscale/RGB; SpikeVPR receives two channels (resized to its trained
346×260 input), and Event-GeM receives ten channels. These conversions make all
representations usable; they do not guarantee the accuracy of the training
representation. Descriptor/result cache paths include rendering settings.

Example
-------

.. code-block:: yaml

   gps_tolerance: 25.0
   filter_places_sec: 0
   timewindows: [50, 100, 200]
   representation: polarity
   representation_options: {}
   hot_pixel_filter: true
   reconstruction: {} # Disabled pending named EventCV methods

Terminal workbench
------------------

Launch ``pixi run tui`` to inspect and edit runtime or baseline-specific settings.
The home screen includes the Event-LAB Team, repository, and reference paper.
Home is the single entry point: choose Run an experiment, Manage downloads,
Edit configuration, View activity, or View results with Up/Down and Enter,
a mouse click, or 1–5.
Escape or the Home button returns here without clearing your selections.
The centered menu and animated ASCII wordmark share the workspaces' coral/cyan
theme. Logo animation pauses while you work in another screen.

Within experiments, Left/Right cycles only through Methods, Datasets, and
Trajectories. Within downloads, it cycles between baseline assets and recordings.
Up/Down selects rows. Tab/Shift+Tab reaches action buttons and other controls;
mouse clicks and scrolling are also supported. The batch filename is changed
through File…, keeping text input out of the panel navigation loop.
The configuration table shows one setting per row. Enter opens its editor,
Enter accepts, and Escape cancels. Space toggles boolean settings. S saves, U reloads, V reviews experiments, H returns home, and X quits.
R and Q assign trajectory roles and automatically select the dataset. Help appears below the
selected setting rather than taking space beside every field.
The dataset folder (``data_path``) is saved immediately when Enter accepts its
editor, and download availability is refreshed. Other pending settings still
require Save settings or S.
The review dialog focuses Run immediately: Enter starts the queue, Tab then
Enter selects Back, and Escape cancels.
Starting a queue opens Activity automatically. TUI logs use a compact banner.

Space assigns the first trajectory as R. When a reference already exists,
Space assigns an unused trajectory as Q; another press promotes it to R and
clears the previous reference. Pressing Space on R clears it. The R and Q keys
assign roles directly, and Delete excludes a trajectory.
Each selection panel has a Select all / Unselect all button; A performs the same
action in the focused list. For trajectories, Select all keeps the current
reference (or uses the first trajectory) and assigns every other trajectory as
a query. Unselect all clears every role. Bulk controls are disabled for imported
batches containing overrides, just like the individual selections.

View results lists saved experiment matrices from ``output/``. Select a result
with Up/Down, choose Recall@K, Similarity matrix, or PR curve from the view selector,
and press Enter on the experiment to focus the scrollable preview. Tab reaches
the view selector, Open plot, and Refresh. Open plot opens a full-size PNG in the
default browser; generated figures are saved under the experiment's ``previews/``
directory, while existing matrix overlays are reused. Terminal matrix previews
are sampled to keep large results responsive; evaluation is never rerun.

New evaluations store metrics and PR points in ``results.json`` beside the
matrix. Historical ``eventlab_results.xlsx`` data is matched only when baseline,
trajectory pair, matrix name and dimensions identify a single output. Ambiguous
or unavailable metrics are labelled as such; the matrix remains viewable.

In Experiments, Space or a mouse click toggles inclusion: ``[ ]`` excludes,
``[✓]`` includes. Highlight a dataset to see its trajectories. Each trajectory
has one role: blank (excluded), ``R`` (reference), or ``Q`` (query). Press R or Q
to assign directly, Delete to clear, or Space/click to cycle blank → R → Q → blank.
Assigning a new reference clears the previous reference; any number of queries
is allowed. New selections start with no reference. Dataset exclusion retains
its draft roles but omits it from the batch.

Save writes selections to batch YAML; Load restores them. Complex batches with
legacy overrides or multiple references per dataset remain read-only in the
selection interface. New starts a fresh editable selection.

New batches use the current saved runtime and baseline settings when started.
The queue keeps those settings stable for its duration, continues after failed
experiments, and stores output in ``logs/terminal-*/``. Existing baseline result
locations under ``output/`` are unchanged. Cancel stops the active process tree
and remaining jobs. Existing batch files with configuration overrides and method
sweeps still work; overrides are applied through temporary configuration files.
``pixi run batch`` runs ``batch_config.yaml``; append another path to run a saved
batch. Its exit status is nonzero if any experiment fails or is cancelled.

Downloads has independent baseline and dataset-sequence checkboxes. It fetches
repositories, required submodules, configured model weights, raw recordings, and
ground truth without generating frames or running inference. Some upstream weight
loaders construct a model to fill their cache, but do not perform a forward pass.
Existing assets are reused; interrupted transfers remain in ``.part`` files.
Downloads groups recordings by dataset. Space or a click on a dataset selects
all its recordings; Right enters the recording list for individual selection,
and Left returns to the dataset list. Available/total recording counts appear beside each dataset, independently of selection.
All downloads start unchecked, including missing recordings.

A background scan checks the configured ``data_path`` for each recording and its
required ground-truth and clock-sync files. Green indicates complete availability, orange indicates partial availability,
and red indicates no available files. Individual incomplete recordings are orange. Empty files and ``.part`` files do not count
as complete. The active folder is shown below the lists. Saved ``config.yaml``
changes, including external edits, automatically trigger a new scan. Check files
refreshes availability without selecting missing recordings. Manual selections
persist across datasets and visits; newly completed downloads are unchecked.
This is a local file-presence check without network requests or integrity hashing.

Local-only datasets with no download URL must be staged manually. Existing Torch
Hub trust settings are respected. Downloads can also be invoked directly::

   pixi run prepare baseline eventvlad
   pixi run prepare dataset brisbane_event sunset2 sunrise

The baseline panel also shows available/required local asset counts and uses
green, orange, and red for complete, partial, and missing installations.
Checks cover the repository, required external repositories/submodules, and
known configured weight files. No model is loaded and no update is fetched during
this scan. For VPR methods whose model cache cannot be verified locally, the
entry remains orange; the tooltip explains missing or unverified assets.
Checkboxes independently select methods to download or update.

Before running or preparing a baseline, Event-LAB fetches its configured Git
upstream and fast-forwards the existing external checkout under ``baselines/``.
The Activity log (or CLI output) reports the old/new commit, already-current
status, or why an update was skipped. Required submodules are synchronised to
the revisions pinned by the baseline, rather than following their latest branches
independently. This also applies to separately cloned NetVLAD dependencies.

Local edits, local commits, divergent branches, detached/pinned checkouts, and
branches without an upstream are preserved. The updater does not stash, reset,
or create merge commits. Untracked or ignored files that would be overwritten
prevent the update. If the upstream cannot be reached within 60 seconds, an
existing checkout remains usable and the failed check is reported. Missing
repositories or required dependencies still need an initial download.

Editing runtime settings from the CLI
-------------------------------------

The filename is ``config.yaml``. Running ``pixi run config`` displays its contents.
Options use the exact YAML key names, including underscores::

   pixi run config --timewindows 100 200 300
   pixi run config --representation voxel --representation_options '{bins: 5}'
   pixi run config --gps_tolerance 3.5 --filter_places_sec 0.5
   pixi run config --help

Booleans accept ``true`` or ``false``. Numeric lists require positive integers. Invalid updates leave the file
unchanged. ``--config PATH`` targets a different YAML file. The CLI edits runtime
settings; use the TUI for baseline settings and experiment selections.

The CLI and TUI share the same runtime fields. Retired pipeline settings are
rejected by the runner with a migration message rather than silently ignored.
