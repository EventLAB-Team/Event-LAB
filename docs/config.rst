Configuration
=============

Event-LAB reads its main settings from ``config.yaml`` in the project root.
Most users only need to change the frame generation and evaluation settings
shown below.

Evaluation settings
-------------------

``ground_truth_tolerance``
    Distance in metres used when building the ground truth tolerance around a
    matched reference place. A value of ``3`` means a prediction within 3 m of
    the ground truth match is counted as correct.

``filter_places_sec``
    Minimum time in seconds between frames used as places. Set this to ``0.0``
    to keep every generated frame. Increase it when you want fewer, more
    separated places along the route.

Frame generation settings
-------------------------

``timewindows``
    List of time windows, in milliseconds, used when generating frames by time.
    Event-LAB runs the baseline once for each value.

``num_events``
    List of maximum event counts per frame. This is only used when the selected
    frame generation mode is event-count based.

``frame_generator``
    Controls the type of frame data prepared for the baseline.

    ``frames``
        Build accumulated event frames directly from the event stream.

    ``reconstruction``
        Build image-like reconstructed frames using the selected reconstruction
        model.

``frame_accumulator``
    Controls how events are accumulated into direct event frames.

    ``eventcount``
        Accumulate by event count.

    ``polarity``
        Keep polarity-aware event information.

``reconstruction_model``
    Reconstruction model used when ``frame_generator`` is ``reconstruction``.
    Current options are ``e2vid`` and ``firenet``.

Example
-------

.. code-block:: yaml

   ground_truth_tolerance: 3
   filter_places_sec: 0.0

   timewindows: [250, 500, 750, 1000]
   num_events: [100000]
   frame_generator: "reconstruction"
   frame_accumulator: "polarity"
   reconstruction_model: "e2vid"

With this configuration Event-LAB will generate reconstructed frames at four
time windows and run the selected baseline once for each window.
