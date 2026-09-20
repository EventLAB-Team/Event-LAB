"""
Extract SpikeVPR descriptors for a single Event-LAB sequence.

This runs inside the ``spikevpr`` pixi environment (the only one carrying
spikingjelly) and is invoked as a subprocess by ``baselines/spikevpr.py`` --
the same "standalone CLI wrapping an upstream model" pattern as
``utils/eventvlad_denoiser.py``.

Only SpikeVPR's model factory is imported from the clone. Its own dataset
classes are deliberately avoided: they expect a sliced-``.npy`` + NMEA layout and
drag in tonic>=1.4/geopy/pynmea2. Events come from EventCV instead, and the
descriptors are written as a plain ``(n_slices, 4096)`` float32 ``.npy`` for the
wrapper to turn into a reference x query matrix.
"""
import argparse
import os
import sys

import numpy as np
import torch
from loguru import logger

import eventcv as ecv


def resolve_device(requested):
    """cuda -> mps -> cpu, unless a specific device was asked for."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(repo_path, checkpoint, encoder, neuron_type, out_channels, out_rows, device):
    """Instantiate SpikeVPR and load a checkpoint, without pip-installing the repo."""
    src = os.path.abspath(os.path.join(repo_path, "src"))
    # append() rather than insert(): the repo ships a top-level `tools` package
    # that would otherwise shadow anything similarly named on the path.
    if src not in sys.path:
        sys.path.append(src)
    from spikevpr.models import build_spikevpr

    # A neuron_type mismatch is SILENT: spikingjelly keeps membrane potential in a
    # plain dict rather than a buffer, so its neurons contribute no state_dict keys
    # and strict=True cannot catch it. The Brisbane/NSAVP checkpoints are LIFNode,
    # NYC is IFNode -- always pass this explicitly.
    model = build_spikevpr(
        encoder=encoder,
        aggregator="mixvpr",
        out_channels=out_channels,
        out_rows=out_rows,
        neuron_type=neuron_type,
        checkpoint=None,
        device=device,
        eval_mode=True,
    )

    # factory.py calls torch.load() without weights_only=, which torch 2.12
    # defaults to True; load it here so the behaviour is explicit either way.
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning(f"weights_only=True load failed ({exc}); retrying unrestricted")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def render_frame(stream, sensor_size, event_count, channel_order):
    """
    One (2, H, W) ON/OFF event-count frame, matching SpikeVPR's training input.

    SpikeVPR was trained on tonic ToFrame(sensor_size=(W, H, 2)), which bins by
    polarity VALUE -- channel 0 is p==0 (OFF), channel 1 is p==1 (ON) -- holding
    RAW integer counts. eventcv's repr="polarity" is the opposite order and is
    rescaled to 0-255, so the two count() calls below are used instead.

    ``event_count`` truncates the window to its first N events, reproducing the
    eval transform ToFrame(event_count=15000) + FirstSingleFrame() so the network
    sees the per-pixel count distribution its BatchNorms were fitted on, while
    still yielding exactly one frame per Event-LAB time window.
    """
    if event_count:
        arr = np.asarray(stream.numpy())          # (N, 4), columns x, y, t, p
        if arr.shape[0] > event_count:
            stream = ecv.from_numpy(
                arr[:event_count], sensor_size=sensor_size, time_unit="us"
            )
    off = np.asarray(stream.filter_polarity(0).count().numpy())[0]
    on = np.asarray(stream.filter_polarity(1).count().numpy())[0]
    pair = [off, on] if channel_order == "off_on" else [on, off]
    return np.stack(pair).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Extract SpikeVPR descriptors for one sequence")
    ap.add_argument("--hdf5_path", required=True, help="Formatted Event-LAB event recording")
    ap.add_argument("--out", required=True, help="Destination .npy for the (N, D) descriptors")
    ap.add_argument("--repo_path", default="./baselines/SpikeVPR")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dt_ms", type=float, required=True)
    ap.add_argument("--offset", type=float, default=None,
                    help="Absolute timestamp in milliseconds (EventCV's bare offset unit)")
    ap.add_argument("--sensor_width", type=int, required=True)
    ap.add_argument("--sensor_height", type=int, required=True)
    ap.add_argument("--encoder", default="sew_resnet34")
    ap.add_argument("--neuron_type", default="LIFNode")
    ap.add_argument("--out_channels", type=int, default=512)
    ap.add_argument("--out_rows", type=int, default=8)
    ap.add_argument("--event_count", type=int, default=15000,
                    help="Truncate each window to its first N events; 0 uses the whole window")
    ap.add_argument("--channel_order", default="off_on", choices=["off_on", "on_off"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hot_pixel_filter", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    sensor_size = (args.sensor_width, args.sensor_height)
    device = resolve_device(args.device)
    logger.info(f"SpikeVPR descriptors: {args.hdf5_path} -> {args.out} (device={device})")

    kwargs = {"dt_ms": args.dt_ms, "hot_pixel_filter": args.hot_pixel_filter,
              "sensor_size": sensor_size}
    if args.offset is not None:
        kwargs["offset"] = args.offset
    reader = ecv.open(args.hdf5_path, **kwargs)
    n_slices = int(reader.n_slices)
    logger.info(f"{n_slices} slices at dt_ms={args.dt_ms}, event_count={args.event_count}")

    model = build_model(args.repo_path, args.checkpoint, args.encoder, args.neuron_type,
                        args.out_channels, args.out_rows, device)

    descriptors = []
    with torch.no_grad():
        for start in range(0, n_slices, args.batch_size):
            end = min(start + args.batch_size, n_slices)
            batch = np.stack([
                render_frame(reader.slice(i), sensor_size, args.event_count, args.channel_order)
                for i in range(start, end)
            ])
            x = torch.from_numpy(batch).to(device)
            descriptors.append(model(x).float().cpu().numpy())
            if start % (args.batch_size * 20) == 0:
                logger.info(f"  {end}/{n_slices}")

    out = np.concatenate(descriptors, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, out)
    logger.info(f"Wrote {out.shape} descriptors to {args.out}")


if __name__ == "__main__":
    main()
