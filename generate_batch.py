#!/usr/bin/env python3
# gen_run_from_config.py
# Usage: python gen_run_from_config.py <batch_yaml> <target_config_yaml> <out_bash>

import sys, os, pathlib, json

try:
    import yaml  # pyyaml for reading only
except Exception:
    print("Please `pip install pyyaml`", file=sys.stderr)
    sys.exit(1)

# This updater edits TOP-LEVEL keys in the target config file (config.yaml)
UPDATE_CONFIG_FN = r"""update_config_inline() {
python - "$TARGET_CONFIG" <<'PY'
import os, sys
from ruamel.yaml import YAML

cfg_file = sys.argv[1]
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open(cfg_file, "r") as f:
    data = yaml.load(f) or {}

def parse_csv(s):
    s = (s or "").strip()
    if not s: return []
    out = []
    for v in s.split(","):
        v = v.strip()
        if not v: continue
        try:
            out.append(int(v))
        except ValueError:
            pass
    return out

env = os.environ

# Update top-level keys in the target config file
if "FRAME_GENERATOR" in env:
    data["frame_generator"] = env.get("FRAME_GENERATOR")
if "RECON_MODEL" in env:
    data["reconstruction_model"] = env.get("RECON_MODEL")
if "FRAME_ACCUMULATOR" in env:
    data["frame_accumulator"] = env.get("FRAME_ACCUMULATOR")
if "GPS_TOLERANCE" in env:
    try:
        data["gps_tolerance"] = int(env.get("GPS_TOLERANCE"))
    except Exception:
        data["gps_tolerance"] = env.get("GPS_TOLERANCE")

# timewindows is top-level in your config.yaml; preserve other keys under timewindows
tw = data.get("timewindows", {}) or {}
tw_list   = parse_csv(env.get("TW_LIST", ""))
tw_events = parse_csv(env.get("TW_EVENTS", ""))

if tw_list:
    tw["list"] = tw_list
if tw_events:
    tw["num_events"] = tw_events
data["timewindows"] = tw

with open(cfg_file, "w") as f:
    yaml.dump(data, f)

print(f"[config] updated {cfg_file} (top-level) with FRAME_GENERATOR={env.get('FRAME_GENERATOR')} GPS_TOLERANCE={env.get('GPS_TOLERANCE')}")
PY
}
"""

# Baseline updater: unchanged behavior (writes/merges baseline config YAML)
UPDATE_BASELINE_FN = r"""update_baseline_inline() {
# ENV:
#   BASELINE_NAME      (required)  e.g. sparse_event
#   BASELINE_JSON      (required)  JSON object of overrides
#   BASELINE_CFG_PATH  (optional)  explicit path to YAML
#   BASELINE_DIR       (optional)  directory to search (default: ./baselines)
python - <<'PY'
import os, sys, json
from pathlib import Path
from ruamel.yaml import YAML

name = os.environ.get("BASELINE_NAME")
if not name:
    print("[baseline] BASELINE_NAME is required", file=sys.stderr); sys.exit(2)

raw = os.environ.get("BASELINE_JSON","").strip()
if not raw:
    print(f"[baseline] no options for {name}; skipping", file=sys.stderr); sys.exit(0)

try:
    overrides = json.loads(raw)
    if not isinstance(overrides, dict):
        raise ValueError("BASELINE_JSON must be a JSON object")
except Exception as e:
    print(f"[baseline] invalid BASELINE_JSON: {e}", file=sys.stderr); sys.exit(2)

explicit = os.environ.get("BASELINE_CFG_PATH", "").strip()
base_dir = os.environ.get("BASELINE_DIR", "").strip() or "baselines"

candidates = []
if explicit:
    candidates.append(Path(explicit))
else:
    for d in [base_dir, "baselines", "baseline_configs"]:
        candidates.append(Path(d) / f"{name}.yaml")
        candidates.append(Path(d) / f"{name}.yml")

target = None
for p in candidates:
    if p.exists():
        target = p; break
if target is None:
    target = candidates[0]

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

data = {}
if target.exists():
    try:
        with target.open("r") as f:
            data = yaml.load(f) or {}
    except Exception:
        data = {}

# shallow merge (override top-level keys)
for k, v in overrides.items():
    data[k] = v

target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w") as f:
    yaml.dump(data, f)

print(f"[baseline] updated {target} with {overrides}")
PY
}
"""

def csv_join(vals):
    return ",".join(str(v) for v in vals) if vals else ""

def _norm_baseline_cfg(bc):
    """
    Normalize baseline_config to {name: {overrides}}.
    Accepts either:
      - {"sparse_event": {...}, "lens": {...}}
      - [{"sparse_event": {...}}, {"lens": {...}}]
    """
    if not bc:
        return {}
    if isinstance(bc, dict):
        return {str(k): (v or {}) for k, v in bc.items()}
    if isinstance(bc, list):
        out = {}
        for item in bc:
            if isinstance(item, dict):
                for k, v in item.items():
                    out[str(k)] = v or {}
        return out
    return {}

def main():
    # Expect three args: batch_yaml, target_config_yaml, out_bash
    if len(sys.argv) != 4:
        print("Usage: python gen_run_from_config.py <batch_yaml> <target_config_yaml> <out_bash>")
        sys.exit(1)

    batch_cfg_path = pathlib.Path(sys.argv[1]).resolve()
    target_cfg_path = pathlib.Path(sys.argv[2]).resolve()
    out_bash = pathlib.Path(sys.argv[3]).resolve()

    try:
        batch_cfg = yaml.safe_load(batch_cfg_path.read_text()) or {}
    except Exception as e:
        print(f"Failed to read {batch_cfg_path}: {e}", file=sys.stderr)
        sys.exit(1)

    experiments = batch_cfg.get("batch_experiments", [])
    if not experiments:
        print("No 'batch_experiments' found in batch YAML.", file=sys.stderr)
        sys.exit(1)

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    # Set TARGET_CONFIG to the target config file (this is the file that will be edited)
    lines.append(f'TARGET_CONFIG="{target_cfg_path}"')
    lines.append("")
    # helper functions
    lines.append(UPDATE_CONFIG_FN)
    lines.append("")
    lines.append(UPDATE_BASELINE_FN)
    lines.append("")

    exp_idx = 0
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        if "dataset" not in exp or "reference" not in exp:
            continue

        exp_idx += 1
        dataset   = str(exp["dataset"])
        reference = str(exp["reference"])
        queries   = [str(q) for q in (exp.get("queries", []) or [])]
        baselines = [str(b) for b in (exp.get("baselines", []) or [])]

        econf     = exp.get("config", {}) or {}
        fg        = str(econf.get("frame_generator", "frames"))
        recon     = str(econf.get("reconstruction_model", "e2vid"))
        accum     = str(econf.get("frame_accumulator", "polarity"))
        tw        = econf.get("timewindows", {}) or {}
        tw_list   = csv_join(tw.get("list", []))
        tw_events = csv_join(tw.get("num_events", []))
        gps_val = econf.get("gps_tolerance", None)
        tw = econf.get("timewindows", {}) or {}
        if gps_val in (None, "", []):
            gps_val = tw.get("gps_tolerance", None)

        gps_env = ""
        if gps_val not in (None, ""):
            try:
                gps_env = f' GPS_TOLERANCE="{int(gps_val)}"'
            except Exception:
                s = str(gps_val).strip()
                if s != "":
                    gps_env = f' GPS_TOLERANCE="{s}"'  # last resort, still non-empty

        bc_map    = _norm_baseline_cfg(exp.get("baseline_config", {}) or {})

        lines.append(f'echo "=== Experiment {exp_idx}: {dataset} (ref={reference}) ==="')
        # IMPORTANT: this update_config_inline call now targets TARGET_CONFIG (config.yaml),
        # so the top-level keys in config.yaml will be updated (timewindows, frame_generator, gps_tolerance, etc.)
        lines.append(
            f'FRAME_GENERATOR="{fg}" TW_LIST="{tw_list}" TW_EVENTS="{tw_events}" '
            f'RECON_MODEL="{recon}" FRAME_ACCUMULATOR="{accum}"{gps_env} '
            f'update_config_inline "{target_cfg_path}"'
        )
        lines.append("")

        for q in queries:
            for b in baselines:
                overrides = bc_map.get(b, {}) or {}

                # If overrides has a 'method' key which is a list, expand into multiple runs
                if isinstance(overrides.get("method"), list):
                    methods = overrides["method"]
                    # keep other keys unchanged
                    other_overrides = {k: v for k, v in overrides.items() if k != "method"}
                    for m in methods:
                        # produce override where method is a single string (not a list)
                        ov = dict(other_overrides)
                        ov["method"] = m
                        ov_json = json.dumps(ov)
                        # update baseline config for this single-method run
                        lines.append(f'BASELINE_NAME="{b}" BASELINE_JSON=\'{ov_json}\' update_baseline_inline')
                        cmd = f'python eventlab_run.py "{b}" "{dataset}" "{reference}" "{q}"'
                        lines.append(f'echo "+ {cmd} (method={m})"')
                        lines.append(cmd)
                else:
                    # normal path: either no overrides, or overrides without a method-list
                    if overrides:
                        ov_json = json.dumps(overrides)
                        lines.append(f'BASELINE_NAME="{b}" BASELINE_JSON=\'{ov_json}\' update_baseline_inline')
                    cmd = f'python eventlab_run.py "{b}" "{dataset}" "{reference}" "{q}"'
                    lines.append(f'echo "+ {cmd}"')
                    lines.append(cmd)

        lines.append("")

    out = "\n".join(lines) + "\n"
    out_bash.write_text(out)
    os.chmod(out_bash, 0o755)
    print(f"Wrote {out_bash}")

if __name__ == "__main__":
    main()