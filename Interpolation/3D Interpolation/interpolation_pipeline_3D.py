"""
interpolation_pipeline_3D.py
----------------------------
Apply temporal smoothing to lifted 3D pose JSONs.

Reads a 3D pose JSON (MMPose-style, with 'keypoints_3d' per instance per
frame), applies either Savitzky-Golay or Gaussian filtering along the
temporal axis, and writes the smoothed result to a new JSON.

Smoothing is applied to every frame uniformly (not gated by classifier
reliability), because the visible 3D wildness in Unity originates from
VideoPose3D's temporal lifting, not from per-frame 2D errors. Smoothing
the whole trajectory addresses the actual source of instability.

USAGE
-----
python interpolation_pipeline_3D.py \\
    --json /path/to/Ramona_1_1639_lifted_3d.json \\
    --output /path/to/Ramona_1_1639_lifted_3d_smoothed.json \\
    --filter savgol \\
    --window 7 \\
    --polyorder 2

# Or with Gaussian:
python interpolation_pipeline_3D.py \\
    --json /path/to/Ramona_1_1639_lifted_3d.json \\
    --output /path/to/Ramona_1_1639_lifted_3d_gauss.json \\
    --filter gaussian \\
    --sigma 2.0

OPTIONS
-------
--filter      'savgol' or 'gaussian'.
--window      Savitzky-Golay window length (frames). Must be odd. Default 7.
--polyorder   Savitzky-Golay polynomial order. Must be < window. Default 2.
--sigma       Gaussian standard deviation in frames. Default 2.0.
--per_track   If set, smooth each track_id independently rather than
              smoothing instance positions in absolute order. Recommended
              when multiple people are in frame.

WHAT GETS SMOOTHED
------------------
keypoints_3d (the 17x3 array per instance per frame). Per joint, per axis,
along the temporal dimension. Other fields (keypoint_scores_3d, track_id,
keypoints_2d) are passed through unchanged.

WHY UNIFORM SMOOTHING (not gated by reliability)
------------------------------------------------
Diagnostic on frame 245 of Ramona showed that 2D positions differed from
raw baseline by ~2 pixels at most, yet Unity rendered visible 3D artifacts
on the same frame. This means VideoPose3D's temporal convolutions amplify
small 2D differences into large 3D differences, regardless of whether the
classifier flagged anything. Smoothing the 3D output addresses this
directly. Conditioning on per-frame reliability would not help because
the reliability is mostly fine on the affected frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def load_3d_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_3d_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent="\t")


# ----------------------------------------------------------------------------
# Trajectory extraction and reinsertion
# ----------------------------------------------------------------------------

def extract_trajectories(data, per_track=True):
    """
    Returns a dict mapping track_id (or instance positional index if no
    track_id is present) to:
        {
            'frame_indices': list of frame indices (0-based, contiguous),
            'kpts_3d':       np.ndarray of shape (n_frames, 17, 3),
            'instance_refs': list of (frame_entry_idx, inst_idx) tuples
                             so we can write smoothed values back exactly.
        }

    A 'track' is a temporally contiguous sequence of (frame, instance)
    entries with the same track_id. Gaps in track presence (track absent
    for some frames) split it into separate sub-trajectories so that
    smoothing doesn't bridge across gaps.
    """
    tracks = {}  # key -> list of (frame_entry_idx, inst_idx, frame_id, kpts_3d)

    for fe_idx, frame_entry in enumerate(data.get("instance_info", [])):
        frame_id = int(frame_entry.get("frame_id", fe_idx + 1))
        instances = frame_entry.get("instances", [])
        for inst_idx, instance in enumerate(instances):
            kp3d = instance.get("keypoints_3d", None)
            if kp3d is None:
                continue
            arr = np.asarray(kp3d, dtype=np.float32)
            # Some MMPose outputs nest a singleton outer dim; flatten if so.
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.shape != (17, 3):
                # Unexpected shape — skip but warn.
                print(f"  [warn] Frame {frame_id}, instance {inst_idx}: "
                      f"keypoints_3d shape {arr.shape}, expected (17, 3). Skipping.")
                continue

            if per_track:
                key = instance.get("track_id", None)
                if key is None or (isinstance(key, float) and np.isnan(key)) or key == -1:
                    # Fall back to positional index keyed within this frame.
                    # Treating positional index as a "track" only works if
                    # instance order is stable across frames. If it isn't,
                    # use --per_track False and accept whole-sequence smoothing.
                    key = f"pos_{inst_idx}"
            else:
                # Smooth whole sequence uniformly (single trajectory).
                # Only safe with 1 person on screen.
                key = "global"

            tracks.setdefault(key, []).append(
                (fe_idx, inst_idx, frame_id, arr)
            )

    # Sort each track by frame_id and split on temporal gaps (>1 frame jump).
    out = {}
    for key, entries in tracks.items():
        entries.sort(key=lambda e: e[2])  # sort by frame_id
        sub_idx = 0
        cur = []
        prev_fid = None
        for entry in entries:
            fe_idx, inst_idx, fid, kp = entry
            if prev_fid is not None and fid - prev_fid > 1:
                # Gap — close current sub-trajectory.
                if cur:
                    out[f"{key}__seg{sub_idx}"] = _materialize(cur)
                    sub_idx += 1
                cur = []
            cur.append(entry)
            prev_fid = fid
        if cur:
            label = f"{key}__seg{sub_idx}" if sub_idx > 0 or f"{key}__seg0" in out else key
            out[label] = _materialize(cur)

    return out


def _materialize(entries):
    """Convert list of entries into the trajectory dict shape."""
    fe_idxs = [e[0] for e in entries]
    inst_idxs = [e[1] for e in entries]
    fids = [e[2] for e in entries]
    arr = np.stack([e[3] for e in entries], axis=0)  # (n, 17, 3)
    return {
        "frame_indices": fids,
        "kpts_3d":       arr,
        "instance_refs": list(zip(fe_idxs, inst_idxs)),
    }


def reinsert_trajectories(data, trajectories):
    """Write smoothed kpts_3d back into the JSON structure."""
    for key, traj in trajectories.items():
        smoothed = traj["kpts_3d"]
        for (fe_idx, inst_idx), kp in zip(traj["instance_refs"], smoothed):
            instance = data["instance_info"][fe_idx]["instances"][inst_idx]
            # Match the original nesting if the original was [[...]].
            original = instance["keypoints_3d"]
            if isinstance(original, list) and len(original) == 1 and \
               isinstance(original[0], list) and len(original[0]) == 17:
                instance["keypoints_3d"] = [kp.tolist()]
            else:
                instance["keypoints_3d"] = kp.tolist()
    return data


# ----------------------------------------------------------------------------
# Smoothers
# ----------------------------------------------------------------------------

def smooth_savgol(kpts_3d, window_length, polyorder):
    n_frames = kpts_3d.shape[0]
    if n_frames < window_length:
        # Reduce window to largest valid odd value <= n_frames.
        adj = max(3, n_frames if n_frames % 2 == 1 else n_frames - 1)
        if adj <= polyorder:
            return kpts_3d.copy()
        if adj > n_frames:                # ← ADD THIS GUARD
            return kpts_3d.copy()         # too short to smooth
        print(f"  [info] Trajectory has {n_frames} frames; "
              f"reducing window from {window_length} to {adj}.")
        window_length = adj
    if window_length <= polyorder:
        return kpts_3d.copy()
    # ... rest unchanged

    out = np.empty_like(kpts_3d)
    for j in range(kpts_3d.shape[1]):       # 17 joints
        for axis in range(kpts_3d.shape[2]):  # x, y, z
            out[:, j, axis] = savgol_filter(
                kpts_3d[:, j, axis],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",  # treats edges sensibly without padding
            )
    return out


def smooth_gaussian(kpts_3d, sigma):
    """
    kpts_3d: (n_frames, 17, 3)
    Returns: same shape, smoothed along axis=0.
    """
    n_frames = kpts_3d.shape[0]
    if n_frames < 3:
        return kpts_3d.copy()
    return gaussian_filter1d(kpts_3d, sigma=sigma, axis=0, mode="nearest")


# ----------------------------------------------------------------------------
# Diagnostic
# ----------------------------------------------------------------------------

def report_smoothing(traj_in, traj_out):
    """Report per-joint mean/max displacement before vs after smoothing."""
    diff = traj_out - traj_in  # (n, 17, 3)
    per_joint_mean = np.linalg.norm(diff, axis=2).mean(axis=0)  # (17,)
    per_joint_max  = np.linalg.norm(diff, axis=2).max(axis=0)   # (17,)
    return per_joint_mean, per_joint_max


JOINT_NAMES = [
    "root", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck_base", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Apply temporal smoothing to lifted 3D pose JSONs."
    )
    ap.add_argument("--json",     default = '/Users/emmavejcik/Desktop/DeepScreens/Interpolation/2D Interpolation/Outputs/3D Outputs/results_segment_1_1639.json',required=True, help="Input lifted-3D JSON.")
    ap.add_argument("--output",   default = 'Smoothed_Output/smoothed.json', required=True, help="Output smoothed JSON.")
    ap.add_argument("--filter",   choices=["savgol", "gaussian"], default="savgol",
                    help="Filter to apply (default: savgol).")
    ap.add_argument("--window",   type=int,   default=7,
                    help="Savgol window length in frames (must be odd). Default 7.")
    ap.add_argument("--polyorder", type=int,  default=2,
                    help="Savgol polynomial order (must be < window). Default 2.")
    ap.add_argument("--sigma",    type=float, default=2.0,
                    help="Gaussian sigma in frames. Default 2.0.")
    ap.add_argument("--per_track", action="store_true", default=True,
                    help="Smooth each track_id independently. Default True.")
    ap.add_argument("--no_per_track", dest="per_track", action="store_false",
                    help="Disable per-track smoothing (use only when 1 person on screen).")
    args = ap.parse_args()

    if args.filter == "savgol":
        if args.window % 2 == 0:
            raise SystemExit("ERROR: --window must be odd.")
        if args.polyorder >= args.window:
            raise SystemExit("ERROR: --polyorder must be < --window.")

    print(f"Loading: {args.json}")
    data = load_3d_json(args.json)
    n_frames_total = len(data.get("instance_info", []))
    print(f"  {n_frames_total} frames in JSON.")

    # Extract trajectories
    trajectories = extract_trajectories(data, per_track=args.per_track)
    print(f"  Found {len(trajectories)} contiguous trajectories.")
    for key, traj in trajectories.items():
        print(f"    [{key}] {len(traj['frame_indices'])} frames "
              f"(frame_id {traj['frame_indices'][0]}..{traj['frame_indices'][-1]})")

    # Smooth each
    print(f"\nApplying filter: {args.filter}")
    if args.filter == "savgol":
        print(f"  window={args.window}  polyorder={args.polyorder}")
    else:
        print(f"  sigma={args.sigma}")

    aggregate_displacement = []

    for key, traj in trajectories.items():
        kp_in = traj["kpts_3d"]
        if args.filter == "savgol":
            kp_out = smooth_savgol(kp_in, args.window, args.polyorder)
        else:
            kp_out = smooth_gaussian(kp_in, args.sigma)
        traj["kpts_3d"] = kp_out

        per_joint_mean, per_joint_max = report_smoothing(kp_in, kp_out)
        aggregate_displacement.append(per_joint_mean)

        print(f"\n  [{key}]  mean displacement per joint (3D units):")
        for j in range(17):
            print(f"    {JOINT_NAMES[j]:<15s} mean={per_joint_mean[j]:.4f}  "
                  f"max={per_joint_max[j]:.4f}")

    # Aggregate report
    if aggregate_displacement:
        all_mean = np.stack(aggregate_displacement, axis=0).mean(axis=0)
        print(f"\nAggregate mean displacement across all trajectories (per joint):")
        for j in range(17):
            print(f"  {JOINT_NAMES[j]:<15s} {all_mean[j]:.4f}")

    # Reinsert and write
    data = reinsert_trajectories(data, trajectories)
    save_3d_json(args.output, data)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()