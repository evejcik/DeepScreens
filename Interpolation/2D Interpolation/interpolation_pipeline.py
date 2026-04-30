import json
import numpy as np
import pandas as pd
from pathlib import Path
from kalman import apply_kalman_filter
from cubic_spline import apply_cubic_spline

UNRELIABLE_THRESHOLD = 0.5
VALIDITY_HORIZON = 20

H36M_JOINT_NAMES = {
    0: 'root',        1: 'right_hip',     2: 'right_knee',   3: 'right_ankle',
    4: 'left_hip',    5: 'left_knee',     6: 'left_ankle',   7: 'spine',
    8: 'thorax',      9: 'neck_base',     10: 'head',
    11: 'left_shoulder', 12: 'left_elbow', 13: 'left_wrist',
    14: 'right_shoulder', 15: 'right_elbow', 16: 'right_wrist'
}

def json_to_dataframe(data: dict, film: str, threshold: float) -> pd.DataFrame:
    """
    Deserialize cleaned 17-kp JSON into the long DataFrame format
    expected by kalman.py and cubic_spline.py.

    Derives reliability_category_int from prob_unreliable scores:
        prob_unreliable > threshold  -> 2 (dont_trust)
        prob_unreliable <= threshold -> 0 (trust)

    No partial_trust (1) class is used — the Kalman R matrix should
    be scaled continuously from prob_unreliable instead.
    """
    rows = []
    for frame_entry in data['instance_info']:
        frame_id = int(frame_entry['frame_id']) - 1  # JSON is 1-indexed
        for instance_id, instance in enumerate(frame_entry.get('instances', [])):
            kps    = instance['keypoints']           # list of [x, y], len 17
            scores = instance['keypoint_scores']     # list of floats, len 17
            interp = instance.get('keypoint_interpolated', [False] * 17)

            for joint_id in range(17):
                prob = scores[joint_id]
                rows.append({
                    'film':                    film,
                    'frame_id':                frame_id,
                    'instance_id':             instance_id,
                    'joint_id':                joint_id,
                    'joint_name':              H36M_JOINT_NAMES[joint_id],
                    'x':                       kps[joint_id][0],
                    'y':                       kps[joint_id][1],
                    'prob_unreliable':         prob,
                    'reliability_category_int': 1 if prob > threshold else 0,
                    'already_interpolated':    interp[joint_id],
                })

    df = pd.DataFrame(rows)

    # Compute velocity features — Kalman init uses these if present
    df = df.sort_values(['film', 'instance_id', 'joint_id', 'frame_id'])
    df['x_velocity'] = df.groupby(['film', 'instance_id', 'joint_id'])['x'].diff().fillna(0.0)
    df['y_velocity'] = df.groupby(['film', 'instance_id', 'joint_id'])['y'].diff().fillna(0.0)

    return df


def dataframe_to_json(data: dict, df: pd.DataFrame) -> dict:
    """
    Write x_filled, y_filled back into the JSON structure.
    Updates keypoints, keypoint_scores (prob_unreliable unchanged),
    and keypoint_interpolated (True where gap was filled).
    """
    # Build lookup: (frame_id, instance_id, joint_id) -> row
    lookup = df.set_index(['frame_id', 'instance_id', 'joint_id'])

    for frame_entry in data['instance_info']:
        frame_id = int(frame_entry['frame_id']) - 1
        for instance_id, instance in enumerate(frame_entry.get('instances', [])):
            new_kps    = instance['keypoints'][:]
            new_interp = instance.get('keypoint_interpolated', [False] * 17)[:]

            for joint_id in range(17):
                key = (frame_id, instance_id, joint_id)
                if key not in lookup.index:
                    continue
                row = lookup.loc[key]
                new_kps[joint_id]    = [float(row['x_filled']), float(row['y_filled'])]
                # Mark as interpolated if it was flagged unreliable and got filled
                if row['reliability_category_int'] == 1:
                    new_interp[joint_id] = True

            instance['keypoints']             = new_kps
            instance['keypoint_interpolated'] = new_interp
            # keypoint_scores (prob_unreliable) deliberately left unchanged

    return data


def run_interpolation_pipeline(json_path: str, output_path: str, film: str,
                                threshold: float = UNRELIABLE_THRESHOLD,
                                validity_horizon: int = VALIDITY_HORIZON):
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = json_to_dataframe(data, film, threshold)

    print(f"Loaded {len(df)} joint-frame rows")
    print(f"Unreliable (flagged): {(df['reliability_category_int'] == 1).sum()}")
    print(f"Reliable:             {(df['reliability_category_int'] == 0).sum()}")

    # Verify joint ID space — catch COCO bleed-through immediately
    id_check = df[['joint_name', 'joint_id']].drop_duplicates().sort_values('joint_id')
    print("\nJoint ID verification:")
    print(id_check.to_string(index=False))
    assert df['joint_id'].max() == 16, "Joint IDs exceed H36M range — possible COCO bleed-through"

    df = apply_kalman_filter(df)
    df = apply_cubic_spline(df, validity_horizon=validity_horizon)

    data = dataframe_to_json(data, df)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent='\t')

    print(f"\nWrote interpolated JSON to {output_path}")
    return df

import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json',        required=True,  help='Path to cleaned 17-kp JSON')
    ap.add_argument('--output',      required=True,  help='Path for interpolated output JSON')
    ap.add_argument('--film',        required=True,  help='Film name key (must match CSV film column)')
    ap.add_argument('--threshold',   type=float, default=0.5,
                    help='prob_unreliable threshold above which a joint is flagged (default 0.5)')
    ap.add_argument('--horizon',     type=int,   default=VALIDITY_HORIZON,
                    help='Max gap size in frames for cubic spline (default 20)')
    args = ap.parse_args()

    df = run_interpolation_pipeline(
        json_path       = args.json,
        output_path     = args.output,
        film            = args.film,
        threshold       = args.threshold,
        validity_horizon = args.horizon,
    )

    # Summary of what was actually interpolated
    if 'x_filled' in df.columns:
        flagged   = df['reliability_category_int'] == 1
        print(f"\nFlagged as unreliable: {flagged.sum()} joint-frames")
        changed = flagged & (
            ((df['x_filled'] - df['x']).abs() > 1e-6) |
            ((df['y_filled'] - df['y']).abs() > 1e-6)
        )
        print(f"Actually interpolated (coordinates changed): {changed.sum()} joint-frames")
        print(f"Left unchanged (no valid anchors / gap too large): {(flagged & ~changed).sum()} joint-frames")

if __name__ == '__main__':
    main()

#run as: 
# python interpolation_pipeline.py \
#   --json /path/to/Ramona_1_1639_pred_aggregated.json \
#   --output /path/to/Ramona_1_1639_interpolated.json \
#   --film Ramona_1_1639 \
#   --threshold 0.5