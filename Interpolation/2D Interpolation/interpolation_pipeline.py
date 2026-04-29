import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

VALIDITY_HORIZON    = 20
UNRELIABLE_THRESHOLD = 0.5  # prob_unreliable > this -> flag as unreliable (1)

H36M_JOINT_NAMES = {
    0: 'root',       1: 'right_hip',    2: 'right_knee',  3: 'right_ankle',
    4: 'left_hip',   5: 'left_knee',    6: 'left_ankle',  7: 'spine',
    8: 'thorax',     9: 'neck_base',    10: 'head',
    11: 'left_shoulder', 12: 'left_elbow',  13: 'left_wrist',
    14: 'right_shoulder', 15: 'right_elbow', 16: 'right_wrist'
}

# ── JSON ↔ DataFrame conversion ───────────────────────────────────────────────

def json_to_trajectories(instance_info: list, track_id: int) -> pd.DataFrame:
    """
    Extract one track from instance_info into long-format DataFrame.
    reliability_category_int: 0=reliable, 1=unreliable (prob_unreliable > threshold)
    """
    rows = []
    for frame in instance_info:
        frame_id = int(frame['frame_id'])
        for instance in frame.get('instances', []):
            if instance.get('track_id') != track_id:
                continue
            kp     = instance['keypoints']
            scores = instance['keypoint_scores']  # prob_unreliable
            interp = instance.get('keypoint_interpolated', [False] * 17)

            for joint_id in range(17):
                prob_unreliable = scores[joint_id]
                rel = 1 if prob_unreliable > UNRELIABLE_THRESHOLD else 0

                rows.append({
                    'film':                    'json',
                    'instance_id':             track_id,
                    'frame_id':                frame_id,
                    'joint_id':                joint_id,
                    'joint_name':              H36M_JOINT_NAMES[joint_id],
                    'x':                       float(kp[joint_id][0]),
                    'y':                       float(kp[joint_id][1]),
                    'prob_unreliable':         prob_unreliable,
                    'reliability_category_int': rel,
                    'keypoint_interpolated':   bool(interp[joint_id]),
                    # backward_pass needs annotator_confidence —
                    # substitute with prob_unreliable-based flag
                    'annotator_confidence':    'certain' if prob_unreliable < 0.3
                                               else 'fairly sure' if prob_unreliable < 0.5
                                               else 'unsure',
                })
    return pd.DataFrame(rows)


def trajectories_to_json(instance_info: list, track_id: int,
                         traj_df: pd.DataFrame) -> list:
    """
    Write x_filled, y_filled back into instance_info for a given track.
    Marks keypoint_interpolated=True for any joint that was filled.
    """
    lookup = {}
    for _, row in traj_df.iterrows():
        x_out = row.get('x_filled', row['x'])
        y_out = row.get('y_filled', row['y'])
        was_filled = (
            not pd.isna(x_out) and not pd.isna(y_out) and
            (abs(x_out - row['x']) > 1e-6 or abs(y_out - row['y']) > 1e-6)
        )
        lookup[(int(row['frame_id']), int(row['joint_id']))] = {
            'x':      float(x_out) if not pd.isna(x_out) else float(row['x']),
            'y':      float(y_out) if not pd.isna(y_out) else float(row['y']),
            'filled': bool(was_filled),
        }
    print(traj_df['joint_name'].unique())
    reliable = traj_df[traj_df['reliability_category_int'] == 0]
    print(reliable['joint_name'].value_counts())
    updated = []
    for frame in instance_info:
        frame_id  = int(frame['frame_id'])
        new_frame = dict(frame)
        new_instances = []

        for instance in frame.get('instances', []):
            new_inst = dict(instance)
            if instance.get('track_id') == track_id:
                new_kp     = [list(kp) for kp in instance['keypoints']]
                new_interp = list(instance.get(
                    'keypoint_interpolated', [False] * 17))

                for joint_id in range(17):
                    key = (frame_id, joint_id)
                    if key in lookup:
                        entry = lookup[key]
                        new_kp[joint_id] = [entry['x'], entry['y']]
                        if entry['filled']:
                            new_interp[joint_id] = True

                new_inst['keypoints']             = new_kp
                new_inst['keypoint_interpolated'] = new_interp
            new_instances.append(new_inst)

        new_frame['instances'] = new_instances
        updated.append(new_frame)

    return updated

# ── Pipeline ──────────────────────────────────────────────────────────────────

def apply_interpolation_pipeline(instance_info: list) -> list:
    from kalman import apply_kalman_filter
    from cubic_spline import apply_cubic_spline

    track_ids = set()
    for frame in instance_info:
        for inst in frame.get('instances', []):
            tid = inst.get('track_id')
            if tid is not None and tid >= 0:
                track_ids.add(tid)

    total_tracks = len(track_ids)

    for track_id in sorted(track_ids):
        print(f"\nProcessing track {track_id}...")

        traj_df = json_to_trajectories(instance_info, track_id)
        if traj_df.empty:
            print(f"  No data for track {track_id}, skipping.")
            continue

        traj_df = traj_df.sort_values(['joint_name', 'frame_id'])

        print(f"  Applying Kalman filter...")
        traj_df = apply_kalman_filter(traj_df)

        print(f"  Applying cubic spline...")
        traj_df = apply_cubic_spline(traj_df)

        instance_info = trajectories_to_json(instance_info, track_id, traj_df)
        print(f"  Track {track_id} done.")

    print(f"\n{'='*50}")
    print(f"INTERPOLATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total tracks processed: {total_tracks}")
    print(f"  Backward pass: SKIPPED (insufficient annotation coverage)")
    print(f"  Future work: re-enable when full joint annotation is available")
    print(f"{'='*50}\n")

    return instance_info

def run_on_json(input_json_path: str, output_json_path: str):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['instance_info'])} frames.")
    print(f"Threshold: prob_unreliable > {UNRELIABLE_THRESHOLD} -> flagged unreliable")

    data['instance_info'] = apply_interpolation_pipeline(data['instance_info'])

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent='\t')

    print(f"\nSaved to {output_json_path}")

    

    with open('segment_1_1639_interpolated.json') as f:
        data = json.load(f)

    # Check how many joints were marked as interpolated
    total_joints = 0
    interpolated_joints = 0
    for frame in data['instance_info']:
        for inst in frame.get('instances', []):
            interp = inst.get('keypoint_interpolated', [])
            total_joints += len(interp)
            interpolated_joints += sum(interp)

    print(f"Total joint observations: {total_joints}")
    print(f"Interpolated: {interpolated_joints} ({100*interpolated_joints/total_joints:.1f}%)")

    inst = data['instance_info'][0]['instances'][0]
    print('keypoint_scores length:', len(inst['keypoint_scores']))
    print('sample scores:', inst['keypoint_scores'][:5])
    print('keypoint_interpolated:', inst['keypoint_interpolated'][:5])


with open('segment_1_1639_interpolated.json') as f:
    data = json.load(f)

from collections import defaultdict
joint_interp = defaultdict(lambda: [0, 0])  # [interpolated, total]

H36M_JOINT_NAMES = {
    0:'root', 1:'right_hip', 2:'right_knee', 3:'right_ankle',
    4:'left_hip', 5:'left_knee', 6:'left_ankle', 7:'spine',
    8:'thorax', 9:'neck_base', 10:'head', 11:'left_shoulder',
    12:'left_elbow', 13:'left_wrist', 14:'right_shoulder',
    15:'right_elbow', 16:'right_wrist'
}

for frame in data['instance_info']:
    for inst in frame.get('instances', []):
        interp = inst.get('keypoint_interpolated', [])
        scores = inst.get('keypoint_scores', [])
        for jid in range(17):
            name = H36M_JOINT_NAMES[jid]
            joint_interp[name][1] += 1
            if jid < len(interp) and interp[jid]:
                joint_interp[name][0] += 1

print(f"{'Joint':<20} {'Interpolated':>12} {'Total':>8} {'%':>8}")
print('-' * 50)
for jid in range(17):
    name = H36M_JOINT_NAMES[jid]
    n_interp, n_total = joint_interp[name]
    print(f"{name:<20} {n_interp:>12} {n_total:>8} {100*n_interp/n_total:>7.1f}%")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True,
                    help='Path to cleaned 17-kp JSON (output of my_joints_to_json_17_kp.py)')
    ap.add_argument('--output', required=True,
                    help='Path to save interpolated JSON')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='prob_unreliable threshold (default 0.5)')
    args = ap.parse_args()

    UNRELIABLE_THRESHOLD = args.threshold
    run_on_json(args.input, args.output)
    