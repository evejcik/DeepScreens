import numpy as np
import pandas as pd

# Bone connections as (parent_id, child_id, bone_name)
# These are the only structurally meaningful lower body bones
LOWER_BODY_BONES = [
    (11, 13, 'left_thigh'),       # left_hip -> left_knee
    (12, 14, 'right_thigh'),      # right_hip -> right_knee
    (13, 15, 'left_shin'),        # left_knee -> left_ankle
    (14, 16, 'right_shin'),       # right_knee -> right_ankle
    (11, 12, 'pelvis_width'),     # left_hip <-> right_hip
]

# Physiologically plausible bone length ratios relative to pelvis width.
# These are loose bounds — tighter bounds will produce more false positives
# on unusual poses (crouching, sitting, extreme angles).
# Format: (min_ratio, max_ratio) where ratio = bone_length / pelvis_width
# Derived from anthropometric data — adjust if your movies show unusual cases.
BONE_RATIO_BOUNDS = {
    'left_thigh':    (0.8, 2.8),
    'right_thigh':   (0.8, 2.8),
    'left_shin':     (0.7, 2.5),
    'right_shin':    (0.7, 2.5),
    'pelvis_width':  (1.0, 1.0),  # reference bone, ratio always 1.0
}

def get_joint_coords(instance_keypoints, joint_id):
    """
    Returns (x, y) for a joint_id from a flat keypoints list.
    Returns None if joint_id is out of range.
    """
    if joint_id < len(instance_keypoints):
        return instance_keypoints[joint_id]
    return None


def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_plausibility_for_instance(keypoints, confidences, conf_threshold=0.3):
    """
    Given a list of keypoints [[x,y], ...] and confidence scores,
    compute geometric plausibility for each lower body joint.

    Returns a dict: joint_id -> {
        'geom_plausible': bool or None,
        'geom_flag': str,        # reason if implausible
        'bone_length': float,    # length of bone ending at this joint
        'bone_ratio': float,     # ratio to pelvis width
    }
    """
    results = {}

    # Get pelvis width as reference — this is our normalizing bone
    lhip = get_joint_coords(keypoints, 11)
    rhip = get_joint_coords(keypoints, 12)
    lhip_conf = confidences[11] if 11 < len(confidences) else 0
    rhip_conf = confidences[12] if 12 < len(confidences) else 0

    pelvis_width = None
    if (lhip is not None and rhip is not None and
            lhip_conf > conf_threshold and rhip_conf > conf_threshold):
        pelvis_width = euclidean(lhip, rhip)

    # If pelvis width is degenerate (hips on top of each other),
    # we can't normalize — flag everything
    if pelvis_width is not None and pelvis_width < 5:
        pelvis_width = None

    for parent_id, child_id, bone_name in LOWER_BODY_BONES:
        parent_pt = get_joint_coords(keypoints, parent_id)
        child_pt  = get_joint_coords(keypoints, child_id)
        parent_conf = confidences[parent_id] if parent_id < len(confidences) else 0
        child_conf  = confidences[child_id]  if child_id  < len(confidences) else 0

        # Skip if either endpoint has low confidence — can't make a plausibility
        # judgment if MMPose itself isn't confident about the anchor joint
        if parent_conf < conf_threshold or child_conf < conf_threshold:
            results[child_id] = {
                'geom_plausible': None,
                'geom_flag': 'low_confidence_anchor',
                'bone_length': None,
                'bone_ratio': None,
            }
            continue

        if parent_pt is None or child_pt is None:
            results[child_id] = {
                'geom_plausible': None,
                'geom_flag': 'missing_keypoint',
                'bone_length': None,
                'bone_ratio': None,
            }
            continue

        bone_len = euclidean(parent_pt, child_pt)

        # If pelvis width is unavailable, we can still flag zero-length bones
        if bone_len < 2:
            results[child_id] = {
                'geom_plausible': False,
                'geom_flag': 'zero_length_bone',
                'bone_length': bone_len,
                'bone_ratio': None,
            }
            continue

        if pelvis_width is None:
            # Can't normalize without pelvis reference
            results[child_id] = {
                'geom_plausible': None,
                'geom_flag': 'no_pelvis_reference',
                'bone_length': bone_len,
                'bone_ratio': None,
            }
            continue

        ratio = bone_len / pelvis_width
        lo, hi = BONE_RATIO_BOUNDS.get(bone_name, (0.5, 3.5))

        if ratio < lo or ratio > hi:
            results[child_id] = {
                'geom_plausible': False,
                'geom_flag': f'ratio_{ratio:.2f}_outside_bounds_{lo}_{hi}',
                'bone_length': round(bone_len, 2),
                'bone_ratio': round(ratio, 2),
            }
        else:
            results[child_id] = {
                'geom_plausible': True,
                'geom_flag': None,
                'bone_length': round(bone_len, 2),
                'bone_ratio': round(ratio, 2),
            }

    return results


def add_geometric_plausibility(df, data, conf_threshold=0.3):
    """
    Adds geom_plausible, geom_flag, bone_length, bone_ratio columns to df.

    df must have: frame_id, instance_id, joint_id columns.
    data is the raw JSON dict.
    """
    # Build lookup: (frame_id, instance_id) -> plausibility results per joint_id
    plausibility_cache = {}

    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        for inst_idx, instance in enumerate(frame.get('instances', [])):
            keypoints   = instance.get('keypoints', [])
            confidences = instance.get('keypoint_scores', [])
            key = (frame_id, inst_idx)
            plausibility_cache[key] = compute_plausibility_for_instance(
                keypoints, confidences, conf_threshold
            )

    # Map results back onto dataframe rows
    geom_plausible_col = []
    geom_flag_col      = []
    bone_length_col    = []
    bone_ratio_col     = []

    for _, row in df.iterrows():
        key      = (int(row['frame_id']), int(row['instance_id']))
        joint_id = int(row['joint_id'])
        result   = plausibility_cache.get(key, {}).get(joint_id, None)

        if result is None:
            # Joint not in any bone connection — no plausibility check defined
            geom_plausible_col.append(None)
            geom_flag_col.append('not_checked')
            bone_length_col.append(None)
            bone_ratio_col.append(None)
        else:
            geom_plausible_col.append(result['geom_plausible'])
            geom_flag_col.append(result['geom_flag'])
            bone_length_col.append(result['bone_length'])
            bone_ratio_col.append(result['bone_ratio'])

    df = df.copy()
    df['geom_plausible'] = geom_plausible_col
    df['geom_flag']      = geom_flag_col
    df['bone_length']    = bone_length_col
    df['bone_ratio']     = bone_ratio_col

    return df


def compute_boundary_distance(df, frame_width, frame_height):
    """
    Computes distance from each joint to the nearest frame boundary.
    Adds dist_to_boundary column.
    """
    df = df.copy()
    dist_left   = df['x']
    dist_right  = frame_width  - df['x']
    dist_top    = df['y']
    dist_bottom = frame_height - df['y']
    df['dist_to_boundary'] = np.minimum(
        np.minimum(dist_left, dist_right),
        np.minimum(dist_top,  dist_bottom)
    )
    return df