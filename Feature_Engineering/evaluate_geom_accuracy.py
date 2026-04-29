import pandas as pd

import numpy as np


# Bone connections as (parent_id, child_id, bone_name)
LOWER_BODY_BONES = [
    (11, 13, 'left_thigh'),    # left_hip -> left_knee
    (12, 14, 'right_thigh'),   # right_hip -> right_knee
    (13, 15, 'left_shin'),     # left_knee -> left_ankle
    (14, 16, 'right_shin'),    # right_knee -> right_ankle
    (11, 12, 'pelvis_width'),  # left_hip <-> right_hip (reference bone)
]

# Tightened bounds based on empirical data:
# - Left knee FPs were all ratio > 2.80, going up to 9.99
# - These are genuine MMPose errors under skirt occlusion
# - Upper bound tightened from 2.8 to 2.2 to catch more implausible cases
# - Lower bound raised slightly — very short bones in 2D are usually
#   perspective/foreshortening artifacts, not real anatomy
# Adjust these if you see too many FPs on sitting/crouching poses
BONE_RATIO_BOUNDS = {
    'left_thigh':   (0.9, 2.2),
    'right_thigh':  (0.9, 2.2),
    'left_shin':    (0.8, 2.0),
    'right_shin':   (0.8, 2.0),
    'pelvis_width': (1.0, 1.0),
}

def contains_geom_implausibility(reason_str):
    """
    Returns True if 'geometric implausibility' appears anywhere in the
    reason string, handling compound reasons and underscore variants.
    """
    if pd.isna(reason_str):
        return False
    s = str(reason_str).lower().replace('_', ' ')
    return 'geometric implausibility' in s


def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_plausibility_for_instance(keypoints, confidences, conf_threshold=0.3):
    """
    Computes geometric plausibility for each lower body joint in one instance.

    Returns dict: joint_id -> {
        'geom_plausible': bool or None,
        'geom_flag': str,
        'bone_length': float or None,
        'bone_ratio': float or None,
    }
    """
    results = {}

    # Compute pelvis width as normalizing reference
    lhip = keypoints[11] if 11 < len(keypoints) else None
    rhip = keypoints[12] if 12 < len(keypoints) else None
    lhip_conf = confidences[11] if 11 < len(confidences) else 0
    rhip_conf = confidences[12] if 12 < len(confidences) else 0

    pelvis_width = None
    if (lhip is not None and rhip is not None
            and lhip_conf > conf_threshold
            and rhip_conf > conf_threshold):
        pw = euclidean(lhip, rhip)
        pelvis_width = pw if pw >= 5 else None  # degenerate if < 5px

    for parent_id, child_id, bone_name in LOWER_BODY_BONES:
        parent_pt   = keypoints[parent_id]   if parent_id   < len(keypoints)   else None
        child_pt    = keypoints[child_id]    if child_id    < len(keypoints)    else None
        parent_conf = confidences[parent_id] if parent_id   < len(confidences) else 0
        child_conf  = confidences[child_id]  if child_id    < len(confidences) else 0

        # Can't evaluate if anchor joint has low confidence
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

        # Zero-length bone is always implausible
        if bone_len < 2:
            results[child_id] = {
                'geom_plausible': False,
                'geom_flag': 'zero_length_bone',
                'bone_length': round(bone_len, 2),
                'bone_ratio': None,
            }
            continue

        if pelvis_width is None:
            results[child_id] = {
                'geom_plausible': None,
                'geom_flag': 'no_pelvis_reference',
                'bone_length': round(bone_len, 2),
                'bone_ratio': None,
            }
            continue

        ratio = bone_len / pelvis_width
        lo, hi = BONE_RATIO_BOUNDS.get(bone_name, (0.5, 3.0))

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
    """
    cache = {}
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        for inst_idx, instance in enumerate(frame.get('instances', [])):
            kps   = instance.get('keypoints', [])
            confs = instance.get('keypoint_scores', [])
            cache[(frame_id, inst_idx)] = compute_plausibility_for_instance(
                kps, confs, conf_threshold
            )

    rows_geom = []
    for _, row in df.iterrows():
        key      = (int(row['frame_id']), int(row['instance_id']))
        joint_id = int(row['joint_id'])
        result   = cache.get(key, {}).get(joint_id)

        if result is None:
            rows_geom.append({
                'geom_plausible': None,
                'geom_flag': 'not_checked',
                'bone_length': None,
                'bone_ratio': None,
            })
        else:
            rows_geom.append(result)

    geom_df = pd.DataFrame(rows_geom, index=df.index)
    return pd.concat([df, geom_df], axis=1)


def compute_boundary_distance(df, frame_width, frame_height):
    """Adds dist_to_boundary column."""
    df = df.copy()
    df['dist_to_boundary'] = np.minimum(
        np.minimum(df['x'], frame_width  - df['x']),
        np.minimum(df['y'], frame_height - df['y'])
    ).round(2)
    return df


def evaluate_geometric_plausibility(df):
    """
    Evaluates geom_plausible against your annotations.

    Ground truth positive = reason_for_distrust contains 'geometric implausibility'
    Predicted positive    = geom_plausible == False
    """
    evaluated = df[df['geom_plausible'].notna()].copy()

    evaluated['gt_geom'] = evaluated['reason_for_distrust'].apply(
        contains_geom_implausibility
    )
    evaluated['pred_geom'] = evaluated['geom_plausible'] == False

    tp = ((evaluated['pred_geom']) &  (evaluated['gt_geom'])).sum()
    fp = ((evaluated['pred_geom']) & ~(evaluated['gt_geom'])).sum()
    tn = (~(evaluated['pred_geom']) & ~(evaluated['gt_geom'])).sum()
    fn = (~(evaluated['pred_geom']) &  (evaluated['gt_geom'])).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print("=" * 60)
    print("GEOMETRIC PLAUSIBILITY EVALUATION")
    print("=" * 60)
    print(f"Rows evaluated:  {len(evaluated)}")
    print(f"Rows skipped:    {df['geom_plausible'].isna().sum()}")
    print(f"\nCONFUSION MATRIX")
    print(f"  TP: {tp}   FP: {fp}")
    print(f"  FN: {fn}   TN: {tn}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    print("\nPIVOT: geom_plausible vs reason_for_distrust")
    print(pd.crosstab(
        evaluated['geom_plausible'],
        evaluated['reason_for_distrust'].fillna('none'),
        margins=True
    ))

    print("\nPIVOT: geom_plausible vs reliability_category")
    print(pd.crosstab(
        evaluated['geom_plausible'],
        evaluated['reliability_category'].fillna('none'),
        margins=True
    ))

    fp_rows = evaluated[
        evaluated['pred_geom'] & ~evaluated['gt_geom']
    ][['frame_id', 'instance_id', 'joint_name', 'reliability_category',
       'reason_for_distrust', 'geom_flag', 'bone_ratio', 'mmpose_confidence']]

    fn_rows = evaluated[
        ~evaluated['pred_geom'] & evaluated['gt_geom']
    ][['frame_id', 'instance_id', 'joint_name', 'reliability_category',
       'reason_for_distrust', 'geom_flag', 'bone_ratio', 'mmpose_confidence']]

    print(f"\nFALSE POSITIVES ({len(fp_rows)}): algo flagged, you didn't")
    print(fp_rows.to_string())
    print(f"\nFALSE NEGATIVES ({len(fn_rows)}): you flagged, algo missed")
    print(fn_rows.to_string())

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'false_positives': fp_rows,
        'false_negatives': fn_rows,
    }



# Usage:
# df = pd.read_csv('your_annotated_file.csv')
# results = evaluate_geometric_plausibility(df)

# Usage:
df = pd.read_csv('ramona-demo-clip_segment_1_1639.json - ramona-demo-clip_segment_1_1639.json.csv')

fn = df[
    (df['reason_for_distrust'].str.contains('geometric implausibility', na=False)) &
    (df['geom_plausible'].isna())
]
# print(fn['geom_flag'].value_counts())

print(fn['geom_flag'].value_counts(dropna=False))
print(fn[['frame_id', 'joint_name', 'geom_flag', 'bone_ratio']].head(20))

results = evaluate_geometric_plausibility(df)
