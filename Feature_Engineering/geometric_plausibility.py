import numpy as np
import pandas as pd
import json


# Bone connections as (parent_id, child_id, bone_name)
# These are the only structurally meaningful lower body bones
BODY_BONES = [
    # Lower body
    (11, 13, 'left_thigh'),
    (12, 14, 'right_thigh'),
    (13, 15, 'left_shin'),
    (14, 16, 'right_shin'),
    # Upper body
    (5,  7,  'left_upper_arm'),
    (6,  8,  'right_upper_arm'),
    (7,  9,  'left_forearm'),
    (8,  10, 'right_forearm'),
    (5,  6,  'shoulder_width'),
    # Torso
    (5,  11, 'left_torso'),
    (6,  12, 'right_torso'),
    # Head
    (0,  3,  'nose_to_left_ear'),
    (0,  4,  'nose_to_right_ear'),
]

# Backwards-compat alias for any code that imports the old name


# Physiologically plausible bone length ratios relative to pelvis width.
# These are loose bounds — tighter bounds will produce more false positives
# on unusual poses (crouching, sitting, extreme angles).
# Format: (min_ratio, max_ratio) where ratio = bone_length / pelvis_width
# Derived from anthropometric data — adjust if your movies show unusual cases.
BONE_RATIO_BOUNDS = {
    # Lower body — empirical, all reasonable
    'left_thigh':         (1.65, 2.94),  # was (0.8, 2.8)
    'right_thigh':        (1.65, 2.94),  # mirror left; no empirical data
    'left_shin':          (1.36, 2.59),  # was (0.7, 2.5)
    'right_shin':         (0.85, 2.77),  # was (0.7, 2.5)
    
    # Upper body — empirical where available
    'left_upper_arm':     (0.76, 2.46),  # was (0.7, 2.6)
    'right_upper_arm':    (0.76, 2.46),  # mirror left; no empirical data for right_elbow
    'left_forearm':       (0.42, 1.88),  # using right_wrist data; left_wrist data was Ramona-biased
    'right_forearm':      (0.42, 1.88),  # was (0.6, 2.3)
    'shoulder_width':     (1.14, 2.18),  # was (0.8, 2.6)
    
    # Torso — empirical right side; tighten left to remove edge-case 5th percentile
    'left_torso':         (1.5, 5.13),   # left empirical 5th was 0.63, anatomically implausible
    'right_torso':        (2.04, 5.13),  # was (1.0, 3.5)
    
    # Head — no empirical, keep placeholders
    'nose_to_left_ear':   (0.2, 1.5),
    'nose_to_right_ear':  (0.2, 1.5),
}

H36M_TO_COCO = {
    1:  12,   # right_hip
    2:  14,   # right_knee
    3:  16,   # right_ankle
    4:  11,   # left_hip
    5:  13,   # left_knee
    6:  15,   # left_ankle
    11: 5,    # left_shoulder
    12: 7,    # left_elbow
    13: 9,    # left_wrist
    14: 6,    # right_shoulder
    15: 8,    # right_elbow
    16: 10,   # right_wrist
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

    for parent_id, child_id, bone_name in BODY_BONES:
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


 
# Geom columns that this module owns. They will be dropped from the input
# DataFrame and re-added with fresh values.
GEOM_COLS = ['geom_plausible', 'geom_flag', 'bone_length', 'bone_ratio']
 
 
def _build_film_cache(json_path):
    """
    Open one film's raw 133-kp JSON, run compute_plausibility_for_instance
    per (frame_id_0indexed, instance_id), and return a dict:
 
        {(frame_id, instance_id): {coco_joint_id: result_dict, ...}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
 
    cache = {}
    for frame in data.get('instance_info', []):
        # JSON frame_id is 1-indexed; CSVs store 0-indexed
        frame_id = int(frame['frame_id']) - 1
        for inst_idx, inst in enumerate(frame.get('instances', [])):
            kps    = inst.get('keypoints', [])
            scores = inst.get('keypoint_scores', [])
            cache[(frame_id, inst_idx)] = compute_plausibility_for_instance(
                kps, scores
            )
    return cache
 
 
def _normalize_geom_value(result):
    """
    result is what compute_plausibility_for_instance returns for one joint:
    {'geom_plausible': True/False/None, 'geom_flag': str/None, 'bone_length':
    float/None, 'bone_ratio': float/None}.
 
    Convert it into the four canonical CSV column values:
      geom_plausible: 1 / 0 / -1
      geom_flag:      string (never None)
      bone_length:    float or -1
      bone_ratio:     float or -1
    """
    if result is None:
        return -1, 'not_checked', -1, -1
 
    gp_raw = result.get('geom_plausible', None)
    if gp_raw is True:
        gp = 1
    elif gp_raw is False:
        gp = 0
    else:
        gp = -1
 
    flag = result.get('geom_flag') or 'ok'
 
    bl = result.get('bone_length')
    bl = float(bl) if bl is not None else -1
 
    br = result.get('bone_ratio')
    br = float(br) if br is not None else -1
 
    return gp, flag, bl, br
 
 
def recompute_geom(df, film_to_json_paths):
    """
    df: long-format annotated DataFrame with at least
        ['film', 'frame_id', 'instance_id', 'joint_id'] columns.
        joint_id is in H36M space.
 
    film_to_json_paths: dict film_name -> path to that film's raw 133-kp JSON.
 
    Returns: df with geom columns recomputed. Existing geom_plausible,
    geom_flag, bone_length, bone_ratio columns (if present) are dropped
    first so there is no contamination from old values.
 
    H36M joints that don't map to a single COCO source (root=0, spine=7,
    thorax=8, neck_base=9, head=10) get geom_plausible=-1, geom_flag=
    'derived_joint', bone_length=-1, bone_ratio=-1.
 
    Annotated rows whose film is not in film_to_json_paths get
    geom_flag='no_json' and -1 for all numeric values. A warning is
    printed listing missing films.
    """
    # 1. Drop existing geom columns to avoid mixing old and new values.
    df = df.drop(columns=[c for c in GEOM_COLS if c in df.columns])
 
    # 2. Sanity-check: are all annotated films covered by the JSON paths?
    annotated_films = set(df['film'].unique())
    missing = sorted(annotated_films - set(film_to_json_paths.keys()))
    if missing:
        print(f"[recompute_geom] WARNING: annotated films missing from "
              f"film_to_json_paths: {missing}")
        print(f"[recompute_geom]   rows for these films will get "
              f"geom_flag='no_json'.")
 
    # 3. Build per-film plausibility caches once (much cheaper than per-row).
    print(f"[recompute_geom] Building plausibility caches for "
          f"{len(film_to_json_paths)} films...")
    film_caches = {}
    for film, json_path in film_to_json_paths.items():
        if film not in annotated_films:
            # Don't waste time on films not in this DataFrame
            continue
        try:
            film_caches[film] = _build_film_cache(json_path)
            print(f"[recompute_geom]   {film}: cached "
                  f"{len(film_caches[film])} (frame, instance) entries")
        except Exception as e:
            print(f"[recompute_geom]   {film}: FAILED to load JSON "
                  f"({json_path}): {e}")
 
    # 4. Walk df rows; look up cached geom per (film, frame, instance, h36m_id)
    n = len(df)
    gp_col   = np.full(n, -1, dtype=np.int8)
    bl_col   = np.full(n, -1, dtype=float)
    br_col   = np.full(n, -1, dtype=float)
    flag_col = np.empty(n, dtype=object)
 
    df_reset = df.reset_index(drop=True)
    derived_count = 0
    no_json_count = 0
    miss_count    = 0
    hit_count     = 0
 
    for i in range(n):
        film     = df_reset.at[i, 'film']
        frame_id = int(df_reset.at[i, 'frame_id'])
        inst_id  = int(df_reset.at[i, 'instance_id'])
        h36m_id  = int(df_reset.at[i, 'joint_id'])
 
        cache = film_caches.get(film)
        if cache is None:
            flag_col[i] = 'no_json'
            no_json_count += 1
            continue
 
        # Map H36M -> COCO. Derived H36M ids have no direct COCO source.
        if h36m_id not in H36M_TO_COCO:
            flag_col[i] = 'derived_joint'
            derived_count += 1
            continue
 
        coco_id = H36M_TO_COCO[h36m_id]
        frame_results = cache.get((frame_id, inst_id))
        if frame_results is None:
            # frame/instance not present in JSON (e.g., dropped by tracker)
            flag_col[i] = 'frame_not_in_json'
            miss_count += 1
            continue
 
        result = frame_results.get(coco_id)
        if result is None:
            # COCO joint not checked by current BODY_BONES
            flag_col[i] = 'joint_not_in_bones'
            miss_count += 1
            continue
 
        gp, flag, bl, br = _normalize_geom_value(result)
        gp_col[i]   = gp
        bl_col[i]   = bl
        br_col[i]   = br
        flag_col[i] = flag
        hit_count += 1
 
    df_reset['geom_plausible'] = gp_col
    df_reset['geom_flag']      = flag_col
    df_reset['bone_length']    = bl_col
    df_reset['bone_ratio']     = br_col
 
    print(f"[recompute_geom] Done: {hit_count} rows got real geom values, "
          f"{derived_count} were derived joints (no direct COCO source), "
          f"{no_json_count} had no JSON for their film, "
          f"{miss_count} missed in cache (frame/joint not present).")
 
    return df_reset