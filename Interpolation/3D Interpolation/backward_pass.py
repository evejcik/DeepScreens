import numpy as np
import pandas as pd

# Joints that define a "reliable whole-body detection"
# Both hips + at least one knee must be trust or high-confidence partial_trust
HIP_JOINTS    = {'left_hip', 'right_hip'}
KNEE_JOINTS   = {'left_knee', 'right_knee'}

# Annotator confidence threshold for partial_trust to count as "high-confidence"
# annotator_confidence in your data is: 'certain', 'fairly sure', 'unsure'
HIGH_CONF_VALUES = {'certain', 'fairly sure'}


def is_reliable(row):
    """
    A single joint row is reliable for initialisation purposes if:
    - It is trust (0), OR
    - It is partial_trust (1) AND annotator_confidence is high
    """
    if row['reliability_category_int'] == 0:
        return True
    if row['reliability_category_int'] == 1:
        return row.get('annotator_confidence') in HIGH_CONF_VALUES
    return False


def find_first_reliable_whole_body_frame(track_df):
    """
    Given a dataframe for one (film, instance_id) track (all joints, all frames),
    find the earliest frame_id where:
        - both left_hip and right_hip are reliable
        - at least one of left_knee or right_knee is reliable

    Returns the frame_id (int) or None if no such frame exists.
    """
    # Get all unique frame_ids sorted
    frame_ids = sorted(track_df['frame_id'].unique())

    for frame_id in frame_ids:
        frame = track_df[track_df['frame_id'] == frame_id]

        # Build a dict of joint_name -> row for this frame
        joint_rows = {row['joint_name']: row
                      for _, row in frame.iterrows()}

        # Check both hips present and reliable
        hips_ok = all(
            jn in joint_rows and is_reliable(joint_rows[jn])
            for jn in HIP_JOINTS
        )
        if not hips_ok:
            continue

        # Check at least one knee present and reliable
        knee_ok = any(
            jn in joint_rows and is_reliable(joint_rows[jn])
            for jn in KNEE_JOINTS
        )
        if not knee_ok:
            continue

        return frame_id

    return None  # no reliable whole-body frame found in this track


def get_pose_at_frame(track_df, frame_id):
    """
    Extract the cleaned position (x_filled, y_filled) for every joint
    at a given frame_id within a track.

    Returns a dict of {joint_name: (x, y)}.
    Falls back to raw (x, y) if x_filled not yet computed.
    """
    frame_rows = track_df[track_df['frame_id'] == frame_id]
    pose = {}
    for _, row in frame_rows.iterrows():
        x = row['x_filled'] if 'x_filled' in row and not pd.isna(row['x_filled']) \
            else row['x']
        y = row['y_filled'] if 'y_filled' in row and not pd.isna(row['y_filled']) \
            else row['y']
        pose[row['joint_name']] = (float(x), float(y))
    return pose


def apply_backward_pass(df):
    """
    For each (film, instance_id) track:
      1. Find the first reliable whole-body frame (both hips + one knee reliable).
      2. For all frames BEFORE that frame, propagate the reliable pose backwards —
         i.e. freeze every joint at its position from the first reliable frame.
      3. Mark propagated frames with x_filled, y_filled and
         keypoint_interpolated = True.

    Requires x_filled and y_filled columns to exist (from Kalman + cubic spline passes).
    If they do not exist yet, falls back to raw x, y.

    Modifies df in place and returns it.
    """
    # Ensure columns exist
    if 'x_filled' not in df.columns:
        df['x_filled'] = df['x'].astype(float)
    if 'y_filled' not in df.columns:
        df['y_filled'] = df['y'].astype(float)
    if 'keypoint_interpolated' not in df.columns:
        df['keypoint_interpolated'] = False

    results = []

    for (film, instance_id), track_df in df.groupby(['film', 'instance_id']):
        track_df = track_df.copy().sort_values('frame_id')

        first_reliable_frame = find_first_reliable_whole_body_frame(track_df)

        if first_reliable_frame is None:
            # No reliable whole-body frame exists in this track at all.
            # Cannot initialise — leave as-is and warn.
            print(f"WARNING: No reliable whole-body frame found for "
                  f"film={film}, instance_id={instance_id}. "
                  f"Backward pass skipped for this track.")
            results.append(track_df)
            continue

        # Get the anchor pose from the first reliable frame
        anchor_pose = get_pose_at_frame(track_df, first_reliable_frame)

        if not anchor_pose:
            print(f"WARNING: Could not extract pose at frame {first_reliable_frame} "
                  f"for film={film}, instance_id={instance_id}.")
            results.append(track_df)
            continue

        # Identify all frames that need backward filling
        frames_before = sorted(
            track_df[track_df['frame_id'] < first_reliable_frame]['frame_id'].unique()
        )

        if not frames_before:
            # Track starts at or after the first reliable frame — nothing to fill
            results.append(track_df)
            continue

        print(f"film={film}, instance_id={instance_id}: "
              f"first reliable frame={first_reliable_frame}, "
              f"propagating backwards over {len(frames_before)} frames "
              f"({frames_before[0]}–{frames_before[-1]})")

        # Propagate anchor pose to all frames before first_reliable_frame
        for frame_id in frames_before:
            frame_mask = (track_df['frame_id'] == frame_id)
            frame_rows = track_df[frame_mask]

            for idx, row in frame_rows.iterrows():
                joint_name = row['joint_name']

                if joint_name in anchor_pose:
                    ax, ay = anchor_pose[joint_name]
                    track_df.loc[idx, 'x_filled']             = ax
                    track_df.loc[idx, 'y_filled']             = ay
                    track_df.loc[idx, 'keypoint_interpolated'] = True
                else:
                    # Joint not present in anchor frame — hold raw MMPose value
                    # and flag as interpolated so downstream knows it is uncertain
                    track_df.loc[idx, 'keypoint_interpolated'] = True

        results.append(track_df)

    return pd.concat(results).sort_index()