import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


VALIDITY_HORIZON = 20  # tune this — max frames across which you will interpolate


def apply_cubic_spline_to_group(group_df, validity_horizon=VALIDITY_HORIZON):
    """
    Fill dont_trust gaps using cubic spline interpolation between
    nearest Kalman-smoothed trust/partial_trust anchor frames.

    Requires x_kalman and y_kalman columns to already exist (from Kalman pass).
    Adds x_filled and y_filled columns — these are the final cleaned coordinates.

    Falls back to hold-last-known for gaps beyond validity_horizon.
    """
    group_df = group_df.copy().sort_values('frame_id').reset_index(drop=False)

    # Start from Kalman output for trust/partial, raw MMPose for dont_trust
    x_filled = group_df['x_kalman'].values.copy().astype(float)
    y_filled = group_df['y_kalman'].values.copy().astype(float)

    rel      = group_df['reliability_category_int'].values
    frames   = group_df['frame_id'].values
    n        = len(group_df)

    # Boolean mask of anchor frames (trust or partial_trust)
    is_anchor = rel == 0

    # Walk through and find dont_trust runs
    i = 0
    while i < n:
        if rel[i] == 1:  # start of a dont_trust run
            gap_start = i

            # Find end of this run
            j = i
            while j < n and rel[j] == 1:
                j += 1
            gap_end = j - 1  # last dont_trust frame index (inclusive)

            # Find left anchor: nearest anchor at or before gap_start
            left_anchor = None
            for k in range(gap_start - 1, -1, -1):
                if is_anchor[k]:
                    left_anchor = k
                    break

            # Find right anchor: nearest anchor at or after gap_end
            right_anchor = None
            for k in range(gap_end + 1, n):
                if is_anchor[k]:
                    right_anchor = k
                    break

            gap_size = gap_end - gap_start + 1

            if (left_anchor is not None and
                right_anchor is not None and
                gap_size <= validity_horizon):

                # Both anchors exist and gap is within horizon — use cubic spline
                anchor_indices = [left_anchor, right_anchor]

                # If there are any trust frames just outside the gap,
                # include one more on each side for better curve shape
                if left_anchor > 0 and is_anchor[left_anchor - 1]:
                    anchor_indices = [left_anchor - 1] + anchor_indices
                if right_anchor < n - 1 and is_anchor[right_anchor + 1]:
                    anchor_indices = anchor_indices + [right_anchor + 1]

                anchor_frames = frames[anchor_indices]
                anchor_x      = x_filled[anchor_indices]
                anchor_y      = y_filled[anchor_indices]

                # Fit spline through anchors
                cs_x = CubicSpline(anchor_frames, anchor_x)
                cs_y = CubicSpline(anchor_frames, anchor_y)

                # Fill in the dont_trust frames
                gap_frame_ids = frames[gap_start:gap_end + 1]
                x_filled[gap_start:gap_end + 1] = cs_x(gap_frame_ids)
                y_filled[gap_start:gap_end + 1] = cs_y(gap_frame_ids)

            elif left_anchor is not None:
                # No right anchor within horizon, or gap too large
                # Hold last known trust position
                x_filled[gap_start:gap_end + 1] = x_filled[left_anchor]
                y_filled[gap_start:gap_end + 1] = y_filled[left_anchor]

            else:
                # No left anchor either — track starts with dont_trust
                # Leave as raw MMPose until first trust frame appears
                # (backward pass initialisation handles this separately)
                pass

            i = gap_end + 1  # jump past this gap

        else:
            i += 1

    group_df['x_filled'] = x_filled
    group_df['y_filled'] = y_filled
    return group_df.set_index('index')


def apply_cubic_spline(df, validity_horizon=VALIDITY_HORIZON):
    """
    Apply cubic spline gap filling across all (film, instance_id, joint_name) groups.
    Requires Kalman pass to have already run (x_kalman, y_kalman columns must exist).
    Adds x_filled and y_filled columns — use these as your final cleaned coordinates.
    """
    results = []
    for (film, instance_id, joint_name), group in df.groupby(
            ['film', 'instance_id', 'joint_name']):
        filled = apply_cubic_spline_to_group(group, validity_horizon)
        results.append(filled)

    return pd.concat(results).sort_index()