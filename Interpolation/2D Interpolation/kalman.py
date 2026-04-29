import numpy as np
import pandas as pd
from kalman import KalmanFilter

#Source: https://www.geeksforgeeks.org/python/kalman-filter-in-python/

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        
    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x




dt = 1.0

F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])

B = np.zeros((4, 1))
u = np.zeros((1, 1))

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

Q = np.diag([0.1, 0.1, 1.0, 1.0])

R_base_trust   = np.diag([5.0,  5.0])
R_base_partial = np.diag([50.0, 50.0])


def reliability_to_R(reliability_int, prob_unreliable):
    if reliability_int == 0:  # reliable
        scale = 1.0
        return R_base_trust * scale
    elif reliability_int == 1:  # partial — but in your new encoding this is unreliable
        # higher prob_unreliable -> larger R -> filter trusts observation less
        scale = 1.0 / (1.0 - prob_unreliable + 0.05)
        return R_base_partial * scale
    else:
        return R_base_partial * 100.0


def apply_kalman_to_group(group_df):
    """
    Apply Kalman filter to one (film, instance_id, joint_name) group.
    group_df must be sorted by frame_id.
    Only processes trust (0) and partial_trust (1) joints.
    Returns a copy with new columns: x_kalman, y_kalman.
    """
    group_df = group_df.copy().sort_values('frame_id').reset_index(drop=False)

    x_kalman = group_df['x'].values.copy().astype(float)
    y_kalman = group_df['y'].values.copy().astype(float)

    # Find first trust frame to initialise
    trust_mask = group_df['reliability_category_int'].isin([0, 1])
    if not trust_mask.any():
        # No trust frames at all — nothing to filter, return as-is
        group_df['x_kalman'] = x_kalman
        group_df['y_kalman'] = y_kalman
        return group_df.set_index('index')

    first_trust_idx = trust_mask.idxmax()  # first True index

    # Initialise state from first trust frame
    init_x   = group_df.loc[first_trust_idx, 'x']
    init_y   = group_df.loc[first_trust_idx, 'y']
    init_vx  = group_df.loc[first_trust_idx, 'x_velocity'] \
                if 'x_velocity' in group_df.columns else 0.0
    init_vy  = group_df.loc[first_trust_idx, 'y_velocity'] \
                if 'y_velocity' in group_df.columns else 0.0

    # Replace NaN velocities with 0
    init_vx = 0.0 if np.isnan(init_vx) else init_vx
    init_vy = 0.0 if np.isnan(init_vy) else init_vy

    x0 = np.array([[init_x],
                   [init_y],
                   [init_vx],
                   [init_vy]])   # shape (4, 1)

    P0 = np.eye(4) * 10.0

    kf = KalmanFilter(F, B, H, Q, R_base_trust, x0, P0)

    # Run forward through all rows
    for i, row in group_df.iterrows():
        rel = row['reliability_category_int']

        # Skip rows before the first trust frame — no state yet
        if i < first_trust_idx:
            continue

        # Predict step — always advance the filter
        predicted = kf.predict(u)

        if rel in [0, 1]:  # trust or partial_trust — update with observation
            prob_unreliable = row.get('prob_unreliable', 0.5 if rel == 1 else 0.9)
            R_current  = reliability_to_R(rel, prob_unreliable)
            kf.R       = R_current

            z = np.array([[row['x']],
                          [row['y']]])   # shape (2, 1)
            updated = kf.update(z)

            x_kalman[i] = updated[0, 0]
            y_kalman[i] = updated[1, 0]

        else:
            # dont_trust — do not update, use prediction only
            # but do NOT write prediction here either —
            # cubic spline handles dont_trust gap filling separately
            pass

    group_df['x_kalman'] = x_kalman
    group_df['y_kalman'] = y_kalman
    return group_df.set_index('index')


def apply_kalman_filter(df):
    """
    Apply Kalman filter across all (film, instance_id, joint_name) groups.
    Adds x_kalman and y_kalman columns to df.
    dont_trust joints are left at their original MMPose values here —
    cubic spline interpolation handles them in a separate pass.
    """
    results = []
    for (film, instance_id, joint_name), group in df.groupby(
            ['film', 'instance_id', 'joint_name']):
        filtered = apply_kalman_to_group(group)
        results.append(filtered)

    return pd.concat(results).sort_index()