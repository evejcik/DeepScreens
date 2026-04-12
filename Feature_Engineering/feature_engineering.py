#input: annotated csv 
#output: 



import pandas as pd
import numpy as np
import argparse
import os
import glob

#1. encode reliability category as integer: trust = 0, partial trust = 1, don't trust = 2
#2. encode joint_name as integer (0 - 16), using H36M mapping from mmpose
#3. handle missing values for 'valid' -> 0 if 0 else 1
#4. sort films by (film, instance_id, joint_name) -> make sure to not mix joints or tracks when computing temporal windows!
#5. group by: (film, instance_id, joint_name), sort by frame_id:
    #a. confidence_mean_wk = rolling mean of confidence_scores over a +-k frame window
    #b confidence_std
    #c. position_velocity = Euclidean distance between (x,y) at frame t and t-1
    #d. position_acceleration: change in velocity from t-1 to t
    #e. position_std_wk = rolling std of (x,y) over +-k frames
    #f. position_variance_wk 
    #g. frames_since_trust


RTMW_TO_H36M_ID = {
    "right_hip":       1,
    "right_knee":      2,
    "right_ankle":     3,
    "left_hip":        4,
    "left_knee":       5,
    "left_ankle":      6,
    "left_shoulder":   11,
    "left_elbow":      12,
    "left_wrist":      13,
    "right_shoulder":  14,
    "right_elbow":     15,
    "right_wrist":     16,
    # computed joints (spine, thorax, etc.) can't be mapped 1:1 from a name alone
    }

RTMW_NAME2SRC_ID = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

def csv_stack(path_to_csvs: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(path_to_csvs, "*.csv"))
    if not files:
        raise ValueError(f"No CSVs found in {path_to_csvs}")

    dfs = []

    for f in files:
        filename = os.path.basename(f)
        film_name = "_".join(filename.split(" ")[2:4])
        csv = pd.read_csv(f)
        csv['film'] = film_name
        dfs.append(csv)
    # df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # filename
    df = pd.concat(dfs, ignore_index = True)
    return df

def reliability_int(df):
    trust_map = {
        "trust" : 0,
        "don't trust" : 2,
        "partial trust" : 1,
        "ambiguous" : 3
    }

    # see every unique raw value coming in before mapping
    print(df['reliability_category'].unique())

    # see how many NaNs exist before mapping vs after
    print(f"NaNs before mapping: {df['reliability_category'].isna().sum()}")
    

    df['reliability_category_int'] = df['reliability_category'].map(trust_map)
    df = df.dropna(subset = 'reliability_category')
    print(f"NaNs after mapping: {df['reliability_category_int'].isna().sum()}")

    mask = df['reliability_category_int'].isna()
    if mask.any():
        errors_df = df.loc[mask, ['film', 'frame_id', 'reliability_category', 'reliability_category_int']]

        pd.set_option('display.max_rows', None)

        print(f"Unmapped reliability values:\n{errors_df}")
        # print(df['reliability_category'].unique())
        pd.reset_option('display.max_rows')


    print(f"Count distribution:\n{df['reliability_category_int'].value_counts()}")
    return df

def joint_mapping(df):

    df["joint_id"] = df["joint_name"].map(RTMW_TO_H36M_ID)  # NaN for unmapped
    unmapped = df[df["joint_id"].isna()]["joint_name"].unique()
    if len(unmapped):
        print(f"Warning: unmapped joints: {unmapped}")
    return df

def convert_rtmw_to_h36m17(coords: np.ndarray) -> np.ndarray:
    """
    coords: (133, 2) or (133, 3) array of RTMW keypoints
    returns: (17, coords.shape[1]) H36M keypoints
    """
    print(f"Coords ndim: {coords.ndim}, Coords shape: {coords.shape}")
    assert coords.ndim == 2 and coords.shape[0] >= 17

    NAME2ID = RTMW_TO_H36M_ID  # reuse the dict above, only the 1:1 joints

    def get(name):
        return coords[NAME2ID_RTMW[name]]  

    ls, rs = get("left_shoulder"), get("right_shoulder")
    lh, rh = get("left_hip"),      get("right_hip")
    le, re = get("left_ear"),       get("right_ear")

    hip_mid      = (lh + rh) / 2
    shoulder_mid = (ls + rs) / 2
    spine        = shoulder_mid - hip_mid
    neck_vec     = ((le + re) / 2) - shoulder_mid

    h36m = np.zeros((17, coords.shape[1]), dtype=np.float32)
    h36m[0]  = hip_mid
    h36m[1]  = get("right_hip")
    h36m[2]  = get("right_knee")
    h36m[3]  = get("right_ankle")
    h36m[4]  = get("left_hip")
    h36m[5]  = get("left_knee")
    h36m[6]  = get("left_ankle")
    h36m[7]  = hip_mid + 0.5 * spine       # spine midpoint
    h36m[8]  = shoulder_mid                 # thorax
    h36m[9]  = shoulder_mid + 0.15 * neck_vec  # neck base
    h36m[10] = (le + re) / 2               # head
    h36m[11] = get("left_shoulder")
    h36m[12] = get("left_elbow")
    h36m[13] = get("left_wrist")
    h36m[14] = get("right_shoulder")
    h36m[15] = get("right_elbow")          # was wrong in your original
    h36m[16] = get("right_wrist")

    return h36m

def build_model_input(df: pd.DataFrame, required_joints: int = 12) -> dict:
    """
    Returns dict of {(film, track_id): (T, 17, 2)} arrays
    Skips frames with fewer than required_joints annotated.
    """
    result = {}
    
    for (film, instance_id), track_group in df.groupby(['film', 'instance_id']):
        frames = []
        for frame_id, frame_group in track_group.groupby('frame_id'):
            if len(frame_group) < required_joints:
                continue
            frame_group = frame_group.sort_values('joint_id')
            coords = frame_group[["x", "y"]].to_numpy()
            frames.append(coords)
        
        if frames:
            result[(film, instance_id)] = np.stack(frames, axis=0)  # (T, n_joints, 2)
    
    return result

def is_valid(df):
    df['valid'] = df['valid'].apply(lambda x: 0 if x == 0 else 1)
    return df

def data_loader(csv_path):
    df = csv_stack(csv_path)
    print(f"Columns: {df.columns}")
    # df = pd.read_csv(csv_path)
    # print(f"Type df csv: {type(df)}")
    df = reliability_int(df)
    # print(f"Type df reliability_int: {type(df)}")
    df = joint_mapping(df)
    # print(f"Type df joint_mapping: {type(df)}")
    df = is_valid(df)

    # print(f"Type df: {type(df)}")
    print(df.columns)

    # print(df.head(5))
    df = df.sort_values(['film','instance_id', 'joint_name', 'frame_id'])
    # df = df.groupby(by = ['track_id', 'joint_name'])

    # df_pd = pd.DataFrame(df)
    # print(df.first())
    print(df.iloc[:5, :])

    # df = df_pd.sort_values(['track_id', 'joint_name', 'frame_id'])
    
    # X = build_model_input(df)  # (T, 17, 2)
    return df

def confidence_mean_rolling(df, k):
    #we want by: film, instance, frame, joint
    df = df.sort_values(['film', 'instance_id', 'joint_name', 'frame_id'])
    df['confidence_mean_wk'] = (df.groupby(['film', 'instance_id', 'joint_id'])['mmpose_confidence'].transform(
        lambda x: x.rolling(window=2*k+1, center=True, min_periods=1).mean())
        )
    
    # print(rolling_confidence)
    return df

 def confidence_std_rolling(df, k):
    #we want by: film, instance, frame, joint
    df = df.sort_values(['film', 'instance_id', 'joint_name', 'frame_id'])
    df['confidence_std_wk'] = (df.groupby(['film', 'instance_id', 'joint_id'])['mmpose_confidence'].transform(
        lambda x: x.rolling(window=2*k+1, center=True, min_periods=1).std())
        )
    
    # print(rolling_confidence)
    return df

def position_mean_rolling(df, k):
    df = df.sort_values(['film', 'instance_id', 'joint_name', 'frame_id'])
    df['position_mean_x_wk'] = df.groupby(['film', 'instance_id', 'joint_id'])['x'].transform(
        lambda x: x.rolling(window=2*k+1, center=True, min_periods=1).mean()
    )
    
    df['position_mean_y_wk'] = df.groupby(['film', 'instance_id', 'joint_name'])['y'].transform(
        lambda y: y.rolling(window=2*k+1, center=True, min_periods=1).mean()
    )

    return df

def position_std_rolling(df,k):
    df = df.sort_values(['film', 'instance_id', 'joint_name', 'frame_id'])
    df['position_std_x_wk'] = df.groupby(['film', 'instance_id', 'joint_id'])['x'].transform(
        lambda x: x.rolling(window=2*k+1, center=True, min_periods=1).std()
    )
    
    df['position_std_y_wk'] = df.groupby(['film', 'instance_id', 'joint_name'])['y'].transform(
        lambda y: y.rolling(window=2*k+1, center=True, min_periods=1).std()
    )

    return df

def position_velocity(df):
    #Euclidean distance between (x,y) at frame t and t-1

    df['x_velocity'] = df.groupby(['film', 'instance_id', 'joint_name'])['x'].transform(
        lambda x : x.diff()
    )
    df['y_velocity'] = df.groupby(['film', 'instance_id', 'joint_name'])['y'].transform(
        lambda y : y.diff()
    )
    
    df['position_velocity'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)

    return df

def position_acceleration(df):
    df['acceleration'] = df.groupby(['film', 'instance_id', 'joint_name'])['position_velocity'].transform(
        lambda x : x.diff()
    )

    return df

def frames_since_trust(df):
    results = {}
    for (film, instance_id, joint_name), group in df.groupby(['film', 'instance_id', 'joint_name']):
        frames_since_trust = -1
        for idx, row in group.itterows():
            frames_since_trust = 0 if row['reliability_category_int'] == 0 else frames_since_trust +=1 if frame_since_trust >= 0 else frames_since_trust = -1
            results[idx] = frames_since_trust
    df['frames_since_trust'] = pd.Series(results)


def main(csv,k):
    df = data_loader(csv)
    df = confidence_mean_rolling(df, k)
    df = position_mean_rolling(df, k)
    df = position_std_rolling(df,k)
    df = position_velocity(df)
    df = position_acceleration(df)
    df = frames_since_trust(df)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default = "../ANNOTATED_CSVS")
    ap.add_argument("--k", type = int, default = 5)


    args = ap.parse_args()
    main(
        args.csv,
        args.k
    )
