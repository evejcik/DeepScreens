#input: annotated csv 
#output: 



import pandaas as pd
import numpy as np
import argparse

#1. encode reliability category as integer: trust = 0, partial trust = 1, don't trust = 2
#2. encode joint_name as integer (0 - 16), using H36M mapping from mmpose
#3. handle missing values for 'valid' -> 0 if 0 else 1
#4. sort films by (film, track_id, joint_name) -> make sure to not mix joints or tracks when computing temporal windows!
#5. group by: (film, track_id, joint_name), sort by frame_id:
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

def reliability_int(df):
    trust_map = {
        "trust" : 0,
        "don't trust" : 2,
        "partial trust" : 1
    }

    df['reliability_category_int'] = df['reliability_category'].map(trust_map)

    mask = df['reliability_category_int'].isna()
    errors_df = df.loc[mask, ['reliability_category', 'reliability_category_int']]

    print(f"Count distribution: {df['reliability_category_int'].nunique()}")
    print(df.loc[df['reliability_category_int']].isna(), 'reliability_category')

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
    assert coords.ndim == 2 and coords.shape[0] >= 17

    NAME2ID = RTMW_TO_H36M_ID  # reuse the dict above, only the 1:1 joints

    def get(name):
        return coords[NAME2ID_RTMW[name]]  # your source index from mmpose json

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

def data_loader(csv_path):
    df = pd.read(csv_path)
    df = reliability_int(df)
    df = joint_mapping(df)

    df['h36m17_x_y_coords'] = convert_rtmw_to_h36m17(zip(df['x'], df['y']))



def main(csv, json_path):
    df = data_loader(csv)

    



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--json", default = '/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/ramona-demo-clip copy 2/ramona-demo-clip/segment_1_1639.json')

    args = ap.parse_args()
    main(
        args.csv, 
        args.json
    )
