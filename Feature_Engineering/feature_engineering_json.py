
import pandas as pd
import numpy as np
import argparse
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import json
from geometric_plausibility import compute_plausibility_for_instance, compute_boundary_distance


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

H36M_JOINT_NAMES = {
        0:'root', 1:'right_hip', 2:'right_knee', 3:'right_ankle',
        4:'left_hip', 5:'left_knee', 6:'left_ankle', 7:'spine',
        8:'thorax', 9:'neck_base', 10:'head', 11:'left_shoulder',
        12:'left_elbow', 13:'left_wrist', 14:'right_shoulder',
        15:'right_elbow', 16:'right_wrist'
    }

# film_map = {
#i think, get this from when we stack all of the csvs together
# }

def load_json(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)

def json_to_csv(json, film: str):
    #inputs: film -> film name, should match film name in Feature Engineering/Long Data.csv
    #json -> total unnannotated 133 keypoint json, directly from initial MMPose run

    data = load_json(json)
    # frame_map = build_frame_map(load_json(args.json))
    keypoint_id2name = data['meta_info']['keypoint_id2name']

    rows = []

    for instance_id, instance  in enumerate(instances):
        kps   = instance.get('keypoints', [])
        confs = instance.get('keypoint_scores', [])
        for joint_id, keypoint  in enumerate(keypoints):
            joint_name = keypoint_id2name.get(str(joint_id))

            
            geom_plausibility = compute_plausibility_for_instance(kps, confs)

            x = keypoint[0]
            y = keypoint[1]

            track_id = -1
            mmpose_confidence = -1
            reason_for_distrust = annotator_confidence = reliability_category_int = \
                frames_since_trust = frames_since_dont_trust = frac_trust_wk = \
                frac_partial_wk = frac_dont_trust_wk = film_id = -1

            row = [frame_id, 
                    track_id, 
                    instance_id,  
                    joint_id,
                    joint_name, 
                    x, y,
                    mmpose_confidence,
                    reason_for_distrust,
                    annotator_confidence,
                    dist_to_boundary,
                    valid,
                    geom_plausible,
                    geom_flag,
                    bone_length,
                    bone_ratio,
                    film, 
                    reliability_category_int,
                    confidence_std_wk,
                    position_mean_x_wk,
                    position_mean_y_wk,
                    position_std_x_wk,
                    position_std_y_wk,
                    position_velocity,
                    position_acceleration,
                    frames_since_trust,
                    frames_since_dont_trust,
                    frac_trust_wk,
                    frac_partial_wk,
                    frac_dont_trust_wk,
                    film_id]

            rows.append[row]

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_id", 
                    "track_id", 
                    "instance_id",  
                    "joint_id",
                    "joint_name", 
                    "x", "y",
                    "mmpose_confidence",
                    "reason_for_distrust",
                    "annotator_confidence",
                    "dist_to_boundary",
                    "valid",
                    "geom_plausible",
                    "geom_flag",
                    "bone_length",
                    "bone_ratio",
                    "film", 
                    "reliability_category_int",
                    "confidence_std_wk",
                    "position_mean_x_wk",
                    "position_mean_y_wk",
                    "position_std_x_wk",
                    "position_std_y_wk",
                    "position_velocity",
                    "position_acceleration",
                    "frames_since_trust",
                    "frames_since_dont_trust",
                    "frac_trust_wk",
                    "frac_partial_wk",
                    "frac_dont_trust_wk",
                    "film_id" 
                        ])
        writer.writerows(rows)

    return rows


#once we have the csv we can compute the regular features
#no reliability int

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
    df['position_acceleration'] = df.groupby(['film', 'instance_id', 'joint_name'])['position_velocity'].transform(
        lambda x : x.diff()
    )

    return df

def frames_since_trust(df):
    results = {}
    for (film, instance_id, joint_name), group in df.groupby(['film', 'instance_id', 'joint_name']):
        frames_since_trust = -1
        for idx, row in group.iterrows():
            frames_since_trust = 0 if row['reliability_category_int'] == 0 else frames_since_trust + 1 if frames_since_trust >= 0 else -1
            results[idx] = frames_since_trust
    df['frames_since_trust'] = pd.Series(results)
    return df

def frames_since_dont_trust(df):
    results = {}
    for (film, instance_id, joint_name), group in df.groupby(['film', 'instance_id', 'joint_name']):
        frames_since_dont_trust = -1
        for idx, row in group.iterrows():
            frames_since_dont_trust = 0 if row['reliability_category_int'] == 2 else frames_since_dont_trust + 1 if frames_since_dont_trust >= 0 else -1
            results[idx] = frames_since_dont_trust
    df['frames_since_dont_trust'] = pd.Series(results)
    return df

def window_fractions(df, k = 5):
    df = df.sort_values(['film', 'instance_id', 'joint_name', 'frame_id'])
    
    # create binary indicator for each class
    df['is_trust']       = (df['reliability_category_int'] == 0).astype(int)
    df['is_partial']     = (df['reliability_category_int'] == 1).astype(int)
    df['is_dont_trust']  = (df['reliability_category_int'] == 2).astype(int)
    
    grp = ['film', 'instance_id', 'joint_name']
    window = 2*k + 1
    
    for col, out in [
        ('is_trust',      'frac_trust_wk'),
        ('is_partial',    'frac_partial_wk'),
        ('is_dont_trust', 'frac_dont_trust_wk'),
    ]:
        df[out] = df.groupby(grp)[col].transform(
            lambda x: x.rolling(window=window, center=True, min_periods=1).mean()
        )
    
    df = df.drop(columns=['is_trust', 'is_partial', 'is_dont_trust'])
    return df

def re_labeling_rules(df):
    df = sandwiched_partial_trust(df)

def film_int_encoding(df):
    film_map = {film: idx for idx, film in enumerate(df['film'].unique())}

    df['film_id'] = df['film'].map(film_map)

    print(f"UNIQUE FILMS: {df['film'].unique()}")
    print(f"UNIQUE FILMS IDS: {df['film_id'].unique()}")
    return df

def geom_accuracy(df):
    df['geometric_plausibility'] = 
