
import pandas as pd
import numpy as np
import argparse
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import json
import csv
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
    keypoint_id2name = data['meta_info']['keypoint_id2name']

    rows = []

    for frame_entry in data['instance_info']:
        frame_id = frame_entry['frame_id']
        for instance_id, instance in enumerate(frame_entry['instances']):
            kps   = instance.get('keypoints', [])
            confs = instance.get('keypoint_scores', [])

            geom_results = compute_plausibility_for_instance(kps, confs)

            for joint_id, keypoint in enumerate(kps):
                joint_name = keypoint_id2name.get(str(joint_id))

                x = keypoint[0]
                y = keypoint[1]

                geom = geom_results.get(joint_id, {})
                geom_plausible = geom.get('geom_plausible', None)
                geom_flag      = geom.get('geom_flag', 'not_checked')
                bone_length    = geom.get('bone_length', None)
                bone_ratio     = geom.get('bone_ratio', None)

                track_id = -1
                mmpose_confidence = -1

                position_mean_x_wk = -1
                position_mean_y_wk = -1
                position_std_x_wk = -1
                position_std_y_wk = -1
                position_velocity = -1
                position_acceleration = -1

                reason_for_distrust = annotator_confidence = reliability_category_int = \
                    frames_since_trust = frames_since_dont_trust = frac_trust_wk = \
                    frac_partial_wk = frac_dont_trust_wk = film_id = confidence_std_wk = \
                    dist_to_boundary = valid = -1

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

                rows.append(row)

    return rows

def frame_width_height(json):
    data = load_json(json)
    video_shape = data['video_info'][0]['video_shape']
    video_width = video_shape[0]
    video_height = video_shape[1]
    return video_width, video_height

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

def data_checking(df):
    # print(f"Reliability counts: {df['reliability_category_int'].value_counts()}")
    print(f"Final columns: {df.columns}")
    print(f"Shape final df: {df.shape}")

    # print(f"Max frames since trust: {df['frames_since_trust'].max()}")

    # print(f"Frames more than 20 away from a trust: {(df['frames_since_trust'] > 20).sum()}")
    # print(f"Percentage of frames more than 20 away from a trust: {(df['frames_since_trust'] > 20).sum()/df.shape[0]}")

    print(f"Geom plausible values: {df['geom_plausible'].value_counts(dropna=False)}")

    feature_cols = [
        'reliability_category_int',
        'mmpose_confidence',
        # 'confidence_mean_wk',
        'confidence_std_wk', 
        'position_velocity',
        'position_acceleration',
        'position_std_x_wk',
        'position_std_y_wk'
        # 'frames_since_trust'
    ]

    corr_matrix = df[feature_cols].corr()
    print(corr_matrix)


    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')

    # print(df[df['joint_name'] == 'right_hip']['joint_id'].value_counts())
    # print(df[df['joint_name'] == 'left_elbow']['joint_id'].value_counts())
    # print(df[df['joint_name'].isin([7, 12])][['joint_name', 'joint_id', 'film']].head(20))
    print(df['joint_name'].unique())
    pd.set_option('display.max_columns', None)
        
    # print(df.groupby('film')['joint_name'].unique())
    pd.reset_option('display.max_columns')
    # print(df.groupby(['film', 'joint_name']).size().unstack(fill_value=0))

    print(f"NULL VALUES: {df.isnull().sum()[df.isnull().sum() > 0]}")

    unmapped = df[df['film_id'].isna()]['film'].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Films in df not found in annotated CSV: {unmapped}")

def film_int_encoding(csv_annotated, df):
    annotated_df = pd.read_csv(csv_annotated)
    
    film_to_int_map = annotated_df.drop_duplicates('film').set_index('film')['film_id'].to_dict()
    
    df['film_id'] = df['film'].map(film_to_int_map)

    return df

def main(json, film, annotated_csv, k):
    output_path = 'Long Long Data.csv'
    
    if os.path.isfile(output_path):
        existing = pd.read_csv(output_path, usecols=['film'])
        if film in existing['film'].values:
            print(f"Film '{film}' already in {output_path}, skipping.")
            return
    rows = json_to_csv(json, film)

    frame_width, frame_height = frame_width_height(json)
    df = pd.DataFrame(rows, 
                    columns=["frame_id", 
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
                                ]
                        )
    # df = confidence_mean_rolling(df, k)
    # df = confidence_std_rolling(df, k)
    df = position_mean_rolling(df, k)
    df = position_std_rolling(df,k)
    df = position_velocity(df)
    df = position_acceleration(df)
    df = compute_boundary_distance(df, frame_width, frame_height)
    # df = frames_since_trust(df)
    # df = frames_since_dont_trust(df)
    # df = window_fractions(df)
    df = film_int_encoding(annotated_csv, df)

    df = df.drop(columns = [
                            'joint_name.1', 
                            'valid instance bbox', 
                            'reliability_category', 
                            'confidence_mean_wk',
                            'x_velocity',
                            'y_velocity'
                            ], errors = 'ignore')

    data_checking(df)


    file_exists = os.path.isfile('Long Long Data.csv')
    df.to_csv('Long Long Data.csv', mode='a', header=not file_exists, index=False)
    print("Final Data saved.")               


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json") 
    ap.add_argument("--film")
    ap.add_argument("--annotated_csv", default = "Long Data.csv")
    ap.add_argument("--k", type = int, default = 5)


    args = ap.parse_args()
    main(
        args.json,
        args.film,
        args.annotated_csv,
        args.k
    )