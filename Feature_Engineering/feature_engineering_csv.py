#input: annotated csv 
#output: 



import pandas as pd
import numpy as np
import argparse
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

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
        print(f"Film name: {film_name}")
        csv = pd.read_csv(f)

        csv['film'] = film_name
        print(f"{film_name} Shape: {csv.shape}")

        dfs.append(csv)
    # df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # filename
    df = pd.concat(dfs, ignore_index = True)
    return df

def reliability_int(df):
    trust_map = {
        "trust" : 0,
        "don't trust" : 2,
        "partial trust" : 2, #also mapping to dont trust for now
        "ambiguous" : 2 #mapping to dont trust for now, can discuss later
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

def clean_nans(df):
    cols = ['geom_plausible', 
            'bone_length', 
            'geom_flag', 
            'bone_ratio', 
            'reason_for_distrust', 
            'position_velocity',
            'position_acceleration',
            'dist_to_boundary'
            ]
    df[cols] = df[cols].fillna(-1)

    return df

# def joint_mapping(df):

#     df["joint_id"] = df["joint_name"].map(RTMW_TO_H36M_ID)  # NaN for unmapped
#     unmapped = df[df["joint_id"].isna()]["joint_name"].unique()
#     if len(unmapped):
#         print(f"Warning: unmapped joints: {unmapped}")
#     return df

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

    df = reliability_int(df)

    # df = joint_mapping(df)

    df = is_valid(df)

    print(df.columns)

    # print(df.head(5))
    df = df.sort_values(['film','instance_id', 'joint_name', 'frame_id'])

    print(df.iloc[:5, :])

    df['geom_plausible'] = df['geom_plausible'].map({True: 1, False: 0, -1: -1})
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

### Data Checking ###
def data_checking(df):
    print(f"Reliability counts: {df['reliability_category_int'].value_counts()}")
    print(f"Final columns: {df.columns}")
    print(f"Shape final df: {df.shape}")

    print(f"Max frames since trust: {df['frames_since_trust'].max()}")

    print(f"Frames more than 20 away from a trust: {(df['frames_since_trust'] > 20).sum()}")
    print(f"Percentage of frames more than 20 away from a trust: {(df['frames_since_trust'] > 20).sum()/df.shape[0]}")

    print(f"Geom plausible values: {df['geom_plausible'].value_counts(dropna=False)}")

    feature_cols = [
        'reliability_category_int',
        'mmpose_confidence',
        # 'confidence_mean_wk',
        'confidence_std_wk', 
        'position_velocity',
        'position_acceleration',
        'position_std_x_wk',
        'position_std_y_wk',
        'frames_since_trust'
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
        
    print(df.groupby('film')['joint_name'].unique())
    pd.reset_option('display.max_columns')
    print(df.groupby(['film', 'joint_name']).size().unstack(fill_value=0))

    print(f"NULL VALUES: {df.isnull().sum()[df.isnull().sum() > 0]}")

##NEW
def extract_features_from_json(json_path: str, film_name: str) -> pd.DataFrame:
    import json
    from geometric_plausibility import (compute_plausibility_for_instance,
                                         compute_boundary_distance)

    H36M_JOINT_NAMES = {
        0:'root', 1:'right_hip', 2:'right_knee', 3:'right_ankle',
        4:'left_hip', 5:'left_knee', 6:'left_ankle', 7:'spine',
        8:'thorax', 9:'neck_base', 10:'head', 11:'left_shoulder',
        12:'left_elbow', 13:'left_wrist', 14:'right_shoulder',
        15:'right_elbow', 16:'right_wrist'
    }

    with open(json_path) as f:
        data = json.load(f)

    video_info   = data.get('video_info', [{}])[0]
    video_shape  = video_info.get('video_shape', [640, 360])
    frame_width  = video_shape[0]
    frame_height = video_shape[1]

    # Build plausibility cache per (frame_id, instance_id)
    plaus_cache = {}
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        for inst_idx, inst in enumerate(frame.get('instances', [])):
            kps   = inst.get('keypoints', [])
            confs = inst.get('keypoint_scores', [])
            plaus_cache[(frame_id, inst_idx)] = \
                compute_plausibility_for_instance(kps, confs)

    rows = []
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        for inst_idx, inst in enumerate(frame.get('instances', [])):
            kps    = inst['keypoints']
            scores = inst['keypoint_scores']
            track_id = inst.get('track_id', -1)
            plaus  = plaus_cache.get((frame_id, inst_idx), {})

            for joint_id in range(17):
                x, y = float(kps[joint_id][0]), float(kps[joint_id][1])
                conf = float(scores[joint_id])

                p = plaus.get(joint_id)
                geom_plausible = (1 if p and p['geom_plausible'] is True
                                  else 0 if p and p['geom_plausible'] is False
                                  else -1)
                bone_length = p['bone_length'] if p and p['bone_length'] else -1
                bone_ratio  = p['bone_ratio']  if p and p['bone_ratio']  else -1

                # dist_to_boundary
                dist = min(x, frame_width - x, y, frame_height - y)

                rows.append({
                    'frame_id':               frame_id,
                    'instance_id':            inst_idx,
                    'track_id':               track_id,
                    'joint_id':               joint_id,
                    'joint_name':             H36M_JOINT_NAMES[joint_id],
                    'x':                      x,
                    'y':                      y,
                    'mmpose_confidence':      conf,
                    'dist_to_boundary':       round(dist, 2),
                    'bone_length':            bone_length,
                    'bone_ratio':             bone_ratio,
                    'geom_plausible':         geom_plausible,
                    'film':                   film_name,
                    # label-dependent — fill with -1 for inference
                    'reliability_category_int': np.nan,
                    'frames_since_trust':      -1,
                    'frames_since_dont_trust': -1,
                    'frac_trust_wk':           -1,
                    'frac_partial_wk':         -1,
                    'frac_dont_trust_wk':      -1,
                    'film_id':                 -1,
                })

    df = pd.DataFrame(rows)
    df = confidence_std_rolling(df, k=5)
    df = position_velocity(df)
    df = position_acceleration(df)

    return df
def main(csv,k):
    df = data_loader(csv)

    df = confidence_mean_rolling(df, k)
    df = confidence_std_rolling(df, k)
    df = position_mean_rolling(df, k)
    df = position_std_rolling(df,k)
    df = position_velocity(df)
    df = position_acceleration(df)
    df = frames_since_trust(df)
    df = frames_since_dont_trust(df)
    df = window_fractions(df)
    df = film_int_encoding(df)


    df = df.drop(columns = [
                            'joint_name.1', 
                            'valid instance bbox', 
                            'reliability_category', 
                            'confidence_mean_wk',
                            'x_velocity',
                            'y_velocity'
                            ], errors = 'ignore')

    df = clean_nans(df)
    data_checking(df)
    df.to_csv('Long Data.csv', index = False)
    # print("Final Data saved.")


# def main_unannotated(csv,k):
#     df = data_loader(csv)

#     df = confidence_mean_rolling(df, k)
#     df = confidence_std_rolling(df, k)
#     df = position_mean_rolling(df, k)
#     df = position_std_rolling(df,k)
#     df = position_velocity(df)
#     df = position_acceleration(df)
#     # df = frames_since_trust(df)
#     # df = frames_since_dont_trust(df)
#     # df = window_fractions(df)
#     df = film_int_encoding(df)


#     df = df.drop(columns = [
#                             'joint_name.1', 
#                             'valid instance bbox', 
#                             'reliability_category', 
#                             'confidence_mean_wk',
#                             'x_velocity',
#                             'y_velocity'
#                             ], errors = 'ignore')

#     df = clean_nans(df)
#     data_checking(df)
#     df.to_csv('Long Long Data.csv', index = False)
#     print("Final Data saved.")

    



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default = "../ANNOTATED_CSVS") 
    ap.add_argument("--k", type = int, default = 5)


    args = ap.parse_args()
    main(
        args.csv,
        args.k
    )
