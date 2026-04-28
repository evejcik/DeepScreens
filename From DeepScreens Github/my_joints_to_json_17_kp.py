import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

#input: long data csv
#output a json for a movie/movies

# df = pd.read_csv('../Feature_Engineering/Long_Data_with_probs.csv')
# print(df.columns.tolist())
# print(df['frame_id'].head(5).tolist())
# print(df['instance_id'].head(5).tolist())
# print(df['joint_id'].head(5).tolist())


#     #Input: original MMPose JSON (with 133 kp and stacked 2D and 3D data) and my cleaned dataframe
#     #that includes the confidence scores replaced with my top class probabilites
#     #2D: x,y replaced with x_filled, y_filled, keypoint_interpolated added as a new boolean field per joint
#     #take out the 3D data (anything in meta_info_3d)

#     #Arguments: paths to original json (133 2D, 17 3D), 
#     # path to my cleaned df - columns = [film, instance_id, frame_id, joint_id, x_filled, y_filled, trust_probability, keypoint_interpolated]
#     # and path to output for save
#     df = pd.read_csv(my_df)

def convert_rtmpose133_to_h36m17_2d(coco_keypoints):
    """Convert 133 RTMW keypoints to 17 H36M keypoints (2D version)."""
    h36m_keypoints = np.zeros((17, 2), dtype=np.float32)
    
    # COCO/RTMW keypoint indices
    Nose, L_Eye, R_Eye = 0, 1, 2
    L_Ear, R_Ear = 3, 4
    L_Shoulder, R_Shoulder = 5, 6
    L_Elbow, R_Elbow = 7, 8
    L_Wrist, R_Wrist = 9, 10
    L_Hip, R_Hip = 11, 12
    L_Knee, R_Knee = 13, 14
    L_Ankle, R_Ankle = 15, 16
    
    shoulder_midpoint = (coco_keypoints[L_Shoulder] + coco_keypoints[R_Shoulder]) / 2
    hip_midpoint = (coco_keypoints[L_Hip] + coco_keypoints[R_Hip]) / 2
    ear_midpoint = (coco_keypoints[L_Ear] + coco_keypoints[R_Ear]) / 2
    spine_vector = shoulder_midpoint - hip_midpoint
    neck_vector = ear_midpoint - shoulder_midpoint

    # H36M joint assignments
    h36m_keypoints[0] = hip_midpoint  # root
    h36m_keypoints[1] = coco_keypoints[R_Hip]  # right_hip
    h36m_keypoints[2] = coco_keypoints[R_Knee]  # right_knee
    h36m_keypoints[3] = coco_keypoints[R_Ankle]  # right_foot
    h36m_keypoints[4] = coco_keypoints[L_Hip]  # left_hip
    h36m_keypoints[5] = coco_keypoints[L_Knee]  # left_knee
    h36m_keypoints[6] = coco_keypoints[L_Ankle]  # left_foot
    
    # Torso points (spine, thorax, neck_base)

    h36m_keypoints[7] = hip_midpoint + 0.5 * spine_vector  # spine
    h36m_keypoints[8] = shoulder_midpoint  # thorax
    h36m_keypoints[9] = shoulder_midpoint + 0.15 * neck_vector  # neck_base

    # Head (extrapolate from nose and eyes)

    h36m_keypoints[10] = ear_midpoint

    # Arms
    h36m_keypoints[11] = coco_keypoints[L_Shoulder]  # left_shoulder
    h36m_keypoints[12] = coco_keypoints[L_Elbow]  # left_elbow
    h36m_keypoints[13] = coco_keypoints[L_Wrist]  # left_wrist
    h36m_keypoints[14] = coco_keypoints[R_Shoulder]  # right_shoulder
    h36m_keypoints[15] = coco_keypoints[R_Elbow]  # right_elbow
    h36m_keypoints[16] = coco_keypoints[R_Wrist]  # right_wrist
    
    return h36m_keypoints

def remap_keypoint_scores_133_to_17(coco_scores):
    h36m_scores = np.zeros(17, dtype=np.float32)
    
    Nose, L_Eye, R_Eye = 0, 1, 2
    L_Shoulder, R_Shoulder = 5, 6
    L_Elbow, R_Elbow = 7, 8
    L_Wrist, R_Wrist = 9, 10
    L_Hip, R_Hip = 11, 12
    L_Knee, R_Knee = 13, 14
    L_Ankle, R_Ankle = 15, 16
    
    h36m_scores[0] = geometric_mean([coco_scores[L_Hip], coco_scores[R_Hip]])
    h36m_scores[1] = coco_scores[R_Hip]
    h36m_scores[2] = coco_scores[R_Knee]
    h36m_scores[3] = coco_scores[R_Ankle]
    h36m_scores[4] = coco_scores[L_Hip]
    h36m_scores[5] = coco_scores[L_Knee]
    h36m_scores[6] = coco_scores[L_Ankle]
    
    # Spine and thorax depend on all four torso joints — use geometric mean of all four
    # because their positions are geometric constructions along the hip-shoulder vector,
    # not just averages of the shoulder pair
    torso_score_full = geometric_mean([
        coco_scores[L_Hip], coco_scores[R_Hip],
        coco_scores[L_Shoulder], coco_scores[R_Shoulder]
    ])
    h36m_scores[7] = torso_score_full   # spine
    h36m_scores[8] = torso_score_full   # thorax
    
    # Neck base is closer to shoulder midpoint so use shoulders only
    h36m_scores[9] = geometric_mean([coco_scores[L_Shoulder], coco_scores[R_Shoulder]])
    
    # Head
    h36m_scores[10] = geometric_mean([coco_scores[Nose], coco_scores[L_Eye], coco_scores[R_Eye]])
    
    h36m_scores[11] = coco_scores[L_Shoulder]
    h36m_scores[12] = coco_scores[L_Elbow]
    h36m_scores[13] = coco_scores[L_Wrist]
    h36m_scores[14] = coco_scores[R_Shoulder]
    h36m_scores[15] = coco_scores[R_Elbow]
    h36m_scores[16] = coco_scores[R_Wrist]
    
    return h36m_scores.tolist()
def geometric_mean(scores: list) -> float:
    """
    Calculate geometric mean of confidence scores.

    Args:
        scores: List of confidence score values

    Returns:
        Geometric mean as float
    """
    if len(scores) == 0:
        return 0.0

    # Convert to numpy for easier computation
    scores_array = np.array(scores, dtype=np.float32)

    # Handle zeros and negative values (clamp to small positive value)
    scores_array = np.maximum(scores_array, 1e-10)

    # Geometric mean: (s1 * s2 * ... * sN)^(1/N)
    # Using log space for numerical stability: exp(mean(log(scores)))
    log_scores = np.log(scores_array)
    geometric_mean_value = np.exp(np.mean(log_scores))

    return float(geometric_mean_value)


def build_cleaned_json_per_film(original_json_path, df, output_path, film):
    with open(original_json_path, 'r') as f:
        data = json.load(f)
        # print("1")
        df = df[df['film'] == film]

        lookup = {}
        for ind, row in df.iterrows():
            # print("2")
            key = (row['frame_id'], row['instance_id'], row['joint_id'])
            lookup[key] = {
                'x' : row['x'],
                'y' : row['y'],
                'trust_prob' : row['prob_trust'],
                'interpolated' : False
            }
        
        for frame in data['instance_info']:
            # print(3)
            # print("HERE")
            frame_id = int(frame['frame_id']) - 1
            frame_map = {}
            for instance_id, instance in enumerate(frame.get('instances', [])):
    
                # Step 1: convert original 133 COCO keypoints to 17 H36M baseline
                original_133 = np.array(instance['keypoints'])  # (133, 2)
                if original_133.ndim == 3:
                    original_133 = original_133.squeeze(0)       # handle (1,133,2)
                
                h36m_17 = convert_rtmpose133_to_h36m17_2d(original_133)  # (17, 2)
                
                # Step 2: get original scores, remap to 17
                original_scores_133 = np.array(instance['keypoint_scores']).flatten()
                scores_17 = remap_keypoint_scores_133_to_17(original_scores_133)
                
                # Step 3: build new lists from H36M baseline
                new_keypoints    = h36m_17.tolist()
                new_scores       = list(scores_17)
                new_interpolated = [False] * 17

                # Step 4: overwrite with cleaned values where available
                for joint_id_h36m in range(17):
                    key = (frame_id, instance_id, joint_id_h36m)
                    if key in lookup:
                        entry = lookup[key]
                        new_keypoints[joint_id_h36m]    = [entry['x'], entry['y']]
                        new_scores[joint_id_h36m]       = entry['trust_prob']
                        new_interpolated[joint_id_h36m] = entry['interpolated']

                instance['keypoints']             = new_keypoints
                instance['keypoint_scores']       = new_scores
                instance['keypoint_interpolated'] = new_interpolated
            # print(6)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent='\t')

        # after saving
        with open(output_path, 'r') as f:
            result = json.load(f)

        inst = result['instance_info'][0]['instances'][0]
        print(inst.keys())
        print(inst['keypoint_interpolated'][:5])
        print(inst['keypoint_scores'][:5])
        print(inst['keypoints'][:5])
        print(result['instance_info'][0]['instances'][0]['keypoint_scores'][0])
        print(result['instance_info'][0]['instances'][0]['bbox'])


def main(json, csv, output_path):
    df = pd.read_csv(csv)
    # for film in df['film'].unique():
    film = 'Moonlight_1_1529'
    build_cleaned_json_per_film(json, df, output_path, film)
    print("Done!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--csv")
    ap.add_argument("--output_path")

    args = ap.parse_args()

    main(
        # '/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Moonlight_2016/Moonlight_2016/segment_1_1529.json',
        '/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/ramona-demo-clip 1_1369/segment_1_1639.json',
        '/Users/emmavejcik/Desktop/DeepScreens/Classification/Long_Data_with_probs.csv',
        '/Users/emmavejcik/Desktop/DeepScreens/From DeepScreens Github/Outputs/my_17_json.json'
    )


        #subtract one from json 

        #meta_info: -> dictionary -> these values I need to delete and fill in with my own: containing keys, num_keypoints: 17, keypoint_id2name, keypoint_name2id, 
        #do i need to cut down on the flip_indices, pairs, keypoint_colors, etc. that all contain 133 values? 
        #num_skeleton_links = 65, skeleton_links, skeleton_links_colors, should I cut this down?
        #dataset_keypoint_weights -> what is this?
        #what is sigmas? is this the sigmoid smoothing function?
        #skip everything in meta_info_3d
        #dig into: instance_info['frame_id']['instances']['keypoints']
        #make my own dictionary first in this format

        #take in one film at a time
        #should I group by frame, then instances?
        


#         for (frame, instance, joint), group in df.groupby(['frame','instance']).apply(lambda x: x.sort_values('joint_id')):

#             # x = df.loc[(df['frame_id'] == frame) & (df['instance'] == instance) & (df['joint_id'] == joint), 'x_filled']
#             # y = 
#             
#                 # frames_since_dont_trust = 0 if row['reliability_category_int'] == 2 else frames_since_dont_trust + 1 if frames_since_dont_trust >= 0 else -1
#                 # results[idx] = frames_since_dont_trust

#                 # keypoints.append(['x_filled', 'y_filled'])
            
#             instances = np.array([{"keypoints" : keypoints}], dtype = object)

#         # print(data.keys())
#         # print(type(data['instance_info']))
#         # print(data['instance_info'][0].keys())
#         # print(data['instance_info'][0]['instances'][0].keys())
            

