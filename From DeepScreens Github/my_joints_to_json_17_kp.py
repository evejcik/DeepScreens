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
            for instance_id, instance in enumerate(frame.get('instances',[])):
                # print(4)
                joints = {}
                instance['keypoint_interpolated'] = [False] * len(instance['keypoints'])
                for joint_id, (x,y) in enumerate(instance.get('keypoints', [])):
                    key = (frame_id, instance_id, joint_id)
                    if key in lookup:
                        entry = lookup[key]
                        # print(5)
                        instance['keypoints'][joint_id] = [entry['x'], entry['y']]
                        instance['keypoint_scores'][joint_id] = entry['trust_prob']
                        
                        instance['keypoint_interpolated'][joint_id] = entry['interpolated']
                    else:
                        # print(f"Error! Key values not valid at {key}.")
                        pass
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
        '/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Moonlight_2016/Moonlight_2016/segment_1_1529.json',
        '../Classification/Long_Data_with_probs.csv',
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
            

