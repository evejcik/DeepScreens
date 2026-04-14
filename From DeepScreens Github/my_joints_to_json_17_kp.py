import json
import numpy as np
import pandas as pd
from pathlib import Path


df = pd.read_csv('../Feature_Engineering/Long_Data_with_probs.csv')
print(df.columns.tolist())
print(df['frame_id'].head(5).tolist())
print(df['instance_id'].head(5).tolist())
print(df['joint_id'].head(5).tolist())
# def build_cleaned_json_per_film(original_json_path, my_df, output_path, film):
#     #Input: original MMPose JSON (with 133 kp and stacked 2D and 3D data) and my cleaned dataframe
#     #that includes the confidence scores replaced with my top class probabilites
#     #2D: x,y replaced with x_filled, y_filled, keypoint_interpolated added as a new boolean field per joint
#     #take out the 3D data (anything in meta_info_3d)

#     #Arguments: paths to original json (133 2D, 17 3D), 
#     # path to my cleaned df - columns = [film, instance_id, frame_id, joint_id, x_filled, y_filled, trust_probability, keypoint_interpolated]
#     # and path to output for save
#     df = pd.read_csv(my_df)

    with open(original_json_path, 'r') as f:

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
        data = json.load(f)
        lookup = {}
        for ind, row in df.iterrows():
            
            key = (row['frame_id'], row['instance_id'], row['joint_id'])
            lookup[key] = {
                'x' : row['x'],
                'y' : row['y'],
                'trust_prob' : row['prob_trust'],
                'interpolated' : False
            }
        
        for frame in data['instance_info']:
            frame_id = int(frame['frame_id']) - 1
            frame_map = {}
            for instance_id, instance in enumerate(frame.get('instances',[])):
                joints = {}
                instance['keypoint_interpolated'] = [False] * len(instance['keypoints'])
                for joint_id, (x,y) in enumerate(instance.get('keypoints', [])):
                    key = (frame_id, instance_id, joint_id)
                    if key in lookup:
                        entry = lookup[key]
                        instance['keypoints'][joint_id] = [entry['x'], entry['y']]
                        instance['keypoint_scores'] = entry['trust_prob']
                        
                        instance['keypoint_interpolated'][joint_id] = entry['interpolated']
                    else:
                        print("Error! Key values not valid.")
                        continue


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
            


# build_cleaned_json_per_film('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/ramona-demo-clip 1_1369/segment_1_1639.json', 
#                             '/Users/emmavejcik/Desktop/DeepScreens/Feature_Engineering/Long_Data_with_probs.csv',
#                             '/Users/emmavejcik/Desktop/DeepScreens/From DeepScreens Github/',
#                             'Ramona_1_1639'
#                             )



