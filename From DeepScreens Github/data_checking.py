import pandas as pd
import json

df = pd.read_csv('/Users/emmavejcik/Desktop/DeepScreens/Feature_Engineering/Long_Long_Data_with_probs.csv')
print(df[['joint_name', 'joint_id']].drop_duplicates().sort_values('joint_id'))


with open('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/ramona-demo-clip 1_1369/segment_1_1639.json') as f:
    data = json.load(f)

ramona = df[df['film'] == 'Ramona_1_1639']

print("JSON first frame_id:", data['instance_info'][0]['frame_id'])
print("CSV min frame_id:", ramona['frame_id'].min())
print("CSV max frame_id:", ramona['frame_id'].max())
print("JSON total frames:", len(data['instance_info']))

annotated = pd.read_csv('../Feature_Engineering/Long Data.csv')
unannotated = pd.read_csv('../Feature_Engineering/Long Long Data.csv')

print("Annotated joint_ids:", sorted(annotated['joint_id'].dropna().unique()))
print("Unannotated joint_ids:", sorted(unannotated['joint_id'].dropna().unique()))
print("Annotated joint sample:")
print(annotated[['joint_name','joint_id']].drop_duplicates().sort_values('joint_id'))
print("Unannotated joint sample:")
print(unannotated[['joint_name','joint_id']].drop_duplicates().sort_values('joint_id'))