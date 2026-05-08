import pandas as pd

df = pd.read_csv('Predictions Long Data.csv')

print(df.groupby('joint_name')['joint_id'].unique())
print(df.shape)

import json
with open("/Users/emmavejcik/Desktop/DeepScreens/From DeepScreens Github/Outputs/Ramona_1_1639_pred_aggregated.json") as f:
    data = json.load(f)

# Inspect first few frames
for frame in data['instance_info'][:3]:
    print(f"Frame {frame['frame_id']}:")
    for inst in frame.get('instances', []):
        scores = inst.get('keypoint_scores', [])
        kps = inst.get('keypoints', [])
        print(f"  Scores (should be prob_unreliable): {scores}")
        print(f"  Sample keypoints: {kps[:3]}")


import pandas as pd
import json

# Load predictions CSV
preds = pd.read_csv("/Users/emmavejcik/Desktop/DeepScreens/Classification/Predictions Long Data.csv", low_memory=False)
ramona = preds[preds['film'] == 'Ramona_1_1639']
print(f"Total Ramona rows in predictions: {len(ramona)}")
print(f"Unique frames: {ramona['frame_id'].nunique()}")
print(f"Frame 1 rows:\n{ramona[ramona['frame_id'] == 1][['joint_id','joint_name','prob_unreliable']]}")
print(f"\nFrame 0 rows:\n{ramona[ramona['frame_id'] == 0][['joint_id','joint_name','prob_unreliable']]}")

