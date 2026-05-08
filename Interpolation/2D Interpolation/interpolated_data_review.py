import json

input_path  = "/Users/emmavejcik/Desktop/DeepScreens/From DeepScreens Github/Outputs/Ramona_1_1639_pred_aggregated.json"
output_path = "/Users/emmavejcik/Desktop/DeepScreens/Interpolation/2D Interpolation/Outputs/Ramona_1_1639_interpolated_0.5.json"

with open(input_path) as f:
    aggregated = json.load(f)
with open(output_path) as f:
    interpolated = json.load(f)

diffs = []
for fi in range(len(aggregated['instance_info'])):
    agg_instances = aggregated['instance_info'][fi].get('instances', [])
    int_instances = interpolated['instance_info'][fi].get('instances', [])
    for inst_id in range(min(len(agg_instances), len(int_instances))):
        agg_kps = agg_instances[inst_id]['keypoints']
        int_kps = int_instances[inst_id]['keypoints']
        for j in range(17):
            if agg_kps[j] != int_kps[j]:
                diffs.append((fi, inst_id, j, agg_kps[j], int_kps[j]))

print(f"Total coordinate differences across all frames: {len(diffs)}")

if diffs:
    print("\nFirst 10 actual differences:")
    for fi, inst_id, j, before, after in diffs[:10]:
        print(f"  frame={fi}, instance={inst_id}, joint={j}: {before} -> {after}")
else:
    print("NO differences found — the JSON was not modified.")

import json
import numpy as np

with open("/Users/emmavejcik/Desktop/DeepScreens/From DeepScreens Github/Outputs/Ramona_1_1639_pred_aggregated.json") as f:
    raw = json.load(f)
with open("/Users/emmavejcik/Desktop/DeepScreens/Interpolation/2D Interpolation/Outputs/Ramona_1_1639_interpolated_0.5.json") as f:
    interp = json.load(f)

# Pick a frame where you saw "leg flying off" in Unity
target_frame = 245  # change to a frame where you saw the bug

raw_inst = raw['instance_info'][target_frame-1]['instances'][0]
int_inst = interp['instance_info'][target_frame-1]['instances'][0]

print("Frame", target_frame)
print("\nLeft leg chain comparison (raw vs interpolated 2D):")
for joint_idx, name in [(4, 'left_hip'), (5, 'left_knee'), (6, 'left_ankle')]:
    raw_xy = raw_inst['keypoints'][joint_idx]
    int_xy = int_inst['keypoints'][joint_idx]
    delta = np.array(int_xy) - np.array(raw_xy)
    print(f"  {name}:")
    print(f"    raw:    {raw_xy}")
    print(f"    interp: {int_xy}")
    print(f"    delta:  ({delta[0]:.1f}, {delta[1]:.1f})  magnitude: {np.linalg.norm(delta):.1f} px")

# Check whether bone lengths are sane
def bone_length(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

print("\nBone length comparison (raw vs interpolated 2D):")
for parent, child, name in [(4, 5, 'left_thigh'), (5, 6, 'left_shin')]:
    raw_len = bone_length(raw_inst['keypoints'][parent], raw_inst['keypoints'][child])
    int_len = bone_length(int_inst['keypoints'][parent], int_inst['keypoints'][child])
    print(f"  {name}:  raw={raw_len:.1f}px  interp={int_len:.1f}px  ratio={int_len/raw_len:.2f}")