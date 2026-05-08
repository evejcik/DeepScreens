import json
with open("/Users/emmavejcik/Desktop/DeepScreens/Interpolation/2D Interpolation/Outputs/3D Outputs/results_segment_1_1639.json") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))

# Look at first frame's first instance
first_frame = data['instance_info'][0]
print("Frame keys:", list(first_frame.keys()))

first_inst = first_frame['instances'][0]
print("Instance keys:", list(first_inst.keys()))

import numpy as np
kp3d = np.asarray(first_inst.get('keypoints_3d', []))
print(f"keypoints_3d shape: {kp3d.shape}")
print(f"First 3 joints: {first_inst['keypoints_3d'][:3]}")


track_counts = {}
track_frames = {}
for fe in data['instance_info']:
    fid = fe['frame_id']
    for inst in fe.get('instances', []):
        tid = inst.get('track_id', None)
        if tid is None:
            continue
        track_counts[tid] = track_counts.get(tid, 0) + 1
        track_frames.setdefault(tid, []).append(fid)

print(f"Unique track_ids: {len(track_counts)}")
print(f"Track distribution:")
for tid in sorted(track_counts.keys(), key=lambda t: -track_counts[t]):
    fids = track_frames[tid]
    print(f"  track_id={tid}: {track_counts[tid]} frames "
          f"({fids[0]}..{fids[-1]}, span {fids[-1]-fids[0]})")

# Check whether multiple instances per frame is real or tracker artifact
multi_inst_frames = sum(1 for fe in data['instance_info'] if len(fe.get('instances', [])) > 1)
print(f"\nFrames with multiple instances: {multi_inst_frames}")