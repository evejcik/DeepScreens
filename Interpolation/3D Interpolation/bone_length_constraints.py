import numpy as np
import json

H36M_BONES = [
    # (parent_idx, child_idx, name)
    # Process in order from root outward
    (0, 1, 'right_hip_offset'),    # root → right_hip
    (1, 2, 'right_thigh'),         # right_hip → right_knee
    (2, 3, 'right_shin'),          # right_knee → right_ankle
    (0, 4, 'left_hip_offset'),     # root → left_hip
    (4, 5, 'left_thigh'),          # left_hip → left_knee
    (5, 6, 'left_shin'),           # left_knee → left_ankle
    (0, 7, 'spine_lower'),         # root → spine
    (7, 8, 'spine_upper'),         # spine → thorax
    (8, 9, 'neck'),                # thorax → neck_base
    (9, 10, 'head_offset'),        # neck_base → head
    (8, 11, 'left_clavicle'),      # thorax → left_shoulder
    (11, 12, 'left_upper_arm'),    # left_shoulder → left_elbow
    (12, 13, 'left_forearm'),      # left_elbow → left_wrist
    (8, 14, 'right_clavicle'),     # thorax → right_shoulder
    (14, 15, 'right_upper_arm'),   # right_shoulder → right_elbow
    (15, 16, 'right_forearm'),     # right_elbow → right_wrist
]


def compute_canonical_bone_lengths(all_kpts):
    """
    all_kpts: array of shape (n_frames, 17, 3)
    Returns: dict mapping (parent_idx, child_idx) -> canonical length (median)
    """
    canonical = {}
    for parent_idx, child_idx, name in H36M_BONES:
        # Compute bone length for every frame
        diffs = all_kpts[:, child_idx, :] - all_kpts[:, parent_idx, :]
        lengths = np.linalg.norm(diffs, axis=1)
        # Use median for robustness against outlier frames
        canonical_len = float(np.median(lengths))
        canonical[(parent_idx, child_idx)] = canonical_len
    return canonical


def enforce_bone_lengths(kpts, canonical_lengths, blend=1.0):
    """
    kpts: array of shape (17, 3) — single frame
    canonical_lengths: dict from compute_canonical_bone_lengths
    blend: 1.0 = full enforcement; 0.5 = halfway between original and enforced
    
    Process bones in parent-first order. Each child joint is repositioned 
    along the parent-child direction at the canonical length.
    """
    out = kpts.copy()
    for parent_idx, child_idx, name in H36M_BONES:
        canonical_len = canonical_lengths[(parent_idx, child_idx)]
        
        parent_pos = out[parent_idx]
        child_pos = out[child_idx]
        
        diff = child_pos - parent_pos
        current_len = np.linalg.norm(diff)
        
        if current_len < 1e-6:
            # Degenerate, skip
            continue
        
        direction = diff / current_len
        target_child_pos = parent_pos + canonical_len * direction
        
        # Blend toward target position
        out[child_idx] = child_pos + blend * (target_child_pos - child_pos)
    
    return out


def apply_bone_length_constraint_to_json(input_json_path, output_json_path, blend=0.7):
    with open(input_json_path) as f:
        data = json.load(f)
    
    # Step 1: collect all 3D keypoints to compute canonical lengths
    all_kpts = []
    for frame_entry in data['instance_info']:
        for instance in frame_entry.get('instances', []):
            kp = instance.get('keypoints_3d')
            if kp is None:
                continue
            arr = np.asarray(kp, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.shape == (17, 3):
                all_kpts.append(arr)
    
    if not all_kpts:
        print("No 3D keypoints found.")
        return
    
    all_kpts_arr = np.stack(all_kpts, axis=0)  # (n_frames, 17, 3)
    print(f"Computing canonical lengths from {len(all_kpts_arr)} frames")
    
    canonical = compute_canonical_bone_lengths(all_kpts_arr)
    
    print("\nCanonical bone lengths (median across video):")
    for (p, c, name), in [((p, c, n),) for p, c, n in H36M_BONES]:
        l = canonical[(p, c)]
        # Compute std deviation to show how much frame-to-frame variation existed
        diffs = all_kpts_arr[:, c, :] - all_kpts_arr[:, p, :]
        lens = np.linalg.norm(diffs, axis=1)
        print(f"  {name:<20s} median={l:.3f}  std={lens.std():.3f}  "
              f"min={lens.min():.3f}  max={lens.max():.3f}")
    
    # Step 2: apply constraint to every frame
    n_corrected = 0
    for frame_entry in data['instance_info']:
        for instance in frame_entry.get('instances', []):
            kp = instance.get('keypoints_3d')
            if kp is None:
                continue
            arr = np.asarray(kp, dtype=np.float32)
            was_nested = arr.ndim == 3 and arr.shape[0] == 1
            if was_nested:
                arr = arr[0]
            if arr.shape != (17, 3):
                continue
            
            corrected = enforce_bone_lengths(arr, canonical, blend=blend)
            
            if was_nested:
                instance['keypoints_3d'] = [corrected.tolist()]
            else:
                instance['keypoints_3d'] = corrected.tolist()
            n_corrected += 1
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent='\t')
    
    print(f"\nApplied bone length constraint to {n_corrected} instances.")
    print(f"Wrote: {output_json_path}")


if __name__ == '__main__':
    apply_bone_length_constraint_to_json(
        input_json_path = 'Smoothed_Output/smoothed.json', #smoothed sav golay
        output_json_path ='Smoothed_Output/bone_length_constrained.json',
        blend=1.0
    )