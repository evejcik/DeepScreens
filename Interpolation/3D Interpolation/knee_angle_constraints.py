import numpy as np
import json

H36M_INDEX = {
    'right_hip': 1, 'right_knee': 2, 'right_ankle': 3,
    'left_hip': 4, 'left_knee': 5, 'left_ankle': 6,
}

def project_knee_to_leg_axis(kpts, hip_idx, knee_idx, ankle_idx, blend=0.7):
    """
    Projects the knee onto the hip-ankle line, with a blend factor controlling
    how aggressively the correction is applied.
    
    blend=1.0 -> knee is forced exactly onto hip-ankle line (no real flexion)
    blend=0.0 -> no correction at all
    blend=0.7 -> knee is pulled 70% of the way from its position toward the line
    """
    kpts = np.asarray(kpts)
    hip = kpts[hip_idx]
    knee = kpts[knee_idx]
    ankle = kpts[ankle_idx]
    
    # Project knee onto line from hip to ankle
    leg_axis = ankle - hip
    leg_len_sq = np.dot(leg_axis, leg_axis)
    if leg_len_sq < 1e-6:
        return kpts  # degenerate, skip
    
    knee_offset = knee - hip
    t = np.dot(knee_offset, leg_axis) / leg_len_sq
    t = np.clip(t, 0.4, 0.6)  # knee should be roughly mid-leg
    knee_projected = hip + t * leg_axis
    
    # Blend toward projected position
    new_knee = knee + blend * (knee_projected - knee)
    
    out = kpts.copy()
    out[knee_idx] = new_knee
    return out


def apply_knee_constraint_to_json(input_json_path, output_json_path, blend=0.5):
    with open(input_json_path) as f:
        data = json.load(f)
    
    n_corrected = 0
    for frame_entry in data['instance_info']:
        for instance in frame_entry.get('instances', []):
            if 'keypoints_3d' not in instance:
                continue
            kpts = np.asarray(instance['keypoints_3d'], dtype=np.float32)
            if kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = kpts[0]
            if kpts.shape != (17, 3):
                continue
            
            # Apply to both legs
            kpts = project_knee_to_leg_axis(
                kpts, H36M_INDEX['left_hip'], H36M_INDEX['left_knee'], 
                H36M_INDEX['left_ankle'], blend=blend
            )
            kpts = project_knee_to_leg_axis(
                kpts, H36M_INDEX['right_hip'], H36M_INDEX['right_knee'],
                H36M_INDEX['right_ankle'], blend=blend
            )
            
            # Match the original nesting
            original = instance['keypoints_3d']
            if isinstance(original, list) and len(original) == 1:
                instance['keypoints_3d'] = [kpts.tolist()]
            else:
                instance['keypoints_3d'] = kpts.tolist()
            n_corrected += 1
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent='\t')
    
    print(f"Applied knee constraint to {n_corrected} instances. Wrote {output_json_path}")


if __name__ == '__main__':
    apply_knee_constraint_to_json(
        '/Users/emmavejcik/Desktop/DeepScreens/Interpolation/3D Interpolation/Smoothed_Output/bone_length_constrained.json',
        'Smoothed_Output/knee_angle_constrained_smoothed.json',
        blend=0.7
    )