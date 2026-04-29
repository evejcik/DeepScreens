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
                'prob_unreliable' : row['prob_unreliable'],
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
                        new_scores[joint_id_h36m]       = entry['prob_unreliable']
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


def probs_distribs(df):
    """
    Print probability distribution statistics to help determine
    reliability threshold selection.
    """
    reliable   = df[df['reliability_category_int'] == 0]
    unreliable = df[df['reliability_category_int'] == 1]

    films = df['film'].unique()

    for film in films:
        print(f"\n{'='*60}")
        print(f"  FILM: {film}")
        print(f"{'='*60}")

        film_df   = df[df['film'] == film]
        film_rel  = reliable[reliable['film'] == film]
        film_unrel = unreliable[unreliable['film'] == film]

        total   = len(film_df)
        n_rel   = len(film_rel)
        n_unrel = len(film_unrel)

        print(f"\n  Sample counts:")
        print(f"    Total joints:      {total}")
        print(f"    Reliable:          {n_rel}  ({100*n_rel/total:.1f}%)")
        print(f"    Unreliable:        {n_unrel}  ({100*n_unrel/total:.1f}%)")

        # --- Overall distribution ---
        p = film_df['prob_trust']
        print(f"\n  Overall prob_trust distribution:")
        print(f"    Mean:   {p.mean():.4f}")
        print(f"    Median: {p.median():.4f}")
        print(f"    Std:    {p.std():.4f}")
        print(f"    Min:    {p.min():.4f}")
        print(f"    Max:    {p.max():.4f}")
        print(f"    10th pct: {p.quantile(0.10):.4f}")
        print(f"    25th pct: {p.quantile(0.25):.4f}")
        print(f"    75th pct: {p.quantile(0.75):.4f}")
        print(f"    90th pct: {p.quantile(0.90):.4f}")

        # --- By class ---
        print(f"\n  By reliability class:")
        print(f"    {'Metric':<18} {'Reliable':>12} {'Unreliable':>12}")
        print(f"    {'-'*44}")
        for metric, func in [
            ('Mean',   lambda x: x.mean()),
            ('Median', lambda x: x.median()),
            ('Std',    lambda x: x.std()),
            ('Min',    lambda x: x.min()),
            ('Max',    lambda x: x.max()),
            ('10th pct', lambda x: x.quantile(0.10)),
            ('25th pct', lambda x: x.quantile(0.25)),
            ('75th pct', lambda x: x.quantile(0.75)),
            ('90th pct', lambda x: x.quantile(0.90)),
        ]:
            r_val  = func(film_rel['prob_trust'])  if len(film_rel)  > 0 else float('nan')
            u_val  = func(film_unrel['prob_trust']) if len(film_unrel) > 0 else float('nan')
            print(f"    {metric:<18} {r_val:>12.4f} {u_val:>12.4f}")

        # --- Threshold analysis ---
        print(f"\n  Threshold analysis (what gets flagged as unreliable):")
        print(f"    {'Threshold':<12} {'TP rate':>10} {'FP rate':>10} {'Precision':>10} {'F1':>8}")
        print(f"    {'-'*54}")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # Predicted unreliable = prob_trust < threshold
            pred_unrel = film_df['prob_trust'] > thresh
            actual_unrel = film_df['reliability_category_int'] == 1

            tp = (pred_unrel & actual_unrel).sum()
            fp = (pred_unrel & ~actual_unrel).sum()
            fn = (~pred_unrel & actual_unrel).sum()

            tpr       = tp / (tp + fn)   if (tp + fn) > 0   else 0.0
            fpr       = fp / n_rel        if n_rel > 0        else 0.0
            precision = tp / (tp + fp)    if (tp + fp) > 0    else 0.0
            f1        = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

            print(f"    {thresh:<12.1f} {tpr:>10.4f} {fpr:>10.4f} {precision:>10.4f} {f1:>8.4f}")

        # --- Per joint breakdown ---
        print(f"\n  Per-joint mean prob_trust (reliable vs unreliable):")
        print(f"    {'Joint ID':<10} {'Joint name':<20} {'Rel mean':>10} {'Unrel mean':>12} {'Separation':>12}")
        print(f"    {'-'*66}")

        joint_names = {
            0: 'root', 1: 'R_hip', 2: 'R_knee', 3: 'R_ankle',
            4: 'L_hip', 5: 'L_knee', 6: 'L_ankle', 7: 'spine',
            8: 'thorax', 9: 'neck_base', 10: 'head',
            11: 'L_shoulder', 12: 'L_elbow', 13: 'L_wrist',
            14: 'R_shoulder', 15: 'R_elbow', 16: 'R_wrist'
        }

        for jid in range(17):
            j_rel   = film_rel[film_rel['joint_id'] == jid]['prob_trust']
            j_unrel = film_unrel[film_unrel['joint_id'] == jid]['prob_trust']
            if len(j_rel) == 0 and len(j_unrel) == 0:
                continue
            r_mean  = j_rel.mean()   if len(j_rel)   > 0 else float('nan')
            u_mean  = j_unrel.mean() if len(j_unrel) > 0 else float('nan')
            sep     = r_mean - u_mean if not (np.isnan(r_mean) or np.isnan(u_mean)) else float('nan')
            print(f"    {jid:<10} {joint_names.get(jid,''):<20} {r_mean:>10.4f} {u_mean:>12.4f} {sep:>12.4f}")

    print(f"\n{'='*60}")
    print("  THRESHOLD RECOMMENDATION GUIDE")
    print(f"{'='*60}")
    print("  High TP rate, higher FP rate -> lower threshold (more aggressive flagging)")
    print("  High precision, lower TP rate -> higher threshold (more conservative)")
    print("  F1 peak = best balance between catching unreliable joints")
    print("  and not discarding reliable ones")
    print(f"{'='*60}\n")


def main(film, json, csv, output_path):
    df = pd.read_csv(csv)
    # for film in df['film'].unique():
    # film = 'Moonlight_1_1529'
    build_cleaned_json_per_film(json, df, output_path, film)
    probs_distribs(df)
    print(df.columns.tolist())
    print(df['reliability_category_int'].value_counts() if 'reliability_category_int' in df.columns else "column missing")
    print("Done!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--csv")
    ap.add_argument("--output_path")

    args = ap.parse_args()

    main(
        film = 'Ramona_1_1639',
        # '/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Moonlight_2016/Moonlight_2016/segment_1_1529.json',
        json = '/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/ramona-demo-clip 1_1369/segment_1_1639.json',
        csv = '/Users/emmavejcik/Desktop/DeepScreens/Feature_Engineering/Long_Data_with_probs.csv',
        output_path = f'/Users/emmavejcik/Desktop/DeepScreens/From DeepScreens Github/Outputs/{film}_pred_aggregated.json'
    )

