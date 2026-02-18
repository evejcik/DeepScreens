#group together all of the lower body scores to at least somewhat reliably determine whether or not legs are in the frame .

#go through each frame.
#using highest threshold = 0.85, get confidence score from frame of left leg, right leg, left foot, right foot, left ankle, right ankle, left toe, right toe,
#left heel, right heel.


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
from pprint import pprint
from IPython.display import clear_output
import seaborn as sns



#Key Points from 133 Data
keypoint_id2name = {
            "0": "nose",
			"5": "left_shoulder",
			"6": "right_shoulder",
			"7": "left_elbow",
			"8": "right_elbow",
			"9": "left_wrist",
			"10": "right_wrist",
			"11": "left_hip",
			"12": "right_hip",
			"13": "left_knee",
			"14": "right_knee",
			"15": "left_ankle",
			"16": "right_ankle",
			"17": "left_big_toe",
			"18": "left_small_toe",
			"19": "left_heel",
			"20": "right_big_toe",
			"21": "right_small_toe",
			"22": "right_heel"
}

leg_group = {11,12,13,14,15,16,17,18,19,20,21,22}
chest_group = {5,6,7,8,9,10}
face_group = {0}

with open('133_test.json', 'r') as j_file:
    data = json.load(j_file)

rows = []

for frame in data["instance_info"]:
    frame_id = frame["frame_id"]
    for char_num, instance in enumerate(frame["instances"]):
        keypoint_scores = instance["keypoint_scores"]

        for keypoint_id, keypoint_part in keypoint_id2name.items():
            keypoint_id_num = int(keypoint_id)
            keypoint_score = keypoint_scores[keypoint_id_num]

            is_leg = keypoint_id_num in leg_group
            is_chest = keypoint_id_num in chest_group
            is_face = keypoint_id_num in face_group

            rows.append((frame_id, char_num, keypoint_part, keypoint_score, is_leg, is_chest, is_face))


data133_to17 = pd.DataFrame(rows, columns=["frame_id", "char_num", "kp_name", "score", "is_leg", "is_chest", "is_face"])


#average of leg group confidence score per score:
leg_frame_means = (data133_to17[data133_to17['is_leg'] == True].groupby(['frame_id'])['score'].mean().reset_index())


#average of chest_group confidence score per frame:
chest_frame_means = (data133_to17[data133_to17['is_chest'] == True].groupby(['frame_id'])['score'].mean().reset_index())

#average of face_group confidence score per frame:
face_frame_means = (data133_to17[data133_to17['is_face'] == True].groupby(['frame_id'])['score'].mean().reset_index())

#per frame, check leg means vs chest means. check leg means vs face means. check face means vs leg means.
#or should i check which one is higher?

#check which average is higher. say that this is what is in the scene.

# print(face_frame_means.columns)

frame_leg_means   = leg_frame_means.rename(columns={'score': 'leg_score'})
frame_chest_means = chest_frame_means.rename(columns={'score': 'chest_score'})
frame_face_means  = face_frame_means.rename(columns={'score': 'face_score'})

# print(leg_frame_means.sort_values(by ='score', ascending = False).head(n = 20))
# print(chest_frame_means.sort_values(by='score', ascending = False).head(n = 20))
# print(face_frame_means.sort_values(by='score',ascending = False).head(n = 20))

# leg_frame_means.merge(chest_frame_means, how = "outer", on='frame_id')
# leg_frame_means.merge(face_frame_means, how = 'outer', on='frame_id')

merged_means = (
    frame_leg_means
    .merge(frame_chest_means, on='frame_id', how='outer')
    .merge(frame_face_means, on='frame_id', how='outer')
)

# print(leg_frame_means.sort_values(by ='frame_id', ascending = False).head(n = 20))

#if leg group > chest_group 



#z score, per column, as opposed to whole body
merged_means['leg_score_norm'] = (
    (merged_means["leg_score"] - merged_means["leg_score"].mean()) /
    merged_means["leg_score"].std()
)
merged_means['chest_score_norm'] = (
    (merged_means["chest_score"] - merged_means["chest_score"].mean()) /
    merged_means["chest_score"].std()
)
merged_means['face_score_norm'] = (
    (merged_means["face_score"] - merged_means["face_score"].mean()) /
    merged_means["face_score"].std()
)

merged_means['top_pick'] = merged_means[["leg_score_norm", "chest_score_norm", "face_score_norm"]].idxmax(axis=1)

######
print(merged_means.sort_values(by = "frame_id").head(n=20))

import numpy as np
import pandas as pd

# --- Safety: compute ratios if you haven't already ---
merged_means = merged_means.sort_values("frame_id").reset_index(drop=True)
merged_means[['leg_score','chest_score','face_score']] = merged_means[['leg_score','chest_score','face_score']].fillna(0.0)
merged_means['leg_to_chest']  = (merged_means['leg_score']  / merged_means['chest_score'].replace(0, np.nan)).fillna(0)
merged_means['face_to_chest'] = (merged_means['face_score'] / merged_means['chest_score'].replace(0, np.nan)).fillna(0)

# --- 1) Read ground truth (single column of TRUE/FALSE strings) ---
gt = pd.read_csv("133 manual.csv")          # adjust filename
gt = gt.rename(columns={gt.columns[0]: "full_body"})
gt["full_body"] = gt["full_body"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

# If this GT chunk is "frame 0..259", just align by POSITION to the first N frames in merged_means:
N = min(len(gt), len(merged_means))
gt = gt.iloc[:N].copy()
slice_pred = merged_means.iloc[:N].copy()

# Attach matching frame_id from predictions (which are 1..N)
gt["frame_id"] = slice_pred["frame_id"].values

# --- 2) Merge by frame_id (now they match 1:1) ---
eval_df = slice_pred.merge(gt[["frame_id","full_body"]], on="frame_id", how="inner")

# --- 3) Define a simple/full-body (legs) rule & sweep thresholds ---
def eval_rule(LEG_ABS=0.55, LEG_REL=0.80):
    pred_full = (eval_df['leg_score'] >= LEG_ABS) | (eval_df['leg_to_chest'] >= LEG_REL)
    acc = (pred_full.values == eval_df['full_body'].values).mean()
    return acc, pred_full

# quick grid search
best = (-1, None, None)
for abs_t in np.linspace(0.45, 0.75, 7):
    for rel_t in np.linspace(0.6, 1.2, 7):
        acc, _ = eval_rule(abs_t, rel_t)
        if acc > best[0]:
            best = (acc, abs_t, rel_t)

print(f"Best accuracy on first {N} frames: {best[0]:.3f} with LEG_ABS={best[1]:.2f}, LEG_REL={best[2]:.2f}")

# --- 4) Apply the best, print confusion counts ---
acc, pred = eval_rule(best[1], best[2])
tp = (( pred) & (eval_df["full_body"])).sum()
tn = ((~pred) & (~eval_df["full_body"])).sum()
fp = (( pred) & (~eval_df["full_body"])).sum()
fn = ((~pred) & (eval_df["full_body"])).sum()

print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, Accuracy={acc:.3f}")



#go through data and create body group scores dataset

#i think i need to normalize by everything




#take means per body group vs. face group per frame
# frame_means = (body_group_scores.groupby(['frame_num', 'is_face'])['keypoint_score'].mean().reset_index())

# #shift dimensions of dataset
# frame_means_pivot = frame_means.pivot(
#     index='frame_num',
#     columns='is_face',
#     values='keypoint_score'
# ).rename(columns={True: 'face_mean', False: 'body_mean'})

# #create ratio of face means to body means per frame
# frame_means_pivot['face_to_body_ratio'] = (
#     frame_means_pivot['face_mean'] / frame_means_pivot['body_mean']
# )

# #run through thresholds: if the ratio is higher than the threshold, then it is a head shot. else, it is a body shot.
# thresholds = [0, 0.01, .5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2,3,4,9]
# for threshold in thresholds:
#   headshots = (frame_means_pivot['face_to_body_ratio'] > threshold)

#   true_count = headshots.sum()
#   false_count = (~headshots).sum()
#   total = len(frame_means_pivot)

#   print(f"Threshold {threshold}: "
#         f"{true_count} headshots, {false_count} body shots "
#         f"({true_count/total:.2%} head, {false_count/total:.2%} body)")


# print(f"Taller: {true_count}, Wider: {false_count}")


# ####------------Checking against manual frame tracking-----------------
# ground_truth = pd.read_csv("133 manual.csv")

# # print(ground_truth.shape)
# # print(ground_truth.head)
# # print(ground_truth.columns)

# # Threshold sweep
# thresholds = [0, 0.01, .5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,2,3,4,9]


# print(f"Size Predictions: {frame_means_pivot.shape}")
# print(f"Ground Truth Predictions: {ground_truth.shape}")

# results = []

# for threshold in thresholds:
#     preds = frame_means_pivot['face_to_body_ratio'] < threshold

#     correct = (preds.values == ground_truth['full_body'].values).sum()
#     accuracy = correct / len(ground_truth)

#     results.append((threshold, accuracy))

# results_df = pd.DataFrame(results, columns=['threshold', 'accuracy'])
# best = results_df.loc[results_df['accuracy'].idxmax()]

# print("Best threshold:", best['threshold'], "with accuracy:", best['accuracy'])
# print(results_df)




