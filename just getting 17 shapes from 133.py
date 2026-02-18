#identify which are the 17 shapes out of the 133.
#grab each of those 17 per frame.

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


with open('133_test.json', 'r') as j_file:
    data = json.load(j_file)

#go through data and get the values of the keys that correlate to the keys in keypoint_id2name
rows = []

for frame in data["instance_info"]:
    frame_id = frame["frame_id"]
    for char_num, instance in enumerate(frame["instances"]):
        keypoint_scores = instance["keypoint_scores"]

        for keypoint_id, keypoint_part in keypoint_id2name.items():
            keypoint_id_num = int(keypoint_id)
            keypoint_score = keypoint_scores[keypoint_id_num]

            if keypoint_part == "right_knee":

                rows.append((frame_id, char_num, keypoint_part, keypoint_score))



data133_to17 = pd.DataFrame(rows, columns=["frame_id", "char_num", "kp_name", "score"])

# print(kp_subset.head)

#going to look at right shoulder, left knee, nose in frame 1. 
#going to look confidence scores for: right shoulder, left knee, nose in frame 1

data133 = pd.DataFrame(rows, columns=["frame_id", "char_num", "kp_name", "score"])

data133.to_csv("Predictions.csv")


#####------------Let's look at the highest and lowest confidence scores and match them to their frames. these frames are where there is only a 
#shoulder up shot (NO toes/hips).

data133_sorted = pd.read_csv("Predictions.csv")

data133_sorted = data133_sorted.sort_values(by = "score")

data133_sorted = data133_sorted.loc[(data133_sorted['frame_id'] < 3812) & (data133_sorted['frame_id'] >1151)]


# print(data133_sorted.head(n=30))

#now let's see what the sorted descending order of averaged confidence scores per body group is. what is the body group with the highest
#average confidence score for these frames, what is the body group with the lowest average confidence score.

# data133_sorted_average = data133_sorted
rows_bodygroup = []

for frame in data["instance_info"]:
    frame_id = frame["frame_id"]
    for char_num, instance in enumerate(frame["instances"]):
        keypoint_scores = instance["keypoint_scores"]

        for keypoint_id, keypoint_part in keypoint_id2name.items():
            keypoint_id_num = int(keypoint_id)
            keypoint_score = keypoint_scores[keypoint_id_num]

            # if 1151 < frame_id < 3812:
            if frame_id == 1002:
                rows_bodygroup.append((frame_id, char_num, keypoint_part, keypoint_score))



			



data133_to17 = pd.DataFrame(rows_bodygroup, columns=["frame_id", "char_num", "kp_name", "score"])

data133_sorted_average = data133_to17.groupby('kp_name')['score'].mean()

print(data133_sorted_average.sort_values())

