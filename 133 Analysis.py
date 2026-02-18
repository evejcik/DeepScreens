import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
from pprint import pprint
from IPython.display import clear_output
import seaborn as sns

#body points
total_body_group_dict= {
			"0": "nose",
			"1": "left_eye",
			"2": "right_eye",
			"3": "left_ear",
			"4": "right_ear",
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
			"22": "right_heel",
			"23": "face-0",
			"24": "face-1",
			"25": "face-2",
			"26": "face-3",
			"27": "face-4",
			"28": "face-5",
			"29": "face-6",
			"30": "face-7",
			"31": "face-8",
			"32": "face-9",
			"33": "face-10",
			"34": "face-11",
			"35": "face-12",
			"36": "face-13",
			"37": "face-14",
			"38": "face-15",
			"39": "face-16",
			"40": "face-17",
			"41": "face-18",
			"42": "face-19",
			"43": "face-20",
			"44": "face-21",
			"45": "face-22",
			"46": "face-23",
			"47": "face-24",
			"48": "face-25",
			"49": "face-26",
			"50": "face-27",
			"51": "face-28",
			"52": "face-29",
			"53": "face-30",
			"54": "face-31",
			"55": "face-32",
			"56": "face-33",
			"57": "face-34",
			"58": "face-35",
			"59": "face-36",
			"60": "face-37",
			"61": "face-38",
			"62": "face-39",
			"63": "face-40",
			"64": "face-41",
			"65": "face-42",
			"66": "face-43",
			"67": "face-44",
			"68": "face-45",
			"69": "face-46",
			"70": "face-47",
			"71": "face-48",
			"72": "face-49",
			"73": "face-50",
			"74": "face-51",
			"75": "face-52",
			"76": "face-53",
			"77": "face-54",
			"78": "face-55",
			"79": "face-56",
			"80": "face-57",
			"81": "face-58",
			"82": "face-59",
			"83": "face-60",
			"84": "face-61",
			"85": "face-62",
			"86": "face-63",
			"87": "face-64",
			"88": "face-65",
			"89": "face-66",
			"90": "face-67",
			"91": "left_hand_root",
			"92": "left_thumb1",
			"93": "left_thumb2",
			"94": "left_thumb3",
			"95": "left_thumb4",
			"96": "left_forefinger1",
			"97": "left_forefinger2",
			"98": "left_forefinger3",
			"99": "left_forefinger4",
			"100": "left_middle_finger1",
			"101": "left_middle_finger2",
			"102": "left_middle_finger3",
			"103": "left_middle_finger4",
			"104": "left_ring_finger1",
			"105": "left_ring_finger2",
			"106": "left_ring_finger3",
			"107": "left_ring_finger4",
			"108": "left_pinky_finger1",
			"109": "left_pinky_finger2",
			"110": "left_pinky_finger3",
			"111": "left_pinky_finger4",
			"112": "right_hand_root",
			"113": "right_thumb1",
			"114": "right_thumb2",
			"115": "right_thumb3",
			"116": "right_thumb4",
			"117": "right_forefinger1",
			"118": "right_forefinger2",
			"119": "right_forefinger3",
			"120": "right_forefinger4",
			"121": "right_middle_finger1",
			"122": "right_middle_finger2",
			"123": "right_middle_finger3",
			"124": "right_middle_finger4",
			"125": "right_ring_finger1",
			"126": "right_ring_finger2",
			"127": "right_ring_finger3",
			"128": "right_ring_finger4",
			"129": "right_pinky_finger1",
			"130": "right_pinky_finger2",
			"131": "right_pinky_finger3",
			"132": "right_pinky_finger4"
}
keypoint_name2id = {
			"nose": 0,
			"left_eye": 1,
			"right_eye": 2,
			"left_ear": 3,
			"right_ear": 4,
			"left_shoulder": 5,
			"right_shoulder": 6,
			"left_elbow": 7,
			"right_elbow": 8,
			"left_wrist": 9,
			"right_wrist": 10,
			"left_hip": 11,
			"right_hip": 12,
			"left_knee": 13,
			"right_knee": 14,
			"left_ankle": 15,
			"right_ankle": 16,
			"left_big_toe": 17,
			"left_small_toe": 18,
			"left_heel": 19,
			"right_big_toe": 20,
			"right_small_toe": 21,
			"right_heel": 22,
			"face-0": 23,
			"face-1": 24,
			"face-2": 25,
			"face-3": 26,
			"face-4": 27,
			"face-5": 28,
			"face-6": 29,
			"face-7": 30,
			"face-8": 31,
			"face-9": 32,
			"face-10": 33,
			"face-11": 34,
			"face-12": 35,
			"face-13": 36,
			"face-14": 37,
			"face-15": 38,
			"face-16": 39,
			"face-17": 40,
			"face-18": 41,
			"face-19": 42,
			"face-20": 43,
			"face-21": 44,
			"face-22": 45,
			"face-23": 46,
			"face-24": 47,
			"face-25": 48,
			"face-26": 49,
			"face-27": 50,
			"face-28": 51,
			"face-29": 52,
			"face-30": 53,
			"face-31": 54,
			"face-32": 55,
			"face-33": 56,
			"face-34": 57,
			"face-35": 58,
			"face-36": 59,
			"face-37": 60,
			"face-38": 61,
			"face-39": 62,
			"face-40": 63,
			"face-41": 64,
			"face-42": 65,
			"face-43": 66,
			"face-44": 67,
			"face-45": 68,
			"face-46": 69,
			"face-47": 70,
			"face-48": 71,
			"face-49": 72,
			"face-50": 73,
			"face-51": 74,
			"face-52": 75,
			"face-53": 76,
			"face-54": 77,
			"face-55": 78,
			"face-56": 79,
			"face-57": 80,
			"face-58": 81,
			"face-59": 82,
			"face-60": 83,
			"face-61": 84,
			"face-62": 85,
			"face-63": 86,
			"face-64": 87,
			"face-65": 88,
			"face-66": 89,
			"face-67": 90,
			"left_hand_root": 91,
			"left_thumb1": 92,
			"left_thumb2": 93,
			"left_thumb3": 94,
			"left_thumb4": 95,
			"left_forefinger1": 96,
			"left_forefinger2": 97,
			"left_forefinger3": 98,
			"left_forefinger4": 99,
			"left_middle_finger1": 100,
			"left_middle_finger2": 101,
			"left_middle_finger3": 102,
			"left_middle_finger4": 103,
			"left_ring_finger1": 104,
			"left_ring_finger2": 105,
			"left_ring_finger3": 106,
			"left_ring_finger4": 107,
			"left_pinky_finger1": 108,
			"left_pinky_finger2": 109,
			"left_pinky_finger3": 110,
			"left_pinky_finger4": 111,
			"right_hand_root": 112,
			"right_thumb1": 113,
			"right_thumb2": 114,
			"right_thumb3": 115,
			"right_thumb4": 116,
			"right_forefinger1": 117,
			"right_forefinger2": 118,
			"right_forefinger3": 119,
			"right_forefinger4": 120,
			"right_middle_finger1": 121,
			"right_middle_finger2": 122,
			"right_middle_finger3": 123,
			"right_middle_finger4": 124,
			"right_ring_finger1": 125,
			"right_ring_finger2": 126,
			"right_ring_finger3": 127,
			"right_ring_finger4": 128,
			"right_pinky_finger1": 129,
			"right_pinky_finger2": 130,
			"right_pinky_finger3": 131,
			"right_pinky_finger4": 132
}
upper_body_ids = [
			0,
			1,
			2,
			3,
			4,
			5,
			6,
			7,
			8,
			9,
			10,
			23,
			24,
			25,
			26,
			27,
			28,
			29,
			30,
			31,
			32,
			33,
			34,
			35,
			36,
			37,
			38,
			39,
			40,
			41,
			42,
			43,
			44,
			45,
			46,
			47,
			48,
			49,
			50,
			51,
			52,
			53,
			54,
			55,
			56,
			57,
			58,
			59,
			60,
			61,
			62,
			63,
			64,
			65,
			66,
			67,
			68,
			69,
			70,
			71,
			72,
			73,
			74,
			75,
			76,
			77,
			78,
			79,
			80,
			81,
			82,
			83,
			84,
			85,
			86,
			87,
			88,
			89,
			90
]
lower_body_ids =[
			11,
			12,
			13,
			14,
			15,
			16,
			17,
			18,
			19,
			20,
			21,
			22
  ]

#read in file
with open('133_test.json', 'r') as j_file:
    data = json.load(j_file)

#go through data and create body group scores dataset
rows = []

for frame in data["instance_info"]:
    frame_id = frame["frame_id"]
    for char_num, instance in enumerate(frame["instances"]):
        keypoint_scores = instance["keypoint_scores"]
        for kp_score, value in enumerate(keypoint_scores):
            body_group = total_body_group_dict[str(kp_score)]
            rows.append((body_group, value, char_num, frame_id))


body_group_scores = pd.DataFrame(rows, columns=['body_group', 'keypoint_score', 'char_num', 'frame_num'])

#sort rows into face group vs. body group
face_group = {'left_eye', 'right_eye', 'right_ear', 'left_ear', 'nose'}

for value in keypoint_name2id.keys():
  if value.startswith("face"):
    face_group.add(value)

body_group_scores['is_face'] = body_group_scores['body_group'].isin(face_group)

#take means per body group vs. face group per frame
frame_means = (body_group_scores.groupby(['frame_num', 'is_face'])['keypoint_score'].mean().reset_index())

#shift dimensions of dataset
frame_means_pivot = frame_means.pivot(
    index='frame_num',
    columns='is_face',
    values='keypoint_score'
).rename(columns={True: 'face_mean', False: 'body_mean'})

#create ratio of face means to body means per frame
frame_means_pivot['face_to_body_ratio'] = (
    frame_means_pivot['face_mean'] / frame_means_pivot['body_mean']
)

#run through thresholds: if the ratio is higher than the threshold, then it is a head shot. else, it is a body shot.
thresholds = [0, 0.01, .5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2,3,4,9]
for threshold in thresholds:
  headshots = (frame_means_pivot['face_to_body_ratio'] > threshold)

  true_count = headshots.sum()
  false_count = (~headshots).sum()
  total = len(frame_means_pivot)

  print(f"Threshold {threshold}: "
        f"{true_count} headshots, {false_count} body shots "
        f"({true_count/total:.2%} head, {false_count/total:.2%} body)")


print(f"Taller: {true_count}, Wider: {false_count}")


####------------Checking against manual frame tracking-----------------
ground_truth = pd.read_csv("133 manual.csv")

# print(ground_truth.shape)
# print(ground_truth.head)
# print(ground_truth.columns)

# Threshold sweep
thresholds = [0, 0.01, .5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,2,3,4,9]


print(f"Size Predictions: {frame_means_pivot.shape}")
print(f"Ground Truth Predictions: {ground_truth.shape}")

results = []

for threshold in thresholds:
    preds = frame_means_pivot['face_to_body_ratio'] < threshold

    correct = (preds.values == ground_truth['full_body'].values).sum()
    accuracy = correct / len(ground_truth)

    results.append((threshold, accuracy))

results_df = pd.DataFrame(results, columns=['threshold', 'accuracy'])
best = results_df.loc[results_df['accuracy'].idxmax()]

print("Best threshold:", best['threshold'], "with accuracy:", best['accuracy'])
print(results_df)


