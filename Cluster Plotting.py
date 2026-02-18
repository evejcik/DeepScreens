
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


with open('133_test.json', 'r') as j_file: #this is for rancho conversation right
    data = json.load(j_file)


    #go through data and get the values of the keys that correlate to the keys in keypoint_id2name
rows = []

for frame in data["instance_info"]:
    frame_id = frame["frame_id"]
    for char_num, instance in enumerate(frame["instances"]):
        keypoints = instance["keypoints"]
        keypoint_scores = instance["keypoint_scores"]

        # print(f"Coordinates: {coordinates}")

        for keypoint_id, keypoint_part in keypoint_id2name.items():
            keypoint_id_num = int(keypoint_id)
            keypoint_score = keypoint_scores[keypoint_id_num]

            xyz = keypoints[keypoint_id_num]
            x = xyz[0]
            y = xyz[1]
            z = xyz[2] if len(xyz) > 2 else None
            if char_num == 0:
                rows.append((frame_id, char_num, keypoint_part, keypoint_score, x, y, z))


data133_full = pd.DataFrame(rows, columns=["frame_id", "char_num", "kp_name", "score", "x", "y", "z"])
ground_truth = pd.read_csv("rancho convo truth.csv", header = None)

ground_truth.columns = ["full_body"]
ground_truth["frame_id"] = np.arange(1, len(ground_truth) + 1)

df = data133_full.merge(ground_truth, on="frame_id", how="left")

relevant_parts = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_big_toe", "right_big_toe"
]

df_relevant = df[df['kp_name'].isin(relevant_parts)]

frame_confidence = (df_relevant.groupby(['kp_name', 'full_body'])['score'].mean().reset_index())

plt.figure(figsize = (8,5))


##Section 1: Thresholding with Confidence Values

# plt.show()
#####Checking what are the average confidence scores for toes/heels when showing vs. not showing full figure######

### so we are looking at all the relevant parts, by part, individually. calculate their average for all frames in segment.
# 
# but, we can also look at the average score for all of the relevant parts, per individual frame. if the average for the relevant parts is 
# below a certain threshold (looking like it's around 0.21 or so), then we should classify that as a non full figure. 

#furthermore, i think that certain body parts have more variance in their scores per frame. if the MSE for a single body part's keypoint for one frame
#. what is the variance/std? from the mean for this body part across all frames? 

### i think that the relevant parts are the knees and ankles (for now, lets discard the hips and toes).

#if the confidence scores for the knees and ankles are above a certain threshold, then let's see how accurate the model is.

key_joints = [
	"left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

df_keyjoints = df[df['kp_name'].isin(key_joints)]
thresholds = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
num_joints_threshold = [1,2,3,4]
results = []

for threshold in thresholds:
	for num_joints in num_joints_threshold:
		df_keyjoints['full_body_visible'] = df_keyjoints['score'] >= threshold #now we are looking at per frame per body part

		#aggregrate to frame level predictions
		full_body_count_per_frame = (df_keyjoints.groupby('frame_id')['full_body_visible'].sum()) #per frame id, how many joints are visible
		preds_full = full_body_count_per_frame >= num_joints

		gt = ground_truth.set_index("frame_id")["full_body"] #what does this do?
		# align indices just in case
		pred_full = preds_full.reindex(gt.index)

		accuracy = (pred_full == gt).mean()
		results.append((num_joints, threshold, accuracy))

results_df = pd.DataFrame(results, columns=["num_joints", "threshold", "accuracy"]).sort_values(by = ['num_joints', 'threshold'])
print(results_df)


##Section 2: Ratio-ing - let's look at how the length of the legs compares to the length of the upper body

#let's check the scores for relevant body parts for full vs. half figures:
relevant_parts = [
	'left_shoulder', 'right_shoulder',
	'left_hip', 'right_hip',
	'left_elbow', 'right_elbow',
	'nose'
]

#get the rows from the dataset that have kp_name in relevant_parts
df_relevant_upper = (df[df['kp_name'].isin(relevant_parts)])
#check plot for which sections have most variability


sns.stripplot(data=df_relevant_upper, x="kp_name", y="score", hue="full_body", legend=True, dodge = True)
plt.show()


#ratio of confidence stats: comparing the confidence of an upper body joint to the confidence of a lower body joint ->hoping to generalize across shitty frames



