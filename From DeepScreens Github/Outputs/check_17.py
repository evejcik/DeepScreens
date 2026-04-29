import json

with open('my_17_json.json', 'r') as f:
# with open('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Moonlight_2016/Moonlight_2016/segment_1_1529.json', 'r') as f:
    data = json.load(f)


inst = data['instance_info'][0]['instances'][0]
print(inst['keypoints'][0])    # what does entry 0 look like?
print(inst['keypoints'][1])    # entry 1
print(inst['keypoints'][6])    # entry 6 — left_ankle, should be cleaned
print(inst['keypoints'][7])    # entry 7 — spine, not in your annotation set
print(len(inst['keypoints']))

# # import json
# d = json.load(open("../From DeepScreens Github/Outputs/my_17_json.json"))
# inst = d["instance_info"][0]["instances"][0]
# print(len(inst["keypoints"]), len(inst["keypoint_scores"]), len(inst["keypoint_interpolated"]))