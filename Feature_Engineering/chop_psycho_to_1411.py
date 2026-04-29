import json

def truncate_json(input_path, output_path, max_frame):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    data['instance_info'] = [
        frame for frame in data['instance_info']
        if frame['frame_id'] <= max_frame
    ]
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Truncated to {len(data['instance_info'])} frames.")

truncate_json('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_results/Psycho_1960/Psycho_1960/segment_319_2006.json', 
'/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_results/Psycho_1960/Psycho_1960/segment_319_1411.json', max_frame=1410)