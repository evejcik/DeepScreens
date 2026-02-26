import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd

import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

from google.auth import default
from google.auth.transport.requests import Request

# creds, project = default(scopes=[
#     'https://www.googleapis.com/auth/spreadsheets',
#     'https://www.googleapis.com/auth/drive'
# ])

# client = gspread.authorize(creds)


#get instances per frame id
#get 2d keypoint scores per instance per frame id
#get 3d keypoint scores per instance per frame id


# def access_dir_mp4s(root_dir_path):
#     root_dir = Path(root_dir_path)
#     for file in root_dir.iterdir():
#         # if file.suffix == '.json':
#         #     process_json(file)
#         if file.suffix == '.mp4':
#             process_mp4(file)


def load_json(json_path):
    #takes in json file
    json_file_path = Path(json_path)
    #returns dataset containing per frame, array of name instances
    with open(json_file_path, 'r') as j_file:
        return json.load(j_file)


def frame_to_instances_map(data):
    #takes in json data
    #returns dictionary of each frame to array of which instance index (for fast look up)

    frame_map = {}

    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1 #WAS PREVIOUSLY DELAYED 
        instances = frame.get('instances', [])

        frame_map[frame_id] = instances

    return frame_map

def old_df(data, keypoint_id2name, lower_body_ids):
    #takes in json data
    #returns dataframe with one row per frame per instance per joint

    rows = []

    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        instances = frame.get('instances', [])

        for instance_ind, instance in enumerate(instances):
            track_id = instance.get('track_id', None)
            keypoints = instance.get('keypoints', [])
            confidences = instance.get('keypoint_scores', [])

            for joint_id, keypoint in enumerate(keypoints):
                if joint_id in data['meta_info_3d']['lower_body_ids']:
                    joint_name = keypoint_id2name.get(str(joint_id), f"joint_{joint_id}")
                    x,y = keypoint[0], keypoint[1]
                    confidence = confidences[joint_id] if confidences else None

                    row = {
                        'frame_id': frame_id,
                        'instance_id': instance_ind,
                        'track_id': track_id,
                        'joint_id': joint_id,
                        'joint_name': joint_name,
                        
                        # Manual Columns Below
                        'visibility_category': None,
                        'occlusion_severity': None,
                        'occlusion_reason': None,
                        'temporal_pattern': None,
                        'annotator_confidence': None,
                        'reason_for_low_confidence': None,
                        'valid': None,
                        'notes': None,
                        
                        #for later analysis
                        'x': x,
                        'y': y,
                        'mmpose_confidence': confidence,
                    }
                    rows.append(row)
    df = pd.DataFrame(rows)
    return df

def keypoints2D(instance):
    kps = np.array(instance['keypoints']) #gets the list of points of hwere the keypoints are at this one instance
    scores = np.array(instance.get('keypoint_scores', [])) #gets the scores associated with these keypoints
    return scores

def keypoints3D(instance):
    kps = np.array(instance['keypoints_3d'])
    scores = np.array(instance.get('keypoint_scores_3d', []))
    return scores

def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def return_bbox(instance):
    bbox = instance['bbox']
    x1, y1, x2, y2 = map(float, bbox[:4])
    return x1, y1, x2, y2

def draw_bbox_and_label(img, instance, instance_ind, label):
    h,w = img.shape[:2]
    bbox = return_bbox(instance)

    x1, y1, x2, y2 = bbox
    x1 = clamp_int(x1, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y2 = clamp_int(y2, 0, h - 1)

    color = color_for_inst(instance_ind)
    cv2.rectangle(img, (x1, y1+140), (x2, y2+140), color, 1)

    txt_y = y1 + 160
    cv2.putText(
        img, label, (x1, txt_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        color, 2,           # white, thick enough to be bright
        cv2.LINE_AA
    )


def color_for_inst(instance_ind):
    # deterministic per instance index (BGR)
    # b = (97 * instance_ind + 29) % 256
    # g = (17 * instance_ind + 91) % 256
    # r = (37 * instance_ind + 53) % 256


    # deterministic but amplified (max 255)
    b = min((97 * instance_ind + 29) % 256 * 1.5, 255)
    g = min((17 * instance_ind + 91) % 256 * 1.5, 255)
    r = min((37 * instance_ind + 53) % 256 * 1.5, 255)
    return (int(b), int(g), int(r))
    return (b, g, r)


import numpy as np

def visualiser_bbox_from_json(json_bbox, meta, video_shape=None):
    """
    Convert a raw pixel bbox (the one stored in the DeepScreens JSON) to the
    coordinate system that Pose3DVisualizer draws after its normalise →
    denormalise step.

    Parameters
    ----------
    json_bbox : list/tuple of 4 numbers   [x1, y1, x2, y2] (pixel space)
    meta      : dict – the *2‑D* dataset meta (may contain 'stats_info')
    video_shape : [width, height] – required only if meta has no 'stats_info'

    Returns
    -------
    list of 4 floats – bbox transformed into the visualiser’s internal space
    """
    # --------------------------------------------------------------
    # Determine centre and scale
    # --------------------------------------------------------------
    if isinstance(meta, dict) and 'stats_info' in meta:
        stats   = meta['stats_info']
        centre  = np.array(stats['bbox_center'], dtype=np.float32)   # (2,)
        scale   = float(stats['bbox_scale'])
    else:
        # No stats_info → use the real video centre / scale
        if video_shape is None:
            raise ValueError(
                "'stats_info' missing in meta and video_shape not supplied.")
        w, h = video_shape[0], video_shape[1]
        centre = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale  = float(max(w, h))

    # --------------------------------------------------------------
    #  Validate the bbox shape and reshape to (2,2)
    # --------------------------------------------------------------
    raw = np.asarray(json_bbox, dtype=np.float32).reshape(-1)
    if raw.size != 4:
        raise ValueError(
            f"bbox must contain exactly 4 numbers, got {raw.tolist()}")
    pts = raw.reshape(2, 2)                 # [[x1, y1], [x2, y2]]

    # --------------------------------------------------------------
    # Apply the *exact* normalise → denormalise that the visualiser uses
    # --------------------------------------------------------------
    norm      = (pts - centre) / scale       # normalise each (x,y) pair
    vis_pts   = norm * scale + centre        # denormalise → visualiser space
    vis_bbox  = vis_pts.reshape(4)           # back to flat [x1, y1, x2, y2]

    return vis_bbox.tolist()


def new_df(data, keypoint_id2name, lower_body_ids):
    rows = []
    frame_count = 0
    instance_count = 0
    joint_count = 0
    
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        instances = frame.get('instances', [])
        frame_count += 1
        
        for instance_ind, instance in enumerate(instances):
            instance_count += 1
            track_id = instance.get('track_id', None)
            keypoints = instance.get('keypoints', [])
            confidences = instance.get('keypoint_scores', [])
            
            for joint_id, keypoint in enumerate(keypoints):
                if joint_id in data['meta_info_3d']['lower_body_ids']:
                    joint_count += 1
                    joint_name = keypoint_id2name.get(str(joint_id), f"joint_{joint_id}")
                    x, y = keypoint[0], keypoint[1]
                    confidence = confidences[joint_id] if joint_id < len(confidences) else None
                    
                    row = {
                        'frame_id': frame_id,
                        'instance_id': instance_ind,
                        'track_id': track_id,
                        'joint_id': joint_id,
                        'joint_name': joint_name,
                        'visibility_category': None,
                        'occlusion_severity': None,
                        'occlusion_reason': None,
                        'temporal_pattern': None,
                        'annotator_confidence': None,
                        'reason_for_low_confidence': None,
                        'valid': None,
                        'notes': None,
                        'x': x,
                        'y': y,
                        'mmpose_confidence': confidence,
                    }
                    rows.append(row)
    
    # Print debug info
    print(f"DEBUG: Total frames: {frame_count}")
    print(f"DEBUG: Total instances: {instance_count}")
    print(f"DEBUG: Total lower body joints: {joint_count}")
    print(f"DEBUG: Expected rows (if all had lower body): {frame_count * 2 * 7}")
    
    df = pd.DataFrame(rows)
    return df


def push_dataframe_to_google_sheets(df, spreadsheet_name, json_keyfile_path):

    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(json_keyfile_path, scopes=scope)
    client = gspread.authorize(creds)
    try:
        # Open existing sheet or create new one
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.sheet1
        worksheet.clear()
    except gspread.exceptions.SpreadsheetNotFound:
        # Create new spreadsheet if it doesn't exist
        spreadsheet = client.create(spreadsheet_name)
        worksheet = spreadsheet.sheet1
    
    # Write dataframe to sheet
    print(f"Writing {len(df)} rows to Google Sheet...")
    worksheet.append_row(df.columns.tolist())
    for idx, row in df.iterrows():
        worksheet.append_row(row.tolist())
    
    # Define dropdown options 
    validation_rules = {
        'visibility_category': {
            'column': 'F',
            'options': ['0', '1', '2', '3'],
            'help_text': '0=Visible, 1=Occluded, 2=Off-screen, 3=Ambiguous'
        },
        'occlusion_severity': {
            'column': 'G',
            'options': ['0', '1', '2', '3'],
            'help_text': '0=0-25%, 1=25-50%, 2=50-75%, 3=75-100%'
        },
        'occlusion_reason': {
            'column': 'H',
            'options': [
                'self_occlusion',
                'clothing',
                'external_object',
                'external_character',
                'atmospheric',
                'hallucinated'
            ],
            'help_text': 'Select one or more reasons'
        },
        'temporal_pattern': {
            'column': 'I',
            'options': [
                'first_appearance',
                'persistent',
                'intermittent',
                'correcting',
                'degrading'
            ],
            'help_text': 'Select temporal pattern'
        },
        'annotator_confidence': {
            'column': 'J',
            'options': ['1', '2', '3', '4', '5'],
            'help_text': '1=Guessing, 5=Certain'
        },
        'reason_for_low_confidence':{
            'column': 'K',
            'options': ['motion_blur',
                        'low_image_resolution',
                        'lighting_conditions',
                        'unclear_boundary_between_body_and_clothing',
                        'unclear_boundary_between_body_and_other',
                        'multiple_impossible_interpretations',
                        'self_occlusion_vs_external_occlusion_confusion'],
            'help_text': 'when annotator_confidence < 4'
        }
    }
    
    # Get total number of rows (including header)
    num_rows = len(df) + 2
    
    # Apply data validations
    print("Adding dropdown validations...")
    for field_name, rule in validation_rules.items():
        col = rule['column']
        data_range = f"{col}2:{col}{num_rows}"
        
        try:
            request = {
                "setDataValidation": {
                    "range": {
                        "sheetId": worksheet.id,
                        "startRowIndex": 1,  # Skip header
                        "endRowIndex": num_rows,
                        "startColumnIndex": ord(col) - ord('A'),
                        "endColumnIndex": ord(col) - ord('A') + 1
                    },
                    "rule": {
                        "condition": {
                            "type": "LIST",
                            "values": [
                                {"userEnteredValue": opt} for opt in rule['options']
                            ]
                        },
                        "inputMessage": rule['help_text'],
                        "strict": True,
                        "showCustomUI": True
                    }
                }
            }
            
            spreadsheet.batch_update(request)
            print(f"✓ Added validation for {field_name}")
            
        except Exception as e:
            print(f"⚠ Could not add validation for {field_name}: {e}")
    
    print(f"\nSheet created successfully!")
    print(f"Access it here: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
    
    return spreadsheet.id


def main(mp4_path, json_path, start, end, create_new_df):
   
    # ap.add_argument("--output_dir", required = True)
    # credentials_path = Path(__file__).parent / "Google Cloud Credentials" / "credentials.json"
    credentials_path = Path('Google Cloud Credentials/credentials.json')

    json_data = load_json(json_path)
    meta = json_data['meta_info_3d'] 
    instances_map = frame_to_instances_map(json_data)
    cap = cv2.VideoCapture(mp4_path)

    video_shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    
    frame_id = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if end < 0:
        end = total - 1

    data = "rows_df.csv"

    if create_new_df == 1: ##MAKING NEW DATASET START
        rows = []

        if os.path.exists(data):
            os.remove(data)
            print(f"{data} has been deleted.")

        df = new_df(json_data, json_data['meta_info_3d']['keypoint_id2name'], json_data['meta_info_3d']['lower_body_ids'])
        df.to_csv("rows_df_test.csv", index = False)
        print(f"Created new dataset with {len(df)} rows.")

        # try:
        #     sheet_id = push_dataframe_to_google_sheets(
        #         df,
        #         spreadsheet_name="DeepScreens_Annotation",
        #         json_keyfile_path=str(credentials_path)
        #     )
        # except Exception as e:
        #     print(f"⚠ Could not push to Google Sheets: {e}")
        #     print("Continuing with CSV only...")
        create_new_df = 0
            
    ##MAKING NEW DATASET END

    
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    if start == None:
        frame_id = 0
    else: frame_id = start
    # frame_id = max(start, start_frame)
    while True:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        if frame_id > end:
            break

        ok, frame = cap.read()
        if not ok:
            break

        instances = instances_map.get(frame_id, [])

        cv2.putText(
            frame,
            f"Frame: {frame_id}  Instances: {len(instances)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        
        for instance_ind, instance in enumerate(instances):
            track_id = instance.get('track_id', None)

            raw_bbox = instance['bbox']                # <-- raw JSON box (green)
            vis_bbox = visualiser_bbox_from_json(
                raw_bbox,
                meta=meta,               # 2‑D meta (may contain stats_info)
                video_shape=video_shape) # fallback geometry when stats_info missing
            label = f"Instance: {instance_ind}" if track_id is None else f"Frame: {frame_id}, Instance: {instance_ind} out of {len(instances) - 1} Track Id: {track_id}"
            draw_bbox_and_label(frame, instance=instance, instance_ind=instance_ind, label=label)
            
            # --- OPTIONAL: draw the visualiser‑style bbox in yellow ---
            def draw_box(img, bbox, colour, thickness=2):
                h, w = img.shape[:2]
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

            

        cv2.imshow("overlay", frame)

        key = cv2.waitKeyEx(0)
        if key == ord("q"):
            break
        if key == ord(" "):
            frame_id += 1
            continue
        # elif key == 81 or key == 2555904:
        elif key == ord('a'):
            frame_id = max(0, frame_id - 1)

    
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--json", required = True)
    ap.add_argument("--mp4", required = True)
    ap.add_argument("--end", type = int, required = True)
    ap.add_argument("--create_new_df", type = int)
    ap.add_argument("--start", type = int)

    args = ap.parse_args()
    main(args.mp4, args.json, args.start, args.end, args.create_new_df)

