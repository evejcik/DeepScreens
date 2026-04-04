import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd

import re

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
    # json_file_path = Path(json_path)
    p = Path(json_path)
    p_dict = {
                'file_name': p.name,
                'parent_dir': p.parent.name,
                'full_path': p.parent
            }

    print(f"DEBUG - json_path: {json_path}")
    print(f"DEBUG - p.name: {p.name}")
    print(f"DEBUG - p.parent.name: {p.parent.name}")
    print(f"DEBUG - p_dict: {p_dict}")
    #returns dataset containing per frame, array of name instances
    with open(p, 'r') as j_file:
        return json.load(j_file), p_dict


def frame_to_instances_map(data):
    #takes in json data
    #returns dictionary of each frame to array of which instance index (for fast look up)

    frame_map = {}

    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1 #WAS PREVIOUSLY DELAYED 
        instances = frame.get('instances', [])

        frame_map[frame_id] = instances

    return frame_map


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
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    txt_y = y1
    cv2.putText(
        img, label, (x1, txt_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        color, 2,           # white, thick enough to be bright
        cv2.LINE_AA
    )


def color_for_inst(instance_ind):
    if instance_ind == 0:
        r,g,b = 255, 255, 255
    
    if instance_ind == 1:
        r,g,b = 255, 230, 0

    # deterministic per instance index (BGR)
    # b = (97 * instance_ind + 29) % 256
    # g = (17 * instance_ind + 91) % 256
    # r = (37 * instance_ind + 53) % 256


    # deterministic but amplified (max 255)
    # b = min((97 * instance_ind + 29) % 256 * 1.5, 255)
    # g = min((17 * instance_ind + 91) % 256 * 1.5, 255)
    # r = min((37 * instance_ind + 53) % 256 * 1.5, 255)
    return (int(b), int(g), int(r))
    return (b, g, r)


def _segment_start_from_path(path: str) -> int:
    """
    Extract the first integer after the word “segment_” in a filename.
    Example:  “…/segment_4662_5965.mp4” → 4662
    """
    m = re.search(r"segment_(\d+)", Path(path).name)
    return int(m.group(1)) if m else 0

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

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

    seen_keys = set()
    
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        instances = frame.get('instances', [])
        frame_count += 1
        
        for instance_ind, instance in enumerate(instances):
            instance_count += 1
            track_id = instance.get('track_id', None)
            keypoints = instance.get('keypoints', [])
            confidences = instance.get('keypoint_scores', [])

            key = (frame_id, instance_ind, track_id)

            if key in seen_keys:
                print(f"DUPLICATE: frame_id={frame_id}, instance_id={instance_ind}, track_id={track_id}")
            seen_keys.add(key)
            
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

def resize_frame_to_match(frame1, frame2):
    """Resize frame2 to match frame1's dimensions"""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    return frame2


    """Resize frame2 to match frame1's dimensions"""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    return frame2

def main(mp4_path, json_path, start, end, create_new_df, video_nobbox, start_nobbox, output_path):

    cap = cv2.VideoCapture(mp4_path)

    # video‑properties we need for the writer
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")   # works for .mp4

    writer = None
    if output_path is not None:
        # make sure the parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Writing boxed video to: {output_path}")

    credentials_path = Path('Google Cloud Credentials/credentials.json')
    json_data, json_dict = load_json(json_path)
    meta = json_data['meta_info_3d'] 
    instances_map = frame_to_instances_map(json_data)
    cap = cv2.VideoCapture(mp4_path)
    
    cap_nobbox = None
    if video_nobbox is not None:
        cap_nobbox = cv2.VideoCapture(video_nobbox)

    
    video_shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    
    frame_id = 0
    if start is None:
        start = 0
    frame_id = start
    
    frame_id = 0
    if start is None:
        start = 0
    frame_id = start
    
    frame_id_nobbox = 0
    fps_ratio = 1.0
    frame_id_nobbox_fractional = 0.0

    segment_start_frame = 0

    if video_nobbox is not None:
        cap_nobbox = cv2.VideoCapture(video_nobbox)
        if start_nobbox is None:
            start_nobbox = 0
        segment_start_frame = _segment_start_from_path(mp4_path)
        fps_main = cap.get(cv2.CAP_PROP_FPS)
        fps_nobbox = cap_nobbox.get(cv2.CAP_PROP_FPS)
        fps_ratio = fps_nobbox / fps_main
        time_offset = (segment_start_frame + start) / fps_main
        frame_id_nobbox = int(time_offset * fps_nobbox)
        frame_id_nobbox_fractional = time_offset * fps_nobbox - frame_id_nobbox
    else:
        cap_nobbox = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if end is None or end < 0:
        end = total - 1
    data = "rows_df.csv"
    if create_new_df == 1:
        rows = []
        if os.path.exists(data):
            os.remove(data)
            print(f"{data} has been deleted.")
        df = new_df(json_data, json_data['meta_info_3d']['keypoint_id2name'], json_data['meta_info_3d']['lower_body_ids'])
        df.to_csv(f"{json_dict.get('parent_dir')}_{json_dict.get('file_name')}.csv", index=False)
        df_name = f"{json_dict.get('parent_dir')}_{json_dict.get('file_name')}.csv"
        print(f"Created new dataset {df_name} with {len(df)} rows. Columns = [{df.columns}]")
        create_new_df = 0
    
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        if frame_id > end:
            break
        
        ok, frame = cap.read()
        if not ok:
            break
        
        # Read second video frame if available
        frame_nobbox = None
        if cap_nobbox is not None:
            cap_nobbox.set(cv2.CAP_PROP_POS_FRAMES, frame_id_nobbox)
            ok_nobbox, frame_nobbox = cap_nobbox.read()
        
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
            raw_bbox = instance['bbox']
            vis_bbox = visualiser_bbox_from_json(
                raw_bbox,
                meta=meta,
                video_shape=video_shape)
            label = f"Instance: {instance_ind}" if track_id is None else f"Frame: {frame_id}, Instance: {instance_ind} out of {len(instances) - 1} Track Id: {track_id}"
            draw_bbox_and_label(frame, instance=instance, instance_ind=instance_ind, label=label)
            

        if writer is not None:
            # `frame` already contains the drawn rectangles / labels
            writer.write(frame)
        
                # Concatenate or display single frame
        if frame_nobbox is not None:
            # Resize frame_nobbox to match frame dimensions
            frame_nobbox = resize_frame_to_match(frame, frame_nobbox)
            display_frame = cv2.vconcat([frame, frame_nobbox])
            text_y = display_frame.shape[0] // 2
            cv2.putText(
                display_frame,
                f"Frame: {frame_id}",
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                display_frame,
                f"Frame: {frame_id_nobbox}",
                (10, text_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        else:
            display_frame = frame
        
        cv2.imshow("overlay", display_frame)
        key = cv2.waitKeyEx(0)

        if key == ord('q'):
            break
        
        if key == ord(" "):                 # forward one frame
            frame_id        += 1
            if cap_nobbox is not None:
                frame_id_nobbox_fractional += fps_ratio
                frame_id_nobbox += int(frame_id_nobbox_fractional)
                frame_id_nobbox_fractional -= int(frame_id_nobbox_fractional)
            continue
        elif key == ord('a'):               # back one frame
            frame_id = max(0, frame_id - 1)
            if cap_nobbox is not None:
                segment_start_offset = int((segment_start_frame + start) / fps_ratio)
                frame_id_nobbox_fractional -= fps_ratio
                frame_id_nobbox += int(frame_id_nobbox_fractional)
                frame_id_nobbox_fractional -= int(frame_id_nobbox_fractional)
                frame_id_nobbox = max(segment_start_offset, frame_id_nobbox)
            continue
    
    cv2.destroyAllWindows()
    if cap_nobbox is not None:
        cap_nobbox.release()
    if writer is not None:
        writer.release()
        print("[INFO] Finished writing video.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--json", required = True)
    ap.add_argument("--mp4", required = True)
    ap.add_argument("--end", type = int)
    ap.add_argument("--create_new_df", type = int)
    ap.add_argument("--start", type = int)
    ap.add_argument("--video_nobbox", default = None)
    ap.add_argument("--start_nobbox", type=int, default=0)
    ap.add_argument("--output_path", help = "Path for where video with drawn bounding boxes will be.")

    args = ap.parse_args()
    main(
        args.mp4, #regular vid with bboxes path
        args.json, 
        args.start, #regular vid with bboxes start frame
        args.end, 
        args.create_new_df, 
        args.video_nobbox, #vid without bboxes path
        args.start_nobbox, #regular vid without bboxes start frame
        args.output_path
        )

