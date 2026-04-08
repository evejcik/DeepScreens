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

def load_json(json_path):
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
    with open(p, 'r') as j_file:
        return json.load(j_file), p_dict


def frame_to_instances_map(data):
    frame_map = {}
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        instances = frame.get('instances', [])
        frame_map[frame_id] = instances
    return frame_map


def keypoints2D(instance):
    kps = np.array(instance['keypoints'])
    scores = np.array(instance.get('keypoint_scores', []))
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

def draw_bbox_and_label(img, instance, instance_ind, label, show_bbox=True):
    if not show_bbox:
        return
    h, w = img.shape[:2]
    bbox = return_bbox(instance)

    x1, y1, x2, y2 = bbox
    x1 = clamp_int(x1, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y2 = clamp_int(y2, 0, h - 1)

    color = color_for_inst(instance_ind)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    cv2.putText(
        img, label, (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        color, 2,
        cv2.LINE_AA
    )


def color_for_inst(instance_ind):
    if instance_ind == 0:
        r, g, b = 255, 0, 0
    elif instance_ind == 1:
        r, g, b = 255, 230, 0
    else:
        b = min((97 * instance_ind + 29) % 256 * 1.5, 255)
        g = min((17 * instance_ind + 91) % 256 * 1.5, 255)
        r = min((37 * instance_ind + 53) % 256 * 1.5, 255)
    return (int(b), int(g), int(r))


def _segment_start_from_path(path: str) -> int:
    m = re.search(r"segment_(\d+)", Path(path).name)
    return int(m.group(1)) if m else 0

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def visualiser_bbox_from_json(json_bbox, meta, video_shape=None):
    if isinstance(meta, dict) and 'stats_info' in meta:
        stats   = meta['stats_info']
        centre  = np.array(stats['bbox_center'], dtype=np.float32)
        scale   = float(stats['bbox_scale'])
    else:
        if video_shape is None:
            raise ValueError("'stats_info' missing in meta and video_shape not supplied.")
        w, h = video_shape[0], video_shape[1]
        centre = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale  = float(max(w, h))

    raw = np.asarray(json_bbox, dtype=np.float32).reshape(-1)
    if raw.size != 4:
        raise ValueError(f"bbox must contain exactly 4 numbers, got {raw.tolist()}")
    pts = raw.reshape(2, 2)

    norm     = (pts - centre) / scale
    vis_pts  = norm * scale + centre
    vis_bbox = vis_pts.reshape(4)

    return vis_bbox.tolist()

def draw_joint_bbox(img, x, y, color, area=32):
    half_area = int(area / 2)
    cv2.rectangle(img, (x - half_area, y - half_area), (x + half_area, y + half_area), color, 2)

def get_x_y_from_inst(joints_map, frame_id, instance_id, joint_name, keypoint_id2name):
    joint_id = next((int(k) for k, v in keypoint_id2name.items() if v == joint_name), None)
    pt = joints_map[frame_id][instance_id][joint_id]
    return int(pt['x']), int(pt['y'])

def frame_to_joints_map(data):
    joint_map = {}
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        instances = frame.get('instances', [])
        frame_map = {}
        for instance_id, instance in enumerate(instances):
            keypoints = instance.get('keypoints', [])
            joints = {}
            for joint_id, (x, y) in enumerate(keypoints):
                joints[joint_id] = {'x': x, 'y': y}
            frame_map[instance_id] = joints
        joint_map[frame_id] = frame_map
    return joint_map

def new_df(data, keypoint_id2name, keypoint_name2id, lower_body_ids, joint):
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
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.sheet1
        worksheet.clear()
    except gspread.exceptions.SpreadsheetNotFound:
        spreadsheet = client.create(spreadsheet_name)
        worksheet = spreadsheet.sheet1
    
    print(f"Writing {len(df)} rows to Google Sheet...")
    worksheet.append_row(df.columns.tolist())
    for idx, row in df.iterrows():
        worksheet.append_row(row.tolist())
    
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
            'options': ['self_occlusion', 'clothing', 'external_object', 'external_character', 'atmospheric', 'hallucinated'],
            'help_text': 'Select one or more reasons'
        },
        'temporal_pattern': {
            'column': 'I',
            'options': ['first_appearance', 'persistent', 'intermittent', 'correcting', 'degrading'],
            'help_text': 'Select temporal pattern'
        },
        'annotator_confidence': {
            'column': 'J',
            'options': ['1', '2', '3', '4', '5'],
            'help_text': '1=Guessing, 5=Certain'
        },
        'reason_for_low_confidence': {
            'column': 'K',
            'options': ['motion_blur', 'low_image_resolution', 'lighting_conditions',
                        'unclear_boundary_between_body_and_clothing',
                        'unclear_boundary_between_body_and_other',
                        'multiple_impossible_interpretations',
                        'self_occlusion_vs_external_occlusion_confusion'],
            'help_text': 'when annotator_confidence < 4'
        }
    }
    
    num_rows = len(df) + 2
    print("Adding dropdown validations...")
    for field_name, rule in validation_rules.items():
        col = rule['column']
        try:
            request = {
                "setDataValidation": {
                    "range": {
                        "sheetId": worksheet.id,
                        "startRowIndex": 1,
                        "endRowIndex": num_rows,
                        "startColumnIndex": ord(col) - ord('A'),
                        "endColumnIndex": ord(col) - ord('A') + 1
                    },
                    "rule": {
                        "condition": {
                            "type": "LIST",
                            "values": [{"userEnteredValue": opt} for opt in rule['options']]
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

def main(mp4_path, json_path, start, end, create_new_df, video_nobbox, start_nobbox, output_path, joint, show_bbox, show_joint_bbox, segment_mp4):

    cap = cv2.VideoCapture(mp4_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")

    # Compute scale factors if segment_mp4 is provided
    seg_scale_x = 1.0
    seg_scale_y = 1.0
    if segment_mp4 is not None:
        seg_cap = cv2.VideoCapture(segment_mp4)
        seg_w = int(seg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        seg_h = int(seg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        seg_cap.release()
        seg_scale_x = width / seg_w
        seg_scale_y = height / seg_h
        print(f"[INFO] Scaling joints from {seg_w}x{seg_h} -> {width}x{height} (sx={seg_scale_x:.3f}, sy={seg_scale_y:.3f})")

    writer = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Writing boxed video to: {output_path}")

    credentials_path = Path('Google Cloud Credentials/credentials.json')
    json_data, json_dict = load_json(json_path)
    meta = json_data['meta_info_3d'] 
    instances_map = frame_to_instances_map(json_data)
    joints_map = frame_to_joints_map(json_data)
    cap = cv2.VideoCapture(mp4_path)
    
    cap_nobbox = None
    if video_nobbox is not None:
        cap_nobbox = cv2.VideoCapture(video_nobbox)

    video_shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    
    if start is None:
        start = 0
    video_frame_offset = start  # where to seek in the mp4
    frame_id = 0               # always 0-based for JSON lookup
    
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
        time_offset = (segment_start_frame + video_frame_offset) / fps_main
        frame_id_nobbox = int(time_offset * fps_nobbox)
        frame_id_nobbox_fractional = time_offset * fps_nobbox - frame_id_nobbox
    else:
        cap_nobbox = None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if end is None or end < 0:
        end = total - 1

    data = "rows_df.csv"
    if create_new_df == 1:
        if os.path.exists(data):
            os.remove(data)
            print(f"{data} has been deleted.")
        df = new_df(json_data, 
            json_data['meta_info_3d']['keypoint_id2name'], 
            json_data['meta_info_3d']['keypoint_name2id'], 
            json_data['meta_info_3d']['lower_body_ids'],
            joint)
        df.to_csv(f"{json_dict.get('parent_dir')}_{json_dict.get('file_name')}.csv", index=False)
        df_name = f"{json_dict.get('parent_dir')}_{json_dict.get('file_name')}.csv"
        print(f"Created new dataset {df_name} with {len(df)} rows. Columns = [{df.columns}]")
        create_new_df = 0
    
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("overlay", width, height)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id + video_frame_offset)
        
        if frame_id + video_frame_offset > end:
            break
        
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_nobbox = None
        if cap_nobbox is not None:
            cap_nobbox.set(cv2.CAP_PROP_POS_FRAMES, frame_id_nobbox)
            ok_nobbox, frame_nobbox = cap_nobbox.read()
        
        instances = instances_map.get(frame_id, [])

        cv2.putText(
            frame,
            f"Frame: {frame_id + video_frame_offset}  Instances: {len(instances)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        
        for instance_ind, instance in enumerate(instances):
            track_id = instance.get('track_id', None)
            color = color_for_inst(instance_ind)
            raw_bbox = instance['bbox']
            vis_bbox = visualiser_bbox_from_json(raw_bbox, meta=meta, video_shape=video_shape)

            label = f"Instance: {instance_ind}" if track_id is None else f"Instance: {instance_ind} Track: {track_id}"
            draw_bbox_and_label(frame, instance=instance, instance_ind=instance_ind, label=label, show_bbox=show_bbox)

            if joint is not None and show_joint_bbox:
                x, y = get_x_y_from_inst(joints_map, frame_id, instance_ind, joint, meta['keypoint_id2name'])
                
                x_scaled = int(x * seg_scale_x)
                y_scaled = int(y * seg_scale_y)
                print(f"[DEBUG] frame={frame_id} inst={instance_ind} bbox_raw={instance['bbox']} x_scaled={x_scaled} y_scaled={y_scaled} frame_shape={frame.shape}")
                draw_joint_bbox(frame, x_scaled, y_scaled, color=color)
                cv2.circle(frame, (x_scaled, y_scaled), 5, color, -1)

        if writer is not None:
            writer.write(frame)
        
        if frame_nobbox is not None:
            frame_nobbox = resize_frame_to_match(frame, frame_nobbox)
            display_frame = cv2.vconcat([frame, frame_nobbox])
            text_y = display_frame.shape[0] // 2
            cv2.putText(display_frame, f"Frame: {frame_id}", (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Frame: {frame_id_nobbox}", (10, text_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            display_frame = frame
        
        cv2.imshow("overlay", display_frame)
        key = cv2.waitKeyEx(0)

        if key == ord('q'):
            break
        if key == ord(" "):
            frame_id += 1
            if cap_nobbox is not None:
                frame_id_nobbox_fractional += fps_ratio
                frame_id_nobbox += int(frame_id_nobbox_fractional)
                frame_id_nobbox_fractional -= int(frame_id_nobbox_fractional)
            continue
        elif key == ord('a'):
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
    
    ap.add_argument("--json", required=True)
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--end", type=int)
    ap.add_argument("--create_new_df", type=int)
    ap.add_argument("--start", type=int)
    ap.add_argument("--video_nobbox", default=None)
    ap.add_argument("--start_nobbox", type=int, default=0)
    ap.add_argument("--output_path", help="Path for where video with drawn bounding boxes will be.")
    ap.add_argument("--joint", default=None)
    ap.add_argument("--show_bbox", type=int, default=1, help="1=show instance bboxes, 0=hide")
    ap.add_argument("--show_joint_bbox", type=int, default=1, help="1=show joint bbox, 0=hide")
    ap.add_argument("--segment_mp4", default=None, help="Original segment mp4 used to generate the JSON, for coordinate scaling")

    args = ap.parse_args()
    main(
        args.mp4,
        args.json,
        args.start,
        args.end,
        args.create_new_df,
        args.video_nobbox,
        args.start_nobbox,
        args.output_path,
        args.joint,
        bool(args.show_bbox),
        bool(args.show_joint_bbox),
        args.segment_mp4
    )