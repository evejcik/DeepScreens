import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd
import re
import gspread
from google.oauth2.service_account import Credentials


def load_json(json_path):
    p = Path(json_path)
    p_dict = {
        'file_name': p.name,
        'parent_dir': p.parent.name,
        'full_path': p.parent
    }
    with open(p, 'r') as j_file:
        return json.load(j_file), p_dict


def frame_to_instances_map(data):
    frame_map = {}
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        frame_map[frame_id] = frame.get('instances', [])
    return frame_map


def frame_to_joints_map(data):
    joint_map = {}
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        frame_map = {}
        for instance_id, instance in enumerate(frame.get('instances', [])):
            joints = {}
            for joint_id, (x, y) in enumerate(instance.get('keypoints', [])):
                joints[joint_id] = {'x': x, 'y': y}
            frame_map[instance_id] = joints
        joint_map[frame_id] = frame_map
    return joint_map


def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


def color_for_inst(idx):
    if idx == 0:
        return (0, 0, 255)
    elif idx == 1:
        return (0, 230, 255)
    else:
        b = min((97 * idx + 29) % 256 * 1.5, 255)
        g = min((17 * idx + 91) % 256 * 1.5, 255)
        r = min((37 * idx + 53) % 256 * 1.5, 255)
        return (int(b), int(g), int(r))


def make_transform(content_w, content_h, full_w, full_h,
                   offset_x=0, offset_y=0):
    t = {
        'scale_x':  full_w / content_w,
        'scale_y':  full_h / content_h,
        'offset_x': offset_x,
        'offset_y': offset_y,
    }
    print(f"[TRANSFORM] scale_x={t['scale_x']:.4f} scale_y={t['scale_y']:.4f} "
          f"offset_x={offset_x} offset_y={offset_y}")
    return t


def apply_transform(x, y, t):
    x_out = int(x * t['scale_x'] + t['offset_x'])
    y_out = int(y * t['scale_y'] + t['offset_y'])
    return x_out, y_out


def draw_bbox_and_label(img, instance, instance_ind, label, t, show_bbox=True):
    if not show_bbox:
        return
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, instance['bbox'][:4])

    x1, y1 = apply_transform(x1, y1, t)
    x2, y2 = apply_transform(x2, y2, t)

    x1 = clamp_int(x1, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y2 = clamp_int(y2, 0, h - 1)

    color = color_for_inst(instance_ind)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_joint_bbox(img, x, y, color, area=32):
    half = area // 2
    cv2.rectangle(img, (x - half, y - half), (x + half, y + half), color, 2)


def get_x_y_from_inst(joints_map, frame_id, instance_id, joint_name, keypoint_name2id):
    joint_id = keypoint_name2id.get(joint_name)
    if joint_id is None:
        raise ValueError(f"Joint '{joint_name}' not found in keypoint_name2id")
    pt = joints_map[frame_id][instance_id][joint_id]
    return int(pt['x']), int(pt['y'])


def _segment_start_from_path(path: str) -> int:
    m = re.search(r"segment_(\d+)", Path(path).name)
    return int(m.group(1)) if m else 0


def resize_frame_to_match(frame1, frame2):
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    return frame2


def new_df(data, keypoint_id2name, keypoint_name2id, lower_body_ids, joint):
    rows = []
    seen_keys = set()
    for frame in data['instance_info']:
        frame_id = int(frame['frame_id']) - 1
        for instance_ind, instance in enumerate(frame.get('instances', [])):
            track_id = instance.get('track_id', None)
            keypoints = instance.get('keypoints', [])
            confidences = instance.get('keypoint_scores', [])
            key = (frame_id, instance_ind, track_id)
            if key in seen_keys:
                print(f"DUPLICATE: frame_id={frame_id}, instance_id={instance_ind}, track_id={track_id}")
            seen_keys.add(key)
            for joint_id, keypoint in enumerate(keypoints):
                if joint_id in lower_body_ids:
                    joint_name = keypoint_id2name.get(str(joint_id), f"joint_{joint_id}")
                    x, y = keypoint[0], keypoint[1]
                    confidence = confidences[joint_id] if joint_id < len(confidences) else None
                    rows.append({
                        'frame_id': frame_id, 'instance_id': instance_ind,
                        'track_id': track_id, 'joint_id': joint_id,
                        'joint_name': joint_name,
                        'visibility_category': None, 'occlusion_severity': None,
                        'occlusion_reason': None, 'annotator_confidence': None,
                        'reason_for_low_confidence': None, 'valid': None, 'notes': None,
                        'x': x, 'y': y, 'mmpose_confidence': confidence,
                    })
    return pd.DataFrame(rows)


def main(mp4_path, json_path, start, end, create_new_df_flag,
         video_nobbox, start_nobbox, output_path, joint,
         show_bbox, show_joint_bbox,
         content_w, content_h, offset_x, offset_y):

    cap    = cv2.VideoCapture(mp4_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    t = make_transform(content_w, content_h, width, height, offset_x, offset_y)

    writer = None
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Writing to: {output_path}")

    json_data, json_dict = load_json(json_path)
    meta          = json_data['meta_info']   # 2D coco_wholebody, 133 keypoints
    instances_map = frame_to_instances_map(json_data)
    joints_map    = frame_to_joints_map(json_data)

    cap = cv2.VideoCapture(mp4_path)

    if start is None:
        start = 0
    video_frame_offset = start
    frame_id = 0

    cap_nobbox                 = None
    fps_ratio                  = 1.0
    frame_id_nobbox            = 0
    frame_id_nobbox_fractional = 0.0
    segment_start_frame        = 0

    if video_nobbox is not None:
        cap_nobbox = cv2.VideoCapture(video_nobbox)
        if start_nobbox is None:
            start_nobbox = 0
        segment_start_frame = _segment_start_from_path(mp4_path)
        fps_main   = cap.get(cv2.CAP_PROP_FPS)
        fps_nobbox = cap_nobbox.get(cv2.CAP_PROP_FPS)
        fps_ratio  = fps_nobbox / fps_main
        time_offset                = (segment_start_frame + video_frame_offset) / fps_main
        frame_id_nobbox            = int(time_offset * fps_nobbox)
        frame_id_nobbox_fractional = time_offset * fps_nobbox - frame_id_nobbox

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if end is None or end < 0:
        end = total - 1

    if create_new_df_flag == 1:
        if os.path.exists("rows_df.csv"):
            os.remove("rows_df.csv")
        df = new_df(json_data,
                    meta['keypoint_id2name'], meta['keypoint_name2id'],
                    meta['lower_body_ids'], joint)
        df_name = f"{json_dict['parent_dir']}_{json_dict['file_name']}.csv"
        df.to_csv(df_name, index=False)
        print(f"Created {df_name} with {len(df)} rows.")

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
            _, frame_nobbox = cap_nobbox.read()

        instances = instances_map.get(frame_id, [])

        cv2.putText(frame,
                    f"Frame: {frame_id + video_frame_offset}  Instances: {len(instances)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        for instance_ind, instance in enumerate(instances):
            track_id = instance.get('track_id', None)
            color    = color_for_inst(instance_ind)
            label    = (f"Instance: {instance_ind}" if track_id is None
                        else f"Instance: {instance_ind} Track: {track_id}")

            draw_bbox_and_label(frame, instance, instance_ind, label,
                                t=t, show_bbox=show_bbox)

            if joint is not None and show_joint_bbox:
                x, y     = get_x_y_from_inst(joints_map, frame_id, instance_ind,
                                              joint, meta['keypoint_name2id'])
                x_t, y_t = apply_transform(x, y, t)
                draw_joint_bbox(frame, x_t, y_t, color=color)
                cv2.circle(frame, (x_t, y_t), 5, color, -1)

        if writer is not None:
            writer.write(frame)

        if frame_nobbox is not None:
            frame_nobbox  = resize_frame_to_match(frame, frame_nobbox)
            display_frame = cv2.vconcat([frame, frame_nobbox])
            text_y = display_frame.shape[0] // 2
            cv2.putText(display_frame, f"Frame: {frame_id}",
                        (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Frame: {frame_id_nobbox}",
                        (10, text_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        else:
            display_frame = frame

        cv2.imshow("overlay", display_frame)
        key = cv2.waitKeyEx(0)

        if key == ord('q'):
            break
        elif key == ord(' '):
            frame_id += 1
            if cap_nobbox is not None:
                frame_id_nobbox_fractional += fps_ratio
                frame_id_nobbox            += int(frame_id_nobbox_fractional)
                frame_id_nobbox_fractional -= int(frame_id_nobbox_fractional)
        elif key == ord('a'):
            frame_id = max(0, frame_id - 1)
            if cap_nobbox is not None:
                segment_start_offset        = int((segment_start_frame + start) / fps_ratio)
                frame_id_nobbox_fractional -= fps_ratio
                frame_id_nobbox            += int(frame_id_nobbox_fractional)
                frame_id_nobbox_fractional -= int(frame_id_nobbox_fractional)
                frame_id_nobbox             = max(segment_start_offset, frame_id_nobbox)

    cv2.destroyAllWindows()
    if cap_nobbox is not None:
        cap_nobbox.release()
    if writer is not None:
        writer.release()
        print("[INFO] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",            required=True)
    ap.add_argument("--mp4",             required=True)
    ap.add_argument("--start",           type=int)
    ap.add_argument("--end",             type=int)
    ap.add_argument("--create_new_df",   type=int, default=0)
    ap.add_argument("--video_nobbox",    default=None)
    ap.add_argument("--start_nobbox",    type=int, default=0)
    ap.add_argument("--output_path",     default=None)
    ap.add_argument("--joint",           default=None)
    ap.add_argument("--show_bbox",       type=int, default=1)
    ap.add_argument("--show_joint_bbox", type=int, default=1)
    ap.add_argument("--segment_mp4",     default=None,
                    help="Unused — kept for CLI compatibility")
    # These defaults are derived from the fixed segment geometry:
    # - Segments are always 1280x640 with white letterbox top=140, bottom=500,
    #   giving content height=359. Content width=650 was empirically confirmed
    #   by matching bbox widths on the full 1920x1080 movie (scale_x≈2.95≈scale_y≈3.008).
    # - offset_x=10 was empirically confirmed from click correspondence.
    # Override on CLI only if using segments with different geometry.
    ap.add_argument("--content_w",       type=int, default=650,
                    help="Width of pose-estimator input region in segment (default: 650)")
    ap.add_argument("--content_h",       type=int, default=359,
                    help="Height of pose-estimator input region in segment (default: 359)")
    ap.add_argument("--offset_x",        type=int, default=10,
                    help="Pixels to shift all annotations right in full-movie space (default: 10)")
    ap.add_argument("--offset_y",        type=int, default=0,
                    help="Pixels to shift all annotations down in full-movie space (default: 0)")

    args = ap.parse_args()
    main(
        args.mp4, args.json,
        args.start, args.end,
        args.create_new_df,
        args.video_nobbox, args.start_nobbox,
        args.output_path,
        args.joint,
        bool(args.show_bbox),
        bool(args.show_joint_bbox),
        args.content_w, args.content_h,
        args.offset_x, args.offset_y,
    )