"""
visualise_skeleton.py
---------------------
Annotation tool for the new reliability dataset.

Reads RAW 133-keypoint MMPose JSON, draws the H36M 17-joint skeleton on the
display video, and creates a CSV with one row per (frame, instance, joint)
for the joint specified by --joint. CSV rows store H36M joint_id and
joint_name so the resulting dataset is directly compatible with the
existing feature_engineering and classifier pipelines.

Letterbox/pillarbox detection and coordinate transform ported verbatim from
visualiser_no_offsets.py.

Annotation philosophy
---------------------
The overlay deliberately does NOT show classifier predictions, score-based
colors, or any signal that could anchor the annotator's judgment. Joints
are drawn in plain colors and the selected joint is highlighted with a
neutral bbox. The annotator sees the same visual evidence the classifier
sees, with no hint of what the classifier currently thinks.

This matters because we are annotating to fill gaps in joint coverage
(head, neck, right shoulder, etc.) where the classifier has no positive
training examples and is currently failing. Showing classifier colors
during annotation would partially reproduce that bias in the new labels.

Usage
-----
python visualise_skeleton.py \\
    --json          path/to/raw_133kp.json \\
    --mp4           path/to/display_video.mp4 \\
    --joint         head \\
    [--segment_mp4  path/to/letterboxed_segment.mp4] \\
    [--start 0] [--end 999] \\
    [--create_new_df 1] \\
    [--output_path  out.mp4] \\
    [--use_segment_offsets]   # hardcoded Tron geometry

--mp4            The video frames are read from here and drawn on.
--segment_mp4    Used ONLY for letterbox detection (first-frame scan).
                 If omitted, --mp4 is used for detection too.
--joint          H36M joint name to annotate. One of:
                   root, right_hip, right_knee, right_foot,
                   left_hip, left_knee, left_foot,
                   spine, thorax, neck_base, head,
                   left_shoulder, left_elbow, left_wrist,
                   right_shoulder, right_elbow, right_wrist
--create_new_df  1 to create a fresh CSV (overwrites if exists), 0 to skip.
--use_segment_offsets
                 Skip auto-detection; apply hardcoded Tron geometry:
                 content_w=650, content_h=359, offset_x=10.

Keyboard controls:
    s  : next frame
    a  : previous frame
    d  : skip forward 10 frames
    q  : quit

CSV schema (matches feature_engineering_csv.py expectations)
    frame_id, instance_id, track_id, joint_id, joint_name,
    x, y, mmpose_confidence,
    reliability_category, annotator_confidence, reason_for_distrust,
    dist_to_boundary, valid

Annotation columns are left empty; fill them in your spreadsheet/editor
of choice. The Google Sheets push from the original tool has been removed.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# H36M skeleton (drawn from the 133-kp COCO source via COCO_TO_H36M_DERIVE)
# ----------------------------------------------------------------------------

H36M_JOINT_NAMES = {
    0: "root",         1: "right_hip",     2: "right_knee",   3: "right_foot",
    4: "left_hip",     5: "left_knee",     6: "left_foot",    7: "spine",
    8: "thorax",       9: "neck_base",    10: "head",
    11: "left_shoulder", 12: "left_elbow", 13: "left_wrist",
    14: "right_shoulder", 15: "right_elbow", 16: "right_wrist",
}
H36M_NAME_TO_ID = {v: k for k, v in H36M_JOINT_NAMES.items()}

# Drawn skeleton bones in H36M index space.
BONE_PAIRS = [
    (1, 2), (2, 3),       # right leg
    (4, 5), (5, 6),       # left leg
    (7, 8),               # spine -> thorax
    (8, 10),              # thorax -> head
    (11, 12), (12, 13),   # left arm
    (14, 15), (15, 16),   # right arm
    (0, 1), (0, 4),       # root -> hips
    (8, 11), (8, 14),     # thorax -> shoulders
    (8, 9),               # thorax -> neck_base
]

# COCO-wholebody (133-keypoint) indices used by RTMW.
COCO_NOSE       = 0
COCO_LEFT_EYE   = 1
COCO_RIGHT_EYE  = 2
COCO_LEFT_EAR   = 3
COCO_RIGHT_EAR  = 4
COCO_LEFT_SHOULDER  = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW     = 7
COCO_RIGHT_ELBOW    = 8
COCO_LEFT_WRIST     = 9
COCO_RIGHT_WRIST    = 10
COCO_LEFT_HIP       = 11
COCO_RIGHT_HIP      = 12
COCO_LEFT_KNEE      = 13
COCO_RIGHT_KNEE     = 14
COCO_LEFT_ANKLE     = 15
COCO_RIGHT_ANKLE    = 16

# Direct H36M -> COCO mapping (used for both position and confidence).
# Derived joints are handled separately below.
H36M_DIRECT_TO_COCO = {
    1:  COCO_RIGHT_HIP,
    2:  COCO_RIGHT_KNEE,
    3:  COCO_RIGHT_ANKLE,
    4:  COCO_LEFT_HIP,
    5:  COCO_LEFT_KNEE,
    6:  COCO_LEFT_ANKLE,
    11: COCO_LEFT_SHOULDER,
    12: COCO_LEFT_ELBOW,
    13: COCO_LEFT_WRIST,
    14: COCO_RIGHT_SHOULDER,
    15: COCO_RIGHT_ELBOW,
    16: COCO_RIGHT_WRIST,
}

# ----------------------------------------------------------------------------
# Derived joints: positions and scores are computed from DIFFERENT COCO
# sources in the inference pipeline (deepscreens_rtmw_videopose.py).
# These tables mirror the inference pipeline EXACTLY; previous versions of
# this tool approximated the formulas, which produced visually wrong joint
# positions on the annotation overlay (especially spine and head).
#
# POSITION table -- mirrors convert_rtmpose133_to_h36m17_2d
#   Each entry: list of (coco_idx, weight). Position = sum(weight * coco_pos).
#   Weights must sum to 1.0 within each entry.
#
# SCORE table    -- mirrors remap_keypoint_scores_133_to_17
#   Each entry: list of coco_idx. Score = geometric_mean(coco_scores at those idx).
#   Weights are NOT used for scores in the inference pipeline.
# ----------------------------------------------------------------------------

H36M_DERIVED_POSITION = {
    # root = midpoint of hips
    0:  [(COCO_LEFT_HIP, 0.5), (COCO_RIGHT_HIP, 0.5)],
    # spine = hip_mid + 0.5 * (shoulder_mid - hip_mid)
    #       = 0.5 * hip_mid + 0.5 * shoulder_mid
    #       = 0.25 each of L_Hip, R_Hip, L_Shoulder, R_Shoulder
    7:  [(COCO_LEFT_HIP,      0.25), (COCO_RIGHT_HIP,      0.25),
         (COCO_LEFT_SHOULDER, 0.25), (COCO_RIGHT_SHOULDER, 0.25)],
    # thorax = midpoint of shoulders
    8:  [(COCO_LEFT_SHOULDER, 0.5), (COCO_RIGHT_SHOULDER, 0.5)],
    # neck_base = shoulder_mid + 0.15 * (ear_mid - shoulder_mid)
    #           = 0.85 * shoulder_mid + 0.15 * ear_mid
    #           = 0.425 * each shoulder + 0.075 * each ear
    9:  [(COCO_LEFT_SHOULDER, 0.425), (COCO_RIGHT_SHOULDER, 0.425),
         (COCO_LEFT_EAR,      0.075), (COCO_RIGHT_EAR,      0.075)],
    # head = midpoint of EARS ONLY (the inference pipeline does not use the nose)
    10: [(COCO_LEFT_EAR, 0.5), (COCO_RIGHT_EAR, 0.5)],
}

H36M_DERIVED_SCORE = {
    # root score = geometric_mean(L_Hip, R_Hip)
    0:  [COCO_LEFT_HIP, COCO_RIGHT_HIP],
    # torso score (used for spine, thorax, neck_base) = geom_mean(L_Sh, R_Sh)
    7:  [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER],
    8:  [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER],
    9:  [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER],
    # head score = geometric_mean(Nose, L_Eye, R_Eye)
    # Note: the SCORE uses different sources than the POSITION above. This is
    # intentional and matches the inference pipeline. Do not "consolidate".
    10: [COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE],
}


def _geometric_mean(values):
    """Numerically stable geometric mean. Treats any value below 1e-10 as 1e-10."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float32)
    arr = np.maximum(arr, 1e-10)
    return float(np.exp(np.mean(np.log(arr))))


def h36m_xy_and_score_from_coco(kps_coco, scores_coco, h36m_id):
    """
    Return (x, y, score) for a given H36M joint id, computed from a single
    instance's 133-kp COCO arrays. Mirrors the inference pipeline exactly.

    Direct joints: copy position and score from the corresponding COCO joint.
    Derived joints: weighted position from H36M_DERIVED_POSITION,
                    geometric-mean score from H36M_DERIVED_SCORE
                    (the two tables intentionally use different source joints
                    for some H36M ids, matching the inference pipeline).

    Returns (None, None, None) if any required source index is out of range
    of the supplied arrays.
    """
    n_kp = len(kps_coco)
    n_sc = len(scores_coco)

    if h36m_id in H36M_DIRECT_TO_COCO:
        coco_idx = H36M_DIRECT_TO_COCO[h36m_id]
        if coco_idx >= n_kp:
            return None, None, None
        x = float(kps_coco[coco_idx][0])
        y = float(kps_coco[coco_idx][1])
        s = float(scores_coco[coco_idx]) if coco_idx < n_sc else 0.0
        return x, y, s

    if h36m_id in H36M_DERIVED_POSITION:
        # Position
        pos_sources = H36M_DERIVED_POSITION[h36m_id]
        if any(i >= n_kp for (i, _) in pos_sources):
            return None, None, None
        x = sum(w * float(kps_coco[i][0]) for (i, w) in pos_sources)
        y = sum(w * float(kps_coco[i][1]) for (i, w) in pos_sources)

        # Score (different source set, geometric mean per pipeline)
        score_sources = H36M_DERIVED_SCORE.get(h36m_id, [])
        ss = [float(scores_coco[i]) for i in score_sources if i < n_sc]
        s = _geometric_mean(ss)
        return x, y, s

    return None, None, None


# ----------------------------------------------------------------------------
# Drawing constants (no score-driven coloring)
# ----------------------------------------------------------------------------

JOINT_COLOR_DEFAULT  = (200, 200, 200)   # off-white for non-selected joints
JOINT_COLOR_SELECTED = (0, 255, 255)     # cyan for the joint being annotated
BONE_COLOR           = (160, 160, 160)
BBOX_COLOR_DEFAULT   = (255, 0, 0)       # blue (BGR) for instance bbox
JOINT_BBOX_COLOR     = (0, 255, 255)     # cyan box around selected joint

JOINT_RADIUS    = 5
JOINT_BBOX_HALF = 16   # 32px box around selected joint
BONE_THICKNESS  = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Hardcoded Tron segment geometry (--use_segment_offsets)
TRON_CONTENT_W = 650
TRON_CONTENT_H = 359
TRON_OFFSET_X  = 10


# ----------------------------------------------------------------------------
# Segment offset: maps JSON frame indices to MP4 frame indices when the MP4
# is a longer video (e.g. the full Psycho movie) and the JSON describes a
# segment starting deep inside it.
# ----------------------------------------------------------------------------

def segment_start_from_filename(path):
    """
    Extract the first integer after 'segment_' in a filename and convert
    from 1-indexed (filename convention) to 0-indexed (cv2.CAP_PROP_POS_FRAMES
    convention).
        'segment_319_2006.json' (filename 1-indexed) -> 318 (0-indexed seek)
        'segment_1_1639.json'                        -> 0
    Returns 0 if no segment_<int> pattern is found.
    """
    m = re.search(r"segment_(\d+)", Path(path).name)
    if not m:
        return 0
    return max(0, int(m.group(1)) - 1)


# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------

def load_json(path):
    p = Path(path)
    p_dict = {
        "file_name":  p.name,
        "parent_dir": p.parent.name,
        "full_path":  p.parent,
    }
    with open(p, "r") as fh:
        return json.load(fh), p_dict


def build_frame_map(data):
    """
    {frame_id_0indexed: [instance_dict, ...]}.
    JSON frame_id is 1-indexed; subtract 1 to match cap frame index.
    """
    fmap = {}
    for entry in data.get("instance_info", []):
        fid = int(entry["frame_id"]) - 1
        fmap[fid] = entry.get("instances", [])
    return fmap


# ----------------------------------------------------------------------------
# Letterbox detection - ported verbatim from visualiser_no_offsets.py
# ----------------------------------------------------------------------------

def detect_content_region(seg_path):
    cap = cv2.VideoCapture(seg_path)
    ret, frame = cap.read()
    seg_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    seg_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read first frame of segment: {seg_path}")

    top = 0
    for i in range(seg_h):
        if not np.all(frame[i] > 200):
            top = i
            break

    bottom = seg_h - 1
    for i in range(seg_h - 1, -1, -1):
        if not np.all(frame[i] > 200):
            bottom = i
            break

    left = 0
    for j in range(seg_w):
        if not np.all(frame[:, j] > 200):
            left = j
            break

    right = seg_w - 1
    for j in range(seg_w - 1, -1, -1):
        if not np.all(frame[:, j] > 200):
            right = j
            break

    content_w = right - left + 1
    content_h = bottom - top + 1

    print(f"[SEGMENT] {seg_w}x{seg_h} | "
          f"bars: top={top} bottom={seg_h-1-bottom} left={left} right={seg_w-1-right} | "
          f"content: {content_w}x{content_h}")

    return content_w, content_h, left, top


def make_transform(content_w, content_h, full_w, full_h,
                   content_left=0, content_top=0,
                   offset_x=0, offset_y=0):
    scale_x = full_w / content_w
    scale_y = full_h / content_h
    t = {
        "scale_x":      scale_x,
        "scale_y":      scale_y,
        "content_left": content_left,
        "content_top":  content_top,
        "offset_x":     offset_x,
        "offset_y":     offset_y,
    }
    print(f"[TRANSFORM] scale_x={scale_x:.4f} scale_y={scale_y:.4f} "
          f"content_left={content_left} content_top={content_top} "
          f"offset_x={offset_x} offset_y={offset_y}")
    return t


def apply_transform(x, y, t):
    x_out = int((x - t["content_left"]) * t["scale_x"] + t["offset_x"])
    y_out = int((y - t["content_top"])  * t["scale_y"] + t["offset_y"])
    return x_out, y_out


# ----------------------------------------------------------------------------
# Per-instance color (instance bbox only; joints use neutral colors)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Build H36M skeleton from a 133-kp instance (for drawing only)
# ----------------------------------------------------------------------------

def instance_h36m_points(instance):
    """
    Returns a list of (x, y, valid) of length 17, derived from the instance's
    133-kp COCO data. valid=False entries should be skipped when drawing.
    """
    kps    = instance.get("keypoints", [])
    scores = instance.get("keypoint_scores", [])
    out = []
    for h36m_id in range(17):
        x, y, _ = h36m_xy_and_score_from_coco(kps, scores, h36m_id)
        if x is None or y is None:
            out.append((0.0, 0.0, False))
        else:
            out.append((x, y, True))
    return out


# ----------------------------------------------------------------------------
# Draw one instance: skeleton + bbox + selected-joint highlight
# ----------------------------------------------------------------------------

def draw_instance(canvas, instance, transform, instance_idx, selected_h36m_id):
    h, w = canvas.shape[:2]
    h36m_pts = instance_h36m_points(instance)

    # bbox
    bbox = instance.get("bbox", None)
    if bbox and len(bbox) >= 4:
        x1b, y1b, x2b, y2b = map(float, bbox[:4])
        tx1, ty1 = apply_transform(x1b, y1b, transform)
        tx2, ty2 = apply_transform(x2b, y2b, transform)
        tx1 = max(0, min(tx1, w - 1)); ty1 = max(0, min(ty1, h - 1))
        tx2 = max(0, min(tx2, w - 1)); ty2 = max(0, min(ty2, h - 1))
        inst_col = color_for_inst(instance_idx)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), inst_col, 2)
        track_id = instance.get("track_id", None)
        label = (f"Instance: {instance_idx}"
                 if track_id is None
                 else f"Instance: {instance_idx}  Track: {track_id}")
        cv2.putText(canvas, label, (tx1, max(ty1 - 5, 12)),
                    FONT, 0.55, inst_col, 1, cv2.LINE_AA)

    # bones
    for (pa, ch) in BONE_PAIRS:
        if not (h36m_pts[pa][2] and h36m_pts[ch][2]):
            continue
        pt1 = apply_transform(h36m_pts[pa][0], h36m_pts[pa][1], transform)
        pt2 = apply_transform(h36m_pts[ch][0], h36m_pts[ch][1], transform)
        cv2.line(canvas, pt1, pt2, BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # joints (neutral color, except the selected one)
    for h36m_id in range(17):
        x, y, ok = h36m_pts[h36m_id]
        if not ok:
            continue
        pt = apply_transform(x, y, transform)
        is_selected = (h36m_id == selected_h36m_id)
        col = JOINT_COLOR_SELECTED if is_selected else JOINT_COLOR_DEFAULT
        r   = JOINT_RADIUS + 2 if is_selected else JOINT_RADIUS
        cv2.circle(canvas, pt, r, col,        -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, (0, 0, 0),   1, cv2.LINE_AA)

        # 32px bbox around the selected joint to anchor the annotator's eye
        if is_selected:
            tlx = pt[0] - JOINT_BBOX_HALF; tly = pt[1] - JOINT_BBOX_HALF
            brx = pt[0] + JOINT_BBOX_HALF; bry = pt[1] + JOINT_BBOX_HALF
            cv2.rectangle(canvas, (tlx, tly), (brx, bry),
                          JOINT_BBOX_COLOR, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# HUD
# ----------------------------------------------------------------------------

def draw_hud(canvas, frame_id, n_instances, joint_name):
    cv2.putText(canvas,
                f"Frame: {frame_id}  Instances: {n_instances}  Joint: {joint_name}",
                (10, 30), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                "s=next  a=prev  d=+10  q=quit",
                (10, 55), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# CSV creation - one row per (frame, instance) for the selected joint
# ----------------------------------------------------------------------------

def build_annotation_dataframe(data, joint_name):
    """
    Walk every frame x instance in the raw 133-kp JSON and produce one row
    per (frame_id, instance_id) for the selected H36M joint. Coordinates and
    mmpose_confidence are computed from the appropriate COCO source(s).

    The CSV stores H36M joint_id and joint_name (not COCO), so it is directly
    compatible with the existing classifier feature pipeline.
    """
    if joint_name not in H36M_NAME_TO_ID:
        raise ValueError(
            f"Joint '{joint_name}' is not a valid H36M name. "
            f"Valid names: {sorted(H36M_NAME_TO_ID.keys())}"
        )
    h36m_id = H36M_NAME_TO_ID[joint_name]

    rows = []
    seen_keys = set()

    for frame in data.get("instance_info", []):
        frame_id = int(frame["frame_id"]) - 1
        for instance_ind, instance in enumerate(frame.get("instances", [])):
            track_id    = instance.get("track_id", None)
            kps_coco    = instance.get("keypoints", [])
            scores_coco = instance.get("keypoint_scores", [])

            key = (frame_id, instance_ind, track_id)
            if key in seen_keys:
                print(f"DUPLICATE: frame_id={frame_id} "
                      f"instance_id={instance_ind} track_id={track_id}")
            seen_keys.add(key)

            x, y, conf = h36m_xy_and_score_from_coco(
                kps_coco, scores_coco, h36m_id)
            if x is None:
                # Source COCO joint not present; skip but log.
                print(f"  Skipping frame={frame_id} inst={instance_ind} "
                      f"joint={joint_name}: required COCO source missing.")
                continue

            rows.append({
                "frame_id":              frame_id,
                "instance_id":           instance_ind,
                "track_id":              track_id,
                "joint_id":              h36m_id,
                "joint_name":            joint_name,
                "x":                     x,
                "y":                     y,
                "mmpose_confidence":     conf,
                "reliability_category":  None,
                "annotator_confidence":  None,
                "reason_for_distrust":   None,
                "dist_to_boundary":      None,
                "valid":                 None,
            })

    df = pd.DataFrame(rows)
    return df


def csv_filename_from_paths(json_dict, joint_name):
    parent = json_dict.get("parent_dir", "annotation")
    stem   = Path(json_dict.get("file_name", "annotation.json")).stem
    return f"{parent}_{stem}_{joint_name}.csv"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Annotation tool: skeleton overlay + per-joint CSV from "
                    "raw 133-kp MMPose JSON.")
    ap.add_argument("--json",        required=True,
                    help="Raw 133-keypoint MMPose JSON.")
    ap.add_argument("--mp4",         required=True,
                    help="Display video (frames are read from here).")
    ap.add_argument("--joint",       required=True,
                    help="H36M joint name to annotate (e.g. head, neck_base, "
                         "right_shoulder, left_knee, ...).")
    ap.add_argument("--segment_mp4", default=None,
                    help="Letterboxed segment mp4 used ONLY for letterbox "
                         "detection. If omitted, --mp4 is used.")
    ap.add_argument("--start",          type=int, default=0)
    ap.add_argument("--end",            type=int, default=None)
    ap.add_argument("--create_new_df",  type=int, default=1,
                    help="1 to create CSV (overwrites if exists), 0 to skip.")
    ap.add_argument("--output_path",    default=None,
                    help="Optional video output path; if set, runs through all "
                         "frames non-interactively and writes annotated video.")
    ap.add_argument("--use_segment_offsets", action="store_true",
                    help="Skip letterbox auto-detection; use hardcoded Tron "
                         f"geometry (content_w={TRON_CONTENT_W}, "
                         f"content_h={TRON_CONTENT_H}, "
                         f"offset_x={TRON_OFFSET_X}).")
    ap.add_argument("--mp4_frame_offset", type=int, default=None,
                    help="Frame offset added to the JSON frame index before "
                         "seeking into the MP4. Use when the MP4 is a longer "
                         "video and the JSON describes a segment that starts "
                         "deep inside it (e.g. the full Psycho movie + a "
                         "segment_319_2006 JSON). If omitted, auto-derived "
                         "from the JSON filename via 'segment_<int>'. Pass 0 "
                         "explicitly to disable. The CSV always stores "
                         "JSON-relative frame indices regardless of this flag.")
    args = ap.parse_args()

    # ---- validate joint name early ------------------------------------------
    if args.joint not in H36M_NAME_TO_ID:
        sys.exit(f"ERROR: unknown joint '{args.joint}'. "
                 f"Valid: {sorted(H36M_NAME_TO_ID.keys())}")
    selected_h36m_id = H36M_NAME_TO_ID[args.joint]

    # ---- load JSON ----------------------------------------------------------
    print(f"Loading JSON: {args.json}")
    json_data, json_dict = load_json(args.json)
    frame_map = build_frame_map(json_data)
    print(f"  Loaded {len(frame_map)} frames from JSON.")

    # ---- resolve MP4 frame offset (display-only; does not affect CSV) -------
    if args.mp4_frame_offset is not None:
        mp4_frame_offset = args.mp4_frame_offset
        offset_source = "manual --mp4_frame_offset"
    else:
        mp4_frame_offset = segment_start_from_filename(args.json)
        offset_source = f"auto-derived from '{Path(args.json).name}'"
    print(f"[OFFSET] MP4 frame offset: {mp4_frame_offset} ({offset_source}). "
          f"JSON frame 0 -> MP4 frame {mp4_frame_offset}. "
          f"CSV stores JSON-relative indices.")

    # ---- create CSV ---------------------------------------------------------
    if args.create_new_df == 1:
        csv_path = csv_filename_from_paths(json_dict, args.joint)
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"Removed existing {csv_path}")
        df = build_annotation_dataframe(json_data, args.joint)
        df.to_csv(csv_path, index=False)
        print(f"Created {csv_path} with {len(df)} rows. "
              f"Columns: {list(df.columns)}")

    # ---- open display video -------------------------------------------------
    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {args.mp4}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Frame indices throughout this script are JSON-relative. The MP4 may be
    # longer than the segment (e.g. full Psycho movie + segment_319_2006).
    # Bound start/end by both the JSON frame count and the MP4's available
    # length minus the offset, so we never seek past the end of the file.
    json_max              = max(frame_map) if frame_map else 0
    mp4_max_json_relative = (total_frames - 1) - mp4_frame_offset
    upper                 = min(json_max, mp4_max_json_relative)

    start_frame = max(0, args.start)
    end_frame   = min(upper,
                      args.end if args.end is not None else upper)
    if end_frame < start_frame:
        sys.exit(f"ERROR: end_frame ({end_frame}) < start_frame ({start_frame}). "
                 f"Check that mp4_frame_offset ({mp4_frame_offset}) is correct "
                 f"and that the MP4 has enough frames after that offset.")

    print(f"Video: {vid_w}x{vid_h} @ {fps:.2f} fps | "
          f"displaying JSON frames {start_frame}-{end_frame} "
          f"(MP4 frames {start_frame + mp4_frame_offset}-"
          f"{end_frame + mp4_frame_offset}) of {total_frames} total MP4 frames")

    # ---- coordinate transform -----------------------------------------------
    if args.use_segment_offsets:
        print("[TRANSFORM] Using hardcoded Tron segment offsets")
        t = make_transform(TRON_CONTENT_W, TRON_CONTENT_H, vid_w, vid_h,
                           0, 0, TRON_OFFSET_X, 0)
    else:
        seg_path = args.segment_mp4 if args.segment_mp4 is not None else args.mp4
        auto_w, auto_h, content_left, content_top = detect_content_region(seg_path)
        t = make_transform(auto_w, auto_h, vid_w, vid_h,
                           content_left, content_top, 0, 0)

    # ---- optional video writer ----------------------------------------------
    writer = None
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_path,
                                 fourcc, fps, (vid_w, vid_h))
        print(f"Writing output to: {args.output_path}")

    # ---- frame cache for random-access nav ----------------------------------
    cap = cv2.VideoCapture(args.mp4)
    frame_cache = {}

    def get_frame(idx):
        # idx is the JSON-relative frame index. The MP4 may be a longer video
        # (e.g. full Psycho movie) in which our segment starts at
        # mp4_frame_offset. Apply the offset only at seek time. The cache key
        # stays JSON-relative so it remains aligned with frame_map and the
        # CSV indices.
        if idx in frame_cache:
            return frame_cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx + mp4_frame_offset)
        ret, frm = cap.read()
        if not ret:
            return None
        frame_cache[idx] = frm
        return frm

    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("overlay", vid_w, vid_h)

    current = start_frame
    while True:
        if current > end_frame:
            break

        frame = get_frame(current)
        if frame is None:
            print(f"Warning: could not read frame {current}, stopping.")
            break

        instances = frame_map.get(current, [])
        canvas = frame.copy()
        for idx, inst in enumerate(instances):
            draw_instance(canvas, inst, t, idx, selected_h36m_id)
        draw_hud(canvas, current, len(instances), args.joint)

        if writer is not None:
            writer.write(canvas)

        cv2.imshow("overlay", canvas)
        key = cv2.waitKeyEx(0 if writer is None else 1) & 0xFF

        if writer is not None:
            current += 1
            continue

        if key == ord("q"):
            break
        elif key == ord("s"):
            current = min(current + 1, end_frame)
        elif key == ord("a"):
            current = max(current - 1, start_frame)
        elif key == ord("d"):
            current = min(current + 10, end_frame)
        # any other key: redraw

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {args.output_path}")
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()