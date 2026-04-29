"""
visualise_skeleton.py
─────────────────────
Skeleton overlay of cleaned 17-keypoint JSON on film footage.
Letterbox detection and coordinate transform ported directly from
visualiser_no_offsets.py.

Usage
-----
python visualise_skeleton.py \
    --json          path/to/cleaned.json \
    --mp4           path/to/display_video.mp4 \
    [--segment_mp4  path/to/letterboxed_segment.mp4] \
    [--start 0] [--end 999] \
    [--output_path  out.mp4] \
    [--trust_threshold 0.7] \
    [--use_segment_offsets]   # hardcoded Tron geometry

--mp4            The video frames are read from here and drawn on.
--segment_mp4    Used ONLY for letterbox detection (first-frame scan).
                 If omitted, --mp4 is used for detection too.
--use_segment_offsets
                 Skip auto-detection; apply hardcoded Tron geometry:
                 content_w=650, content_h=359, offset_x=10.

Keyboard controls:
    s  : next frame
    a  : previous frame
    d  : skip forward 10 frames
    q  : quit

Joint colouring
    green  : keypoint_interpolated=False AND score >= trust_threshold
    yellow : keypoint_interpolated=False AND score <  trust_threshold
    red    : keypoint_interpolated=True  (regardless of score)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# H36M 17-joint skeleton
# ─────────────────────────────────────────────────────────────────────────────
BONE_PAIRS = [
    (1, 2), (2, 3),      # right leg
    (4, 5), (5, 6),      # left leg
    (7, 8),              # spine -> thorax
    (8, 10),             # thorax -> head
    (11, 12), (12, 13),  # left arm
    (14, 15), (15, 16),  # right arm
    (0, 1), (0, 4),      # root -> hips
    (8, 11), (8, 14),    # thorax -> shoulders
    (8, 9),              # thorax -> neck_base
]

GREEN  = (50, 220, 50)
YELLOW = (30, 220, 220)
RED    = (50, 50, 230)

BONE_COLOUR    = (200, 200, 200)
JOINT_RADIUS   = 5
BONE_THICKNESS = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# Hardcoded Tron segment geometry (--use_segment_offsets)
TRON_CONTENT_W = 650
TRON_CONTENT_H = 359
TRON_OFFSET_X  = 10


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def build_frame_map(data: dict) -> dict:
    """
    Returns {frame_id_0indexed: [instance_dict, ...]}
    JSON frame_id is 1-indexed; we subtract 1 here to match cap frame index,
    consistent with visualiser_no_offsets.py.
    """
    fmap = {}
    for entry in data.get("instance_info", []):
        fid = int(entry["frame_id"]) - 1
        fmap[fid] = entry.get("instances", [])
    return fmap


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox detection — ported verbatim from visualiser_no_offsets.py
# ─────────────────────────────────────────────────────────────────────────────

def detect_content_region(seg_path: str):
    """
    Read the first frame of seg_path and detect white letterbox/pillarbox bars.
    Returns (content_w, content_h, offset_x, offset_y).
    White is defined as all channels > 200.
    """
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
                   offset_x=0, offset_y=0) -> dict:
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


def apply_transform(x: float, y: float, t: dict):
    x_out = int((x - t["content_left"]) * t["scale_x"] + t["offset_x"])
    y_out = int((y - t["content_top"])  * t["scale_y"] + t["offset_y"])
    return x_out, y_out


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance colour — ported from visualiser_no_offsets.py
# ─────────────────────────────────────────────────────────────────────────────

def color_for_inst(idx: int) -> tuple:
    if idx == 0:
        return (0, 0, 255)
    elif idx == 1:
        return (0, 230, 255)
    else:
        b = min((97 * idx + 29) % 256 * 1.5, 255)
        g = min((17 * idx + 91) % 256 * 1.5, 255)
        r = min((37 * idx + 53) % 256 * 1.5, 255)
        return (int(b), int(g), int(r))


# ─────────────────────────────────────────────────────────────────────────────
# Joint colouring
# ─────────────────────────────────────────────────────────────────────────────

def joint_colour(score: float, interpolated: bool,
                 trust_threshold: float) -> tuple:
    if interpolated:
        return RED
    return GREEN if score >= trust_threshold else YELLOW


# ─────────────────────────────────────────────────────────────────────────────
# Draw one instance
# ─────────────────────────────────────────────────────────────────────────────

def draw_instance(canvas: np.ndarray, instance: dict, t: dict,
                  trust_threshold: float, instance_idx: int, trust_only: bool):
    kps    = instance.get("keypoints", [])
    scores = instance.get("keypoint_scores", [0.0] * len(kps))
    interp = instance.get("keypoint_interpolated", [False] * len(kps))
    bbox   = instance.get("bbox", None)
    tid    = instance.get("track_id", instance_idx)

    if len(kps) != 17:
        print(f"  Warning: instance {instance_idx} has {len(kps)} keypoints, expected 17. Skipping.")
        return

    inst_col = color_for_inst(instance_idx)


    # joints
    for j, (x, y) in enumerate(kps):
        score  = scores[j] if j < len(scores) else 0.0
        is_int = interp[j] if j < len(interp) else False
        col    = joint_colour(score, is_int, trust_threshold)
        pt     = apply_transform(x, y, t)
        cv2.circle(canvas, pt, JOINT_RADIUS, col,       -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, JOINT_RADIUS, (0, 0, 0),  1, cv2.LINE_AA)

    # bounding box + label
    if bbox and len(bbox) >= 4:
        x1b, y1b, x2b, y2b = map(float, bbox[:4])
        tx1, ty1 = apply_transform(x1b, y1b, t)
        tx2, ty2 = apply_transform(x2b, y2b, t)
        h, w = canvas.shape[:2]
        tx1 = max(0, min(tx1, w - 1))
        ty1 = max(0, min(ty1, h - 1))
        tx2 = max(0, min(tx2, w - 1))
        ty2 = max(0, min(ty2, h - 1))
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), inst_col, 1)
        label = (f"Instance: {instance_idx}"
                 if tid is None else f"Instance: {instance_idx} Track: {tid}")
        cv2.putText(canvas, label, (tx1, max(ty1 - 5, 0)),
                    FONT, 0.5, inst_col, 1, cv2.LINE_AA)

    # bones
    # for (pa, ch) in BONE_PAIRS:
    #     if pa >= len(kps) or ch >= len(kps):
    #         continue
    #     pt1 = apply_transform(*kps[pa], t)
    #     pt2 = apply_transform(*kps[ch], t)
    #     cv2.line(canvas, pt1, pt2, BONE_COLOUR, BONE_THICKNESS, cv2.LINE_AA)

    # joints
    for j, (x, y) in enumerate(kps):
        score  = scores[j] if j < len(scores) else 0.0
        is_int = interp[j] if j < len(interp) else False
        col    = joint_colour(score, is_int, trust_threshold)
        pt     = apply_transform(x, y, t)
        cv2.circle(canvas, pt, JOINT_RADIUS, col,       -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, JOINT_RADIUS, (0, 0, 0),  1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(canvas: np.ndarray, frame_id_display: int, n_instances: int):
    h, w = canvas.shape[:2]

    # top-left: frame + instance count
    cv2.putText(canvas,
                f"Frame: {frame_id_display}  Instances: {n_instances}",
                (10, 30), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # bottom-right: colour legend
    legend = [(GREEN, "trust"), (YELLOW, "partial"), (RED, "interpolated")]
    xb, yb = w - 170, h - 10
    for colour, text in reversed(legend):
        cv2.circle(canvas, (xb, yb - 3), 5, colour, -1)
        cv2.putText(canvas, text, (xb + 12, yb),
                    FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        yb -= 20


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Skeleton overlay of cleaned 17-kp JSON on film footage")
    ap.add_argument("--json",               required=True,
                    help="Cleaned 17-keypoint JSON")
    ap.add_argument("--mp4",                required=True,
                    help="Display video (frames are read from here)")
    ap.add_argument("--segment_mp4",        default=None,
                    help="Letterboxed segment mp4 used ONLY for letterbox "
                         "detection. If omitted, --mp4 is used.")
    ap.add_argument("--start",              type=int, default=0)
    ap.add_argument("--end",                type=int, default=None)
    ap.add_argument("--output_path",        default=None)
    ap.add_argument("--trust_threshold",    type=float, default=0.7)
    ap.add_argument("--use_segment_offsets", action="store_true",
                    help="Skip auto-detection; use hardcoded Tron geometry "
                         f"(content_w={TRON_CONTENT_W}, "
                         f"content_h={TRON_CONTENT_H}, "
                         f"offset_x={TRON_OFFSET_X})")
    ap.add_argument("--trust_only", action="store_true",
                help="Only render green (trusted, non-interpolated) joints")
    args = ap.parse_args()

    # ── load JSON ─────────────────────────────────────────────────────────
    print(f"Loading JSON: {args.json}")
    frame_map = build_frame_map(load_json(args.json))
    print(f"  Loaded {len(frame_map)} frames from JSON.")

    # ── open display video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {args.mp4}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    start_frame = max(0, args.start)
    end_frame   = min(total_frames - 1,
                      args.end if args.end is not None else total_frames - 1)

    print(f"Video: {vid_w}x{vid_h} @ {fps:.2f} fps | "
          f"displaying frames {start_frame}–{end_frame} of {total_frames}")

    # ── build coordinate transform ────────────────────────────────────────
    if args.use_segment_offsets:
        print(f"[TRANSFORM] Using hardcoded Tron segment offsets")
        t = make_transform(TRON_CONTENT_W, TRON_CONTENT_H, vid_w, vid_h,
                           0, 0, TRON_OFFSET_X, 0)
    else:
        seg_path = args.segment_mp4 if args.segment_mp4 is not None else args.mp4
        auto_w, auto_h, content_left, content_top = detect_content_region(seg_path)
        t = make_transform(auto_w, auto_h, vid_w, vid_h,
                           content_left, content_top, 0, 0)

    # ── optional output writer ────────────────────────────────────────────
    writer = None
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_path,
                                 fourcc, fps, (vid_w, vid_h))
        print(f"Writing output to: {args.output_path}")

    # ── frame cache for random-access navigation ──────────────────────────
    cap = cv2.VideoCapture(args.mp4)
    frame_cache: dict = {}

    def get_frame(idx: int):
        if idx in frame_cache:
            return frame_cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = cap.read()
        if not ret:
            return None
        frame_cache[idx] = frm
        return frm

    # ── window ────────────────────────────────────────────────────────────
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

        # frame_map is 0-indexed (frame_id - 1 already applied in build_frame_map)
        instances = frame_map.get(current, [])

        canvas = frame.copy()
        for idx, inst in enumerate(instances):
            draw_instance(canvas, inst, t, args.trust_threshold, idx, trust_only = args.trust_only)
        draw_hud(canvas, current, len(instances))

        if writer is not None:
            writer.write(canvas)

        cv2.imshow("overlay", canvas)
        key = cv2.waitKeyEx(0 if writer is None else 1) & 0xFF

        if writer is not None:
            current += 1
            continue

        if key == ord('q'):
            break
        elif key == ord('s'):
            current = min(current + 1, end_frame)
        elif key == ord('a'):
            current = max(current - 1, start_frame)
        elif key == ord('d'):
            current = min(current + 10, end_frame)
        # any other key: redraw same frame

    # ── cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved: {args.output_path}")
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()