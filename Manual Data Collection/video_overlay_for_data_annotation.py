import cv2
import os
from pathlib import Path
import json
import numpy as np
import argparse
import pandas as pd

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
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

    txt_y = y1 + 20
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
    # 1️⃣  Determine centre and scale
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
    # 2️⃣  Validate the bbox shape and reshape to (2,2)
    # --------------------------------------------------------------
    raw = np.asarray(json_bbox, dtype=np.float32).reshape(-1)
    if raw.size != 4:
        raise ValueError(
            f"bbox must contain exactly 4 numbers, got {raw.tolist()}")
    pts = raw.reshape(2, 2)                 # [[x1, y1], [x2, y2]]

    # --------------------------------------------------------------
    # 3️⃣  Apply the *exact* normalise → denormalise that the visualiser uses
    # --------------------------------------------------------------
    norm      = (pts - centre) / scale       # normalise each (x,y) pair
    vis_pts   = norm * scale + centre        # denormalise → visualiser space
    vis_bbox  = vis_pts.reshape(4)           # back to flat [x1, y1, x2, y2]

    return vis_bbox.tolist()


def main(mp4_path, json_path, start, end, new_df):
   
    # ap.add_argument("--output_dir", required = True)

    json_data = load_json(json_path)
    meta = json_data['meta_info'] 
    instances_map = frame_to_instances_map(json_data)
    cap = cv2.VideoCapture(mp4_path)

    video_shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    
    frame_id = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if end < 0:
        end = total - 1

    if new_df == 1:
        rows = []
        df_rows_path = "rows_df.csv"
        if os.path.exists(df_rows_path):
            os.remove(df_rows_path)
            print(f"{df_rows_path} has been deleted.")

        # new_df = 1
        while new_df <= total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) #sets what frame im at

            # if frame_id > end:
            #     break

            ok, frame = cap.read()
            if not ok:
                break

            instances = instances_map.get(frame_id, [])
                
            for instance_ind, instance in enumerate(instances):
                track_id = instance.get('track_id', None)
                rows.append((frame_id, instance_ind, track_id))
            frame_id +=1
            


            df = pd.DataFrame(rows,columns = ['frame_id', 'instance_id', 'track_id'] )
            # print(df.head())
            df.to_csv("rows_df.csv", index = False)

            new_df = 0

        # cap.release()

    
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

            # inside the loop, after draw_bbox_and_label():
            # draw_box(frame, vis_bbox, (0, 255, 255), thickness=1)   # yellow

            

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
    ap.add_argument("--new_df", type = int)
    ap.add_argument("--start", type = int)

    args = ap.parse_args()
    main(args.mp4, args.json, args.start, args.end, args.new_df)

