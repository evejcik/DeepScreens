import json, cv2, numpy as np

with open('Outputs/my_17_json.json', 'r') as f:
    data = json.load(f)

video_info = data['video_info'][0]
width, height = video_info['video_shape']   # [width, height]
fps = float(video_info['frame_rate'])
n_frames = len(data['instance_info'])

out = cv2.VideoWriter(
    'dummy_black.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height))

black = np.zeros((height, width, 3), dtype=np.uint8)
for _ in range(n_frames):
    out.write(black)
out.release()
print(f"Written: {n_frames} frames, {width}x{height} @ {fps}fps")