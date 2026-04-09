import cv2
import numpy as np

# seg = cv2.imread('seg_frame0.png')

# # scan from top to find first non-white row
# for i in range(seg.shape[0]):
#     if not np.all(seg[i] > 200):
#         print(f"First non-white row: {i}")
#         break

# # scan from bottom to find last non-white row
# for i in range(seg.shape[0]-1, -1, -1):
#     if not np.all(seg[i] > 200):
#         print(f"Last non-white row: {i}")
#         break

seg = cv2.VideoCapture('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Tron_Legacy_2010/Tron_Legacy_2010/segment_3540_3759.mp4')
ret, frame = seg.read()
seg.release()

print(f"frame shape: {frame.shape}")
print(f"top 20 rows mean brightness: {[frame[i].mean() for i in range(20)]}")
print(f"bottom 20 rows mean brightness: {[frame[frame.shape[0]-1-i].mean() for i in range(20)]}")

# save a crop of just the top 50 rows to inspect
cv2.imwrite('top_rows.png', frame[:50, :])
cv2.imwrite('bottom_rows.png', frame[590:, :])

seg = cv2.VideoCapture('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Tron_Legacy_2010/Tron_Legacy_2010/segment_3540_3759.mp4')
ret, frame = seg.read()
seg.release()

# find exact top and bottom content rows using mean brightness < 240
for i in range(frame.shape[0]):
    if frame[i].mean() < 240:
        print(f"First content row: {i}")
        break

for i in range(frame.shape[0]-1, -1, -1):
    if frame[i].mean() < 240:
        print(f"Last content row: {i}")
        break

print(f"Total frame height: {frame.shape[0]}")

seg = cv2.VideoCapture('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Tron_Legacy_2010/Tron_Legacy_2010/segment_3540_3759.mp4')
ret, frame = seg.read()
seg.release()

# check column means at row 300 (middle of content)
print("Column means, left 20 cols:")
for i in range(20):
    print(f"  col {i}: mean={frame[300, i].mean():.1f}")
print("Column means, right 20 cols:")
for i in range(20):
    col = frame.shape[1] - 1 - i
    print(f"  col {col}: mean={frame[300, col].mean():.1f}")


seg = cv2.VideoCapture('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Tron_Legacy_2010/Tron_Legacy_2010/segment_3540_3759.mp4')
ret, frame = seg.read()
seg.release()

# scan from right to find last content column
for i in range(frame.shape[1]-1, -1, -1):
    if frame[300, i].mean() < 240:
        print(f"Last content column: {i}")
        break

# scan from left to find first content column
for i in range(frame.shape[1]):
    if frame[300, i].mean() < 240:
        print(f"First content column: {i}")
        break