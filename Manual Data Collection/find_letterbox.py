import cv2

seg = cv2.VideoCapture('/Users/emmavejcik/Desktop/DeepScreens/Manual Data Collection/Data Folders (MP4 and JSON)/emma_clip_results_2/Tron_Legacy_2010/Tron_Legacy_2010/segment_3540_3759.mp4')
full = cv2.VideoCapture('/Users/emmavejcik/Desktop/DeepScreens/Movie Recordings/From Youtube/MP4s/Tron Legacy 2010.mp4')

full.set(cv2.CAP_PROP_POS_FRAMES, 3540)

ret1, seg_frame = seg.read()
ret2, full_frame = full.read()

cv2.imwrite('seg_frame0.png', seg_frame)
cv2.imwrite('full_frame3540.png', full_frame)
seg.release()
full.release()