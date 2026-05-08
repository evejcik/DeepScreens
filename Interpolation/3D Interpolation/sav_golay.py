from scipy.signal import savgol_filter

# After 3D lift, before Unity:
# kpts_3d shape: (n_frames, 17, 3)
for joint_idx in range(17):
    for axis in range(3):
        kpts_3d[:, joint_idx, axis] = savgol_filter(
            kpts_3d[:, joint_idx, axis],
            window_length=7,
            polyorder=2
        )