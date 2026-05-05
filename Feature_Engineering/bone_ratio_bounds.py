import pandas as pd

# Mirror of which joint becomes the child for which bone in your BODY_BONES list.
# IMPORTANT: when a joint is a child of multiple bones, only the LAST bone in
# BODY_BONES wins (dict overwrite). This map reflects the winning bone for
# each child joint in your current BODY_BONES.
JOINT_TO_BONE = {
    'left_knee':       'left_thigh',
    'right_knee':      'right_thigh',
    'left_ankle':      'left_shin',
    'right_ankle':     'right_shin',
    'right_hip':       'right_torso',     # was pelvis_width before extension
    'left_hip':        'left_torso',      # new with extension
    'left_elbow':      'left_upper_arm',
    'right_elbow':     'right_upper_arm',
    'left_wrist':      'left_forearm',
    'right_wrist':     'right_forearm',
    'right_shoulder':  'shoulder_width',
    'left_ear':        'nose_to_left_ear',
    'right_ear':       'nose_to_right_ear',
    # NOT IN ANNOTATED DATA but listed for completeness:
    # left_shoulder is parent-only, has no row-level bone_ratio of its own.
}

df = pd.read_csv("Long Data.csv", low_memory=False)
df['bone_ratio'] = pd.to_numeric(df['bone_ratio'], errors='coerce')
trust = df[(df['reliability_category_int'] == 0) & (df['bone_ratio'] != -1)]

print(f"Trust rows with real bone_ratio: {len(trust)}\n")
print(f"{'joint_name':18s} {'bone_name':22s} {'n':>6s} {'med':>6s} {'p05':>6s} {'p95':>6s}")
print("-" * 70)

bounds = {}
for joint in sorted(trust['joint_name'].unique()):
    rows = trust[trust['joint_name'] == joint]
    bone = JOINT_TO_BONE.get(joint, '(no bone mapping)')
    if len(rows) < 30:
        print(f"{joint:18s} {bone:22s} {len(rows):>6d} TOO FEW")
        continue
    p05 = rows['bone_ratio'].quantile(0.05)
    p95 = rows['bone_ratio'].quantile(0.95)
    med = rows['bone_ratio'].median()
    print(f"{joint:18s} {bone:22s} {len(rows):>6d} {med:>6.2f} {p05:>6.2f} {p95:>6.2f}")
    if bone != '(no bone mapping)':
        bounds[bone] = (round(float(p05), 2), round(float(p95), 2))

# Print as a copy-paste-ready dict
print("\nCopy this into BONE_RATIO_BOUNDS in geometric_plausibility.py:")
print("BONE_RATIO_BOUNDS = {")
for bone, (lo, hi) in sorted(bounds.items()):
    print(f"    '{bone:22s}: ({lo:.2f}, {hi:.2f}),")
print("    'pelvis_width':           (1.0, 1.0),  # reference (delete this entry if pelvis_width removed from BODY_BONES)")
print("}")