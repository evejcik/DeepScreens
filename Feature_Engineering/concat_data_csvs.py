import pandas as pd
import numpy as np

annotated_data = pd.read_csv("Long Data.csv")
unannotated_data = pd.read_csv("Long Long Data.csv")
annotated_data_with_probs = pd.read_csv("Long_Data_with_probs.csv")

# geom_plausible: cast to nullable boolean to match annotated float64 (which encodes True/False/NaN)
unannotated_data['geom_plausible'] = pd.to_numeric(unannotated_data['geom_plausible'], errors='coerce')

# reason_for_distrust: -1 sentinel -> NaN so it matches annotated object column
unannotated_data['reason_for_distrust'] = unannotated_data['reason_for_distrust'].replace(-1, np.nan)

# same for other sentinel-filled columns that should be NaN when unannotated
sentinel_cols = ['annotator_confidence', 'mmpose_confidence', 'reliability_category_int',
                 'confidence_std_wk', 'frac_trust_wk', 'frac_partial_wk', 'frac_dont_trust_wk']
for col in sentinel_cols:
    unannotated_data[col] = unannotated_data[col].replace(-1, np.nan)

def diagnostics(annotated, unannotated):
    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(f"Annotated:   {annotated.shape}")
    print(f"Unannotated: {unannotated.shape}")

    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)
    only_in_annotated   = set(annotated.columns)   - set(unannotated.columns)
    only_in_unannotated = set(unannotated.columns) - set(annotated.columns)
    shared              = set(annotated.columns)   & set(unannotated.columns)
    print(f"Shared columns ({len(shared)}):           {sorted(shared)}")
    print(f"Only in annotated ({len(only_in_annotated)}):    {sorted(only_in_annotated)}")
    print(f"Only in unannotated ({len(only_in_unannotated)}):  {sorted(only_in_unannotated)}")

    print("\n" + "=" * 60)
    print("DTYPES (shared columns)")
    print("=" * 60)
    dtype_mismatches = []
    for col in sorted(shared):
        at = annotated[col].dtype
        ut = unannotated[col].dtype
        if at != ut:
            dtype_mismatches.append((col, at, ut))
    if dtype_mismatches:
        print(f"{'COLUMN':<30} {'ANNOTATED':<15} {'UNANNOTATED':<15}")
        for col, at, ut in dtype_mismatches:
            print(f"{col:<30} {str(at):<15} {str(ut):<15}")
    else:
        print("All shared columns have matching dtypes.")

    print("\n" + "=" * 60)
    print("FILMS")
    print("=" * 60)
    ann_films   = set(annotated['film'].unique())
    unann_films = set(unannotated['film'].unique())
    overlap     = ann_films & unann_films
    print(f"Annotated films:             {sorted(ann_films)}")
    print(f"Unannotated films:           {sorted(unann_films)}")
    print(f"Overlap (should be the same):   {sorted(overlap)}")

    print("\n" + "=" * 60)
    print("KEYPOINTS PER FILM")
    print("=" * 60)
    print("Annotated:")
    print(annotated.groupby('film').size())
    print("Unannotated:")
    print(unannotated.groupby('film').size())

    print("\n" + "=" * 60)
    print("NULL VALUES")
    print("=" * 60)
    ann_nulls   = annotated.isnull().sum()
    unann_nulls = unannotated.isnull().sum()
    print("Annotated:")
    print(ann_nulls[ann_nulls > 0] if ann_nulls.any() else "None")
    print("Unannotated:")
    print(unann_nulls[unann_nulls > 0] if unann_nulls.any() else "None")

    print("\n" + "=" * 60)
    print("JOINT COVERAGE")
    print("=" * 60)
    ann_joints   = set(annotated['joint_name'].unique())
    unann_joints = set(unannotated['joint_name'].unique())
    missing_in_unann = ann_joints - unann_joints
    missing_in_ann   = unann_joints - ann_joints
    if missing_in_unann:
        print(f"Joints in annotated but not unannotated: {missing_in_unann}")
    if missing_in_ann:
        print(f"Joints in unannotated but not annotated: {missing_in_ann}")
    if not missing_in_unann and not missing_in_ann:
        print("Joint sets match.")

    print("\n" + "=" * 60)
    print("FILM_ID SANITY")
    print("=" * 60)
    overlap_ids = set(annotated['film_id'].unique()) & set(unannotated['film_id'].unique())
    if overlap_ids:
        print(f"WARNING: overlapping film_ids: {sorted(overlap_ids)}")
    else:
        print("No overlapping film_ids.")

diagnostics(annotated_data, unannotated_data)

concat_df = pd.concat([annotated_data, unannotated_data])
concat_df.to_csv("Concatenated_Data.csv", index = False)

concat_df = pd.concat([annotated_data, unannotated_data])
concat_df.to_csv("Concatenated_Data.csv", index=False)

reloaded = pd.read_csv("Concatenated_Data.csv")

expected_rows = annotated_data.shape[0] + unannotated_data.shape[0]
expected_cols = annotated_data.shape[1]  # should be same for both

print("=" * 60)
print("CONCAT DIAGNOSTICS")
print("=" * 60)
print(f"Annotated rows:    {annotated_data.shape[0]}")
print(f"Unannotated rows:  {unannotated_data.shape[0]}")
print(f"Expected total:    {expected_rows}")
print(f"Actual total:      {reloaded.shape[0]}")

if reloaded.shape[0] == expected_rows:
    print("ROW CHECK: PASSED")
else:
    print(f"ROW CHECK: FAILED — missing {expected_rows - reloaded.shape[0]} rows")

if reloaded.shape[1] == expected_cols:
    print("COL CHECK: PASSED")
else:
    print(f"COL CHECK: FAILED — expected {expected_cols} cols, got {reloaded.shape[1]}")

print(f"\nFinal shape: {reloaded.shape}")

null_counts = reloaded.isnull().sum()
null_counts = null_counts[null_counts > 0]
print(f"\nNULL VALUES:")
print(null_counts if len(null_counts) > 0 else "None")

print(f"\nFILMS: {sorted(reloaded['film'].unique())}")
print(f"\nROWS PER FILM:")
print(reloaded.groupby('film').size())