import pandas as pd
import numpy as np

import argparse


def main(data):

    df = pd.read_csv(data)

    print("="*80)
    print("ANNOTATION SCHEME DIAGNOSTICS")
    print("="*80)

    # 1. Check visibility_category distribution
    print("\n1. VISIBILITY CATEGORY DISTRIBUTION")
    print(df['visibility_category'].value_counts().sort_index())

    # 2. Check occlusion_reason for VISIBLE joints
    print("\n2. VISIBLE JOINTS (visibility_category == 1): Should have NO occlusion reason")
    visible_df = df[df['visibility_category'] == 1.0]
    print(f"Total visible joints: {len(visible_df)}")
    print(f"Visible joints with occlusion_reason: {visible_df['occlusion_reason'].notna().sum()}")
    if visible_df['occlusion_reason'].notna().sum() > 0:
        print("Sample visible joints with occlusion_reason:")
        print(visible_df[visible_df['occlusion_reason'].notna()][['joint_name', 'visibility_category', 'occlusion_reason', 'mmpose_confidence']].head(10))

    # 3. Check occlusion_reason for OCCLUDED joints
    print("\n3. OCCLUDED JOINTS (visibility_category == 2): Should have occlusion_reason")
    occluded_df = df[df['visibility_category'] == 2.0]
    print(f"Total occluded joints: {len(occluded_df)}")
    print(f"Occluded joints with occlusion_reason: {occluded_df['occlusion_reason'].notna().sum()}")
    print(f"Occluded joints WITHOUT occlusion_reason: {occluded_df['occlusion_reason'].isna().sum()}")

    # 4. Check occlusion_reason for OFF-SCREEN joints
    print("\n4. OFF-SCREEN JOINTS (visibility_category == 3): Should have NO occlusion reason")
    offscreen_df = df[df['visibility_category'] == 3.0]
    print(f"Total off-screen joints: {len(offscreen_df)}")
    print(f"Off-screen joints with occlusion_reason: {offscreen_df['occlusion_reason'].notna().sum()}")

    # 5. Check occlusion_reason for AMBIGUOUS joints
    print("\n5. AMBIGUOUS JOINTS (visibility_category == 4): Mixed")
    ambiguous_df = df[df['visibility_category'] == 4.0]
    print(f"Total ambiguous joints: {len(ambiguous_df)}")
    print(f"Ambiguous joints with occlusion_reason: {ambiguous_df['occlusion_reason'].notna().sum()}")

    # 6. Check occlusion_reason for HALLUCINATED joints
    print("\n6. HALLUCINATED JOINTS (visibility_category == 5): Mixed")
    hallucinated_df = df[df['visibility_category'] == 5.0]
    print(f"Total hallucinated joints: {len(hallucinated_df)}")
    print(f"Hallucinated joints with occlusion_reason: {hallucinated_df['occlusion_reason'].notna().sum()}")
    if len(hallucinated_df) > 0:
        print("Hallucinated samples:")
        print(hallucinated_df[['joint_name', 'visibility_category', 'occlusion_reason', 'mmpose_confidence', 'annotator_confidence']].head(10))

    # 7. Count unique occlusion_reason values
    print("\n7. OCCLUSION REASON VALUE COUNTS")
    print(f"Total unique occlusion_reason values: {df['occlusion_reason'].nunique()}")
    print("\nTop 15 occlusion_reason values:")
    print(df['occlusion_reason'].value_counts().head(15))

    # 8. Multi-label analysis
    print("\n8. MULTI-LABEL OCCLUSION REASONS")
    print("Checking for comma-separated values...")
    multi_label_count = df['occlusion_reason'].str.contains(',', na=False).sum()
    print(f"Rows with multi-label occlusion_reason: {multi_label_count}")
    if multi_label_count > 0:
        print("\nExamples of multi-label reasons:")
        multi_label_examples = df[df['occlusion_reason'].str.contains(',', na=False)]['occlusion_reason'].unique()
        for example in multi_label_examples[:5]:
            print(f"  - {example}")