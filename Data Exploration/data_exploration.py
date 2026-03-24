import pandas as pd
import numpy as np

import argparse

def clean_occlusion_reason(df):
    #this is for when the visibility_category is 3 or 5 -> then there is no occlusion reason, so need to replace the blanks to prevent crashes

    df['occlusion_reason'] = df['occlusion_reason'].astype(str) #makes sure object is treated as string, handles NaNs

    # df.loc[~df['visibility_category'].isin([2.0, 4.0, 5.0]), 'occlusion_reason'] = "None"

    df.loc[df['visibility_category'] == 3.0, 'occlusion_reason'] = "off screen"
    df.loc[df['visibility_category'] == 1.0, 'occlusion_reason'] = 'visible'
    df.loc[df['visibility_category'] == 4.0, 'occlusion_reason'] = 'confused, too ambiguous'

    df.loc[df['occlusion_reason'].isna() | (df['occlusion_reason'] == 'nan'), 'occlusion_reason'] = "None" #just doing some double checking cleaning

    return df


def find_nulls_in_occlusion_reason(df):
    df = pd.read_csv(df)
    df.loc[df['occlusion_reason'].isna() | (df['occlusion_reason'] == 'nan'), 'occlusion_reason'] = "None"

    cols = ['frame_id', ' instance_id', 'joint_name', 'visibility_category', 'occlusion_reason']
    display_df = df[cols].head(20)
    print(display_df.to_string(index = False, justify = 'left', col_space = 0))

    # print(f'Frame Id: {df['frame_id']}     Visibility Category: {df['visibility_category']}           Occlusion Reason: {df['occlusion_reason']}')


def find_these_fours(df):
    
    print(f'Number of 4.0 category: {len(df[df['visibility_category'] == 4.0])}')

    cols = ['frame_id', ' instance_id', 'joint_name', 'visibility_category', 'occlusion_reason']
    df_fours = df.loc[df['visibility_category'] == 4.0]
    df_4 = df_fours[cols].head(50)

    print(df_4.to_string(index = False, justify = 'left', col_space = 0))



def main(data):

    df = pd.read_csv(data)
    df = clean_occlusion_reason(df) #cleans up visibility categories 3 and 1
    
    find_these_fours(df.copy(deep=True))
    

    print("="*80)
    print("ANNOTATION SCHEME DIAGNOSTICS")
    print("="*80)

    # 1. Check visibility_category distribution
    print("\n1. VISIBILITY CATEGORY DISTRIBUTION")
    print(df['visibility_category'].value_counts().sort_index())


    print("="*80) #Making order invariant set of occlusin reasons
    #take comma separated strings, split by comma
    df['occlusion_reason'] = (df['occlusion_reason'].str.split(',').apply(lambda x: ", ".join(sorted(set(item.strip() for item in x))) if isinstance(x, list) else ""))
    print("OCCLUSION REASON VALUE COUNTS")

    print(f"Total unique occlusion_reason values: {df['occlusion_reason'].nunique()}")
    print("\nOcclusion Reason Counts")

    print(df['occlusion_reason'].value_counts())


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required = True)
    args = ap.parse_args()

    main(args.data)
    # find_nulls_in_occlusion_reason(args.data)
