import numpy as np
import pandas as pd

def clean_occlusion_reason(df):
    #this is for when the visibility_category is 3 or 5 -> then there is no occlusion reason, so need to replace the blanks to prevent crashes

    df['occlusion_reason'] = df['occlusion_reason'].astype(str) #makes sure object is treated as string, handles NaNs

    # df.loc[~df['visibility_category'].isin([2.0, 4.0, 5.0]), 'occlusion_reason'] = "None"

    df.loc[df['visibility_category'] == 3.0, 'occlusion_reason'] = "off screen"
    df.loc[df['visibility_category'] == 1.0, 'occlusion_reason'] = 'visible'
    df.loc[df['visibility_category'] == 4.0, 'occlusion_reason'] = 'confused, too ambiguous'

    df.loc[df['occlusion_reason'].isna() | (df['occlusion_reason'] == 'nan'), 'occlusion_reason'] = "None" #just doing some double checking cleaning

    return df

def normalize_occlusion_reasons(df):
    #makes occlusion_reason order-invariant
    #Example: "self_occlusion, external_object" == "external_object, self_occlusion"

    df['occlusion_reason'] = df['occlusion_reason'].str.split(", ").apply(lambda x: ", ".join(sort(set(item.strip() for item in x))) if isinstance(x, list) else "")
    #now we have a list of the objects, we want to sort, then concat back together

    return df

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df = clean_occlusion_reason(df)
    df = normalize_occlusion_reasons(df)

    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = load_and_clean_data(sys.argv[1])
        print(f"Loaded {len(df)} rows")
        print(f"Visibility categories: {df['visibility_category'].unique()}")
        print(f"Unique occlusion reasons: {df['occlusion_reason'].nunique()}")

    else: 
        print("Usage: python data_loader.py <path_to_csv>")