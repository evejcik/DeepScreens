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

    df["occlusion_reason"] = df["occlusion_reason"].str.split(' ')