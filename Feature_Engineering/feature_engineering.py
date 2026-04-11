#input: annotated csv 
#output: 



import pandaas as pd
import numpy as np
import argparse

#1. encode reliability category as integer: trust = 0, partial trust = 1, don't trust = 2
#2. encode joint_name as integer (0 - 16), using H36M mapping from mmpose
#3. handle missing values for 'valid' -> 0 if 0 else 1
#4. sort films by (film, track_id, joint_name) -> make sure to not mix joints or tracks when computing temporal windows!
#5. group by: (film, track_id, joint_name), sort by frame_id:
    #a. confidence_mean_wk = rolling mean of confidence_scores over a +-k frame window
    #b confidence_std
    #c. position_velocity = Euclidean distance between (x,y) at frame t and t-1
    #d. position_acceleration: change in velocity from t-1 to t
    #e. position_std_wk = rolling std of (x,y) over +-k frames
    #f. position_variance_wk 
    #g. frames_since_trust


def reliability_int(df):
    trust_map = {
        "trust" : 0,
        "don't trust" : 2,
        "partial trust" : 1
    }

    df['reliability_category_int'] = df['reliability_category'].map(trust_map)

    mask = df['reliability_category_int'].isna()
    errors_df = df.loc[mask, ['reliability_category', 'reliability_category_int']]

    print(f"Count distribution: {df['reliability_category_int'].nunique()}")
    print(df.loc[df['reliability_category_int']].isna(), 'reliability_category')

def data_loader(csv_path):
    df = pd.read(csv_path)
    df = reliability_int(df)




def main(csv):
    df = data_loader(csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(csv)

    args = ap.parse_args()
    main(
        args.csv
    )
