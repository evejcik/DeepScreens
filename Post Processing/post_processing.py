#for downstream -> after the classifier's output, potentially to the 3D coordinates, before going into the animation.

def sandwiched_partial_trust_trust(df, col='reliability_category_int', k = 1):
    #if sandwiched between 0s, partial trust also becomes 0s
    df = df.sort_values(['film', 'instance_id', 'joint_name', 'frame_id'])
    
    def fix_sandwich(x):
        prev = x.shift(1)
        next_ = x.shift(-1)
        is_sandwiched_trust = (x == 1) & (prev == 0) & (next_ == 0)
        is_sandwiched_dont_trust = (x == 1) & (prev == 2) & (next_ == 2)
        x = x.where(~is_sandwiched_trust, 0)
        x = x.where(~is_sandwiched_dont_trust, 2)
    
    df['reliability_sandwiched'] = df.groupby(['film', 'instance_id', 'joint_name'])[col].transform(fix_sandwich)

    return df