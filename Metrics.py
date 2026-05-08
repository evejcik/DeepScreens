import pandas as pd

#CSV Metrics (Annotated)
annotated_films = pd.read_csv('./Feature_Engineering/Long Data.csv')
all_film_data = pd.read_csv('./Feature_Engineering/Long Long Data.csv')

film_count = annotated_films['film'].nunique()
frame_count = annotated_films.groupby('film')['frame_id'].nunique().sum()
annotated_frame_pct = annotated_films.shape[1] / all_film_data.shape[0] * 100 
row_count = annotated_films.groupby('film')['frame_id'].count().sum()

joints_per_film = annotated_films.groupby('film')['joint_name']
total_joints_annotated = annotated_films['joint_name'].nunique()
joint_pcts = annotated_films.groupby('joint_name').mean()


### Json Metrics
#confidence scores mean, median, standard deviation, variation
mmpose_scores_mean = all_film_data['mmpose_confidence'].mean()
mmpose_scores_min = all_film_data['mmpose_confidence'].min()
mmpose_scores_max = all_film_data['mmpose_confidence'].max()
mmpose_scores_std = all_film_data['mmpose_confidence'].std()
mmpose_scores_var = all_film_data['mmpose_confidence'].var()


#Annotations Metrics

features = annotated_films.columns
native_features = ['frame_id', 'instance_id', 'track_id', 'joint_id', 'joint_name', 'x', 'y', 'mmpose_co']
annotated_features = [f for f in features if f not in native_features]


#should i add in bounding box length and width changes as a feature?

reliability_cnts = annotated_films.groupby('reliability_category_int').count()
reliability_cnt_per_joint = annotated_films.groupby(['reliability_category_int'])['joint'].count()
reliability_cnt_per_film = annotated_films.groupby(['reliability_category_int'])['film'].count()
unreliability = (annotated_films.groupby(['film', 'joint_name'])['reliability_category_int'].apply(lambda x: (x == 2).mean()).reset_index(name = 'dont_trust_rate'))
most_unreliable = unreliability.loc[unreliability.groupby('film')['dont_trust_rate'].idxmax()]
most_reliable = unreliability.loc[unreliability.groupby('film')['dont_trust_rate'].idxmin()]

annotator_confidence_per_film = annotated_films.groupby('film')['annotator_confidence'].mean()
occlusion_per_film = annotated_films.groupby('film')['reason_for_distrust'].count()
max_frames_since_trust_per_film = annotated_films.groupby('film')['frames_since_trust'].max()
min_frames_since_trust_per_film = annotated_films.groupby('film')['frames_since_trust'].min()

#give instances of low confidence

#correlation metrics for features vs. labels

dropped_features = ['joint_name.1', 
                        'valid instance bbox', 
                        'reliability_category', 
                        'confidence_mean_wk',
                        'x_velocity',
                        'y_velocity'
                        ]


### Classifier Metrics (log_reg_classifier.py)

#precision, recall, accuracy per film
#precision, recall, accuracy per joint
#design decisions
#which film performed the best, which film performed the worst
#failure modes -> Tron, head joint


### Training Classifier Metrics

#


### Prediction Classifier Metrics



###Interpolation Metrics

#how many interpolated at which thresholds
#how many frames were not interpolated


###Bone Constraint Metrics
#using 1.0 provided the best results
#average size of each bone


###Angle Constraint Metrics

###Sav Golay Smoothing in 3D
#what thesholds

###

### Classifier Performance Metrics (additions)
# - per-tier precision/recall (Tier 1, 2, 3)
# - confusion matrices per fold and aggregate
# - ROC-AUC and PR-AUC per fold and aggregate
# - calibration plot (predicted prob vs empirical frequency, binned)
# - feature importance via LightGBM gain
# - operating-point sensitivity: precision/recall at thresholds 0.1, 0.3, 0.5, 0.7, 0.9

### Pipeline Quality Metrics (additions)
# - mean per-frame position velocity (jitter) per stage
# - bone length coefficient of variation per stage
# - joint angle violation rate per stage
# - fraction of flagged joints actually interpolated vs left unchanged

### Cross-condition Comparison (additions)
# - per-condition (1-5) values for: position jitter, bone length CV,
#   angle violation rate, frames per condition, perceptual rating

### User Study (additions if running)
# - n_participants, n_conditions evaluated each
# - mean rating per condition with confidence intervals
# - pairwise preference matrix
# - inter-rater agreement (kappa)
# - open-ended feedback themes (qualitative section)

### Computational Cost (often overlooked)
# - inference time per frame at each pipeline stage
# - total wall-clock to process Ramona end-to-end
# - reference comparison: VideoPose3D alone vs full pipeline

