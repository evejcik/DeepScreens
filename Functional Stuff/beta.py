#laying down groundwork for simple beta distribution
#we are looking at: p(joint is visible | visibility label, confidence score, any other features from data)
#confidences are bounded 0-1

#we can interpret the beta distribution's 2 parameters cleanly: the number of successes vs the number of failures -> the number of times the joint is visible vs not

from scipy.stats import beta

def beta_per_joint(df, joint):
    #input: dataset and 1 joint name
    #output: dictionary with fitted parameters

    joint_data = df[df['joint_name'] == joint].copy()
    joint_data['is_visible'] = (joint_data['visibility'] == 1).astype(int)

    results = {}

    #need to set loc and scale for scipy stats to bound problem
    #scipy beta uses MLE to calculate a and b for beta distribution
    floc = 0
    fscale = 1 #this fits the distribution to standard [0,1]
    #this works since we are working with probabilities (the confidences) bounded by 0 and 1, so we don't have to bring in unbounded MSE calculations for now
    #so now only alpha and beta are estimated
    scores = 
    alpha, beta, loc, scale = beta.fit(df, floc = floc, fscale = fscale)
