import pandas as pd
import numpy as np
import argparse
from scipy.special import betaln   # log of the beta function B(α,β)
from scipy.optimize import minimize
from scipy.stats import beta


def moments(scores): #quick, closed‑form way to pick α and β just from the sample mean and sample variance of the scores.
    #uses: as an initial guess for a more refined optimizer (MLE or MAP),
    # as a fallback when the optimizer fails, and
    # as a simple sanity check (does the fitted distribution even look plausible?).

    sample_mean = np.mean(scores) #first moment
    sample_var = np.var(scores, ddof = 1) #second spread out moment, ddof = 1 means use unbiased variance
    k = sample_mean * (1 - sample_mean) / sample_var

    alpha_mom = sample_mean * k #mom = Method of Moments
    beta_mom = (1 - sample_mean) * k
    return sample_mean, sample_var, k, alpha_mom, beta_mom



##negative log posterior J(alpha, beta)
def neg_log_posterior(alpha, beta, scores, eps: float = 1e-6, alpha_prior = 2.0, beta_prior = 2.0): 
    #MLE out of the question, confidence scores are all wayyyyy too close to 1 so the optimizer is really struggling to find a good alpha & beta
    #so we choose MAP with weak Beta (2,2) prior
    ##FOR OPTIMIZATION'S SAKE: consider placing a different distribution over alpha and beta parameters 
    # i.e., a gamma or hierarchical bayes distrbution
    #as opposed to the initial exponential distribution

    c = jitter(scores, eps)
    N = c.size

    #(alpha - 1)
    alpha_1 = (alpha - 1.0)
    #sigma log(c_i) 
    sigma_log_ci = np.sum(np.log(c))
    #beta -1
    beta_1 = (beta - 1.0)
    sigma_log_1_ci = np.sum(np.log(1.0 - c))
    
    n_log_beta = N * betaln(alpha, beta)

    negative_log_likelihood = alpha_1 * sigma_log_ci + beta_1 * sigma_log_1_ci - n_log_beta

    ##we get this part by taking the log of the exponential distribution on alpha and beta -> if we change this, we change it here
    prior_penalty = (alpha - alpha_prior) + (beta - beta_prior) #right now, set to 2 and 2 to keep prior weak 

    final = negative_log_likelihood + prior_penalty

    return float(final)

## Logit Gaussian

## Jitter and clip -> alternatives:  Hierarchical / Empirical‑Bayes pooling & Isotonic regression (non‑parametric)

def jitter(scores, epsilon = float(1e-6)): #data now has non zero variance. A zero variance, for a beta distribution, means that when the optimizer
    #is trying to fit the best alpha and beta to the distribution, beta goes to 0 and alpha goes to infinity -> not good for trying to find a reasonal 
    #distribution of probabilities for the confidence scores
    scores = np.clip(scores, epsilon, 1.0 - epsilon)
    jittered_scores = scores + epsilon * np.random.default_rng().normal(size = scores.shape)

    clipped_jittered_scores = np.clip(jittered_scores, epsilon, 1.0 - epsilon)

    return clipped_jittered_scores


def beta_fit(scores, alpha_prior=2.0, beta_prior=2.0):
    mu, var, k, alpha0, beta0 = moments(scores)

    # Guard against invalid MoM start values -> Add better error handling and bounds checking
    if np.isnan(alpha0) or np.isnan(beta0) or np.isinf(alpha0) or np.isinf(beta0):
        print(f"Warning: MoM returned invalid values (α={alpha0}, β={beta0})")
        return float(alpha_prior), float(beta_prior), False  # ← fallback to prior
    
    x0 = np.array([alpha0, beta0])
    bounds = [(1e-3, None), (1e-3, None)]  # ← add upper bounds!


    res = minimize( #minimizes the negative log posterior which is calculated with the alpha and beta we are feeding it in x0
        fun = lambda pars: neg_log_posterior(pars[0], pars[1], scores,
        alpha_prior = alpha_prior, beta_prior=beta_prior),
        x0 = x0,
        method = 'L-BFGS-B',
        bounds = bounds,
        options = {'ftol': 1e-9, 'maxiter': 2000}
      )

    if res.success:
        return float(res.x[0]), float(res.x[1]), True
    else:
        # optimiser failed → fall back to MoM
        return float(alpha0), float(beta0), False


def main(data):
    df = pd.read_csv(data)
    results = {}
    joints = df["joint_name"].unique()
    for joint in joints:
        results[joint] = {}
        
        #fit a distribution for every visibility category:
        # unique_vis = df["visibility_category"].unique()

        #right now, just looking at distributions of visible vs not visible
        # for vis_label, vis_value in [("visible", 1), ("not_visible", [2,3,4])]:
        #     mask = (df["joint_name"] == joint) & (df["visibility_category"].isin(vis_value))
        for vis_label, vis_value in [
                ("visible",     [1]),          # <-- wrap 1 in a list
                ("not_visible", [2, 3, 4])]:   # already a list
            mask = (df["joint_name"] == joint) & df["visibility_category"].isin(vis_value)

            scores = df.loc[mask, "mmpose_confidence"].values
            
            if scores.size == 0:
                # No data for this joint/visibility – we store NaNs so downstream code can detect it.
                results[joint][vis_label] = {"a": np.nan, "b": np.nan,
                                             "n": 0, "success": False}
                continue

            
            alpha_opt, beta_opt, ok = beta_fit(scores)

            results[joint][vis_label] = {
                "a": alpha_opt,
                "b": beta_opt,
                "n": int(scores.size),
                "success": ok
            }

    print("\n=== Beta MAP fit summary ===")
    for joint, d in results.items():
        vis = d["visible"]
        not_vis = d["not_visible"]
        print(f"{joint:>12} | "
              f"vis n={vis['n']:4d}  α={vis['a']:.2f} β={vis['b']:.2f} "
              f"| not_vis n={not_vis['n']:4d}  α={not_vis['a']:.2f} β={not_vis['b']:.2f}")

    for c in [0.95, 0.98, 0.99, 0.999]:
        print(f"c={c:.3f} → trust={posterior(c):.3f}") #posterior confidence -> p(visible | confidence)
    
def posterior(c): #takes parameters a and b for a joint and calculates the posterior p(visible | confidence). just looking at a few of the returned parameters from the table, we can 
    #decide on how confident we are. 
    #c = confidence score
    #p is empirical prior of visibility for that joint
    #beta.pdf(c, a, b) = Beta(c|a,b) from scipy.stats


    #just taking parameters from table by hand for now
    # parameters from the table:
    a_vis, b_vis   = 49.16, 0.21   # visible
    a_not_vis, b_not_vis   = 90.12, 0.22   # not visible
    vis_n, not_vis_n   = 527, 2121
    p = vis_n / (vis_n + not_vis_n)
    beta_vis_pdf = beta.pdf(c,a_vis,b_vis)
    beta_not_vis_pdf = beta.pdf(c, a_not_vis, b_not_vis)
    post = (p * beta_vis_pdf) / ((p * beta_vis_pdf) + ((1-p) * beta_not_vis_pdf))

    return post

    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required = True)
    # ap.add_argument("--alpha", default = ,type = float)
    # ap.add_argument("--beta", default = ,type = float)

    args = ap.parse_args()
    main(args.data)