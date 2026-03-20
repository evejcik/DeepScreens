import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
from pathlib import Path
import warnings

import pandas as pd
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def prepare_data(df):
    """
    Prepare data: create binary label, handle missing values, clip confidences.
    
    Returns:
        df: cleaned dataframe
    """
    # Create binary label: 1 = visible, 0 = not visible
    df['label'] = (df['visibility_category'] == 1).astype(int)
    
    # Clip confidences to avoid exact 0/1 (required for isotonic regression)
    df['mmpose_confidence'] = np.clip(df['mmpose_confidence'], 1e-6, 1 - 1e-6)
    
    # Remove rows with missing occlusion_reason
    df = df.dropna(subset=['occlusion_reason'])
    
    # For multi-label occlusion_reason (comma-separated), keep only primary reason
    # If you want to handle multi-label properly, see Alternative approach below
    if df['occlusion_reason'].dtype == 'object':
        df['occlusion_reason'] = df['occlusion_reason'].str.split(',').str[0].str.strip()
    
    print(f"Data prepared: {len(df)} rows")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"Occlusion reasons: {df['occlusion_reason'].value_counts().to_dict()}\n")
    
    return df


def plot_class_distribution(df, save_dir='./results'):
    """Plot class distribution per occlusion reason."""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count visible vs. not-visible per reason
    reason_counts = df.groupby(['occlusion_reason', 'label']).size().unstack(fill_value=0)
    reason_counts.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'])
    
    ax.set_xlabel('Occlusion Reason')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution per Occlusion Reason')
    ax.legend(['Not Visible (0)', 'Visible (1)'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_distribution.png', dpi=300)
    plt.close()
    
    print(f"Saved: class_distribution.png")


def fit_and_evaluate_per_reason(df, save_dir='./results', test_size=0.3, random_state=42):
    """
    Fit isotonic regression per occlusion reason.
    Evaluate and plot calibration curves.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    reasons = df['occlusion_reason'].unique()
    reasons = [r for r in reasons if pd.notna(r)]  # Remove NaN
    
    results = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(reasons)))
    
    for idx, reason in enumerate(sorted(reasons)):
        print(f"\n{'='*60}")
        print(f"Processing: {reason}")
        print(f"{'='*60}")
        
        # Filter data for this reason
        reason_df = df[df['occlusion_reason'] == reason].copy()
        n_samples = len(reason_df)
        n_visible = reason_df['label'].sum()
        pct_visible = 100 * n_visible / n_samples
        
        print(f"Samples: {n_samples} | Visible: {n_visible} ({pct_visible:.1f}%)")
        
        # Check if we have enough samples
        if n_samples < 20:
            print(f"⚠️  WARNING: Only {n_samples} samples. Skipping.")
            continue
        
        # Check if we have both classes
        if reason_df['label'].nunique() < 2:
            print(f"⚠️  WARNING: Only one class present. Skipping.")
            continue
        
        # Train/test split (stratified to preserve class balance)
        train_idx, test_idx = train_test_split(
            reason_df.index,
            test_size=test_size,
            stratify=reason_df['label'],
            random_state=random_state
        )
        
        train_df = reason_df.loc[train_idx]
        test_df = reason_df.loc[test_idx]
        
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")
        
        # Fit isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(train_df['mmpose_confidence'], train_df['label'])
        
        # Get predictions on test set
        test_pred_proba = iso.predict(test_df['mmpose_confidence'])
        test_pred_binary = (test_pred_proba >= 0.5).astype(int)
        test_true = test_df['label'].values
        
        # Compute metrics
        ll = log_loss(test_true, test_pred_proba)
        brier = brier_score_loss(test_true, test_pred_proba)
        
        # ROC AUC (only if we have both classes in test set)
        if test_true.nunique() == 2:
            auc = roc_auc_score(test_true, test_pred_proba)
        else:
            auc = np.nan
        
        print(f"Log-Loss: {ll:.4f} | Brier: {brier:.4f} | AUC: {auc:.4f}")
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            test_true,
            test_pred_proba,
            n_bins=min(10, len(test_df) // 5),  # Adaptive bin count
            strategy='uniform'
        )
        
        # Plot on subplot
        ax = axes[idx]
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, 
                color=colors[idx], label=f'{reason} (n={n_samples})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{reason}\nLL={ll:.4f}, Brier={brier:.4f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Store results
        results.append({
            'reason': reason,
            'n_samples': n_samples,
            'n_visible': n_visible,
            'pct_visible': pct_visible,
            'log_loss': ll,
            'brier_score': brier,
            'auc': auc,
            'is_monotone': check_monotonicity(prob_pred, prob_true)
        })
    
    # Remove unused subplots
    for idx in range(len(results), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/calibration_curves_per_reason.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: calibration_curves_per_reason.png")
    
    return results


def check_monotonicity(prob_pred, prob_true):
    """
    Check if calibration curve is monotone increasing.
    
    Returns:
        bool: True if monotone, False otherwise
    """
    if len(prob_pred) < 2:
        return None
    
    # Check if prob_true is non-decreasing
    diffs = np.diff(prob_true)
    is_monotone = np.all(diffs >= -1e-6)  # Allow small numerical errors
    
    return is_monotone


def plot_combined_calibration(df, save_dir='./results', test_size=0.3, random_state=42):
    """
    Plot all calibration curves on a single figure for comparison.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    reasons = sorted([r for r in df['occlusion_reason'].unique() if pd.notna(r)])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(reasons)))
    
    for idx, reason in enumerate(reasons):
        reason_df = df[df['occlusion_reason'] == reason].copy()
        
        # Skip if too few samples
        if len(reason_df) < 20:
            continue
        
        # Skip if only one class
        if reason_df['label'].nunique() < 2:
            continue
        
        # Train/test split
        train_idx, test_idx = train_test_split(
            reason_df.index,
            test_size=test_size,
            stratify=reason_df['label'],
            random_state=random_state
        )
        
        train_df = reason_df.loc[train_idx]
        test_df = reason_df.loc[test_idx]
        
        # Fit isotonic
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(train_df['mmpose_confidence'], train_df['label'])
        
        # Predict
        test_pred_proba = iso.predict(test_df['mmpose_confidence'])
        test_true = test_df['label'].values
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            test_true,
            test_pred_proba,
            n_bins=min(10, len(test_df) // 5),
            strategy='uniform'
        )
        
        # Plot
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2.5, markersize=8,
                color=colors[idx], label=f'{reason} (n={len(reason_df)})')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves per Occlusion Reason', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/calibration_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: calibration_combined.png")


def print_results_table(results, save_dir='./results'):
    """
    Print results as a formatted table and save to CSV.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*100}")
    print("RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv(f'{save_dir}/calibration_results.csv', index=False)
    print(f"\n✅ Saved: calibration_results.csv")
    
    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}\n")
    print(f"Average Log-Loss: {results_df['log_loss'].mean():.4f}")
    print(f"Average Brier Score: {results_df['brier_score'].mean():.4f}")
    print(f"Reasons with Monotone Calibration: {results_df['is_monotone'].sum()}/{len(results_df)}")
    print(f"Non-Monotone Reasons: {results_df[~results_df['is_monotone']]['reason'].tolist()}")
    
    return results_df


def main(data_path, save_dir='./results', test_size=0.3, random_state=42):
    """
    Main pipeline for Experiment 1A.
    """

    

    df = pd.read_csv('/Users/emmavejcik/Downloads/Copy of Thesis Data Collection NEW - Ramona 1_1369 (4).csv')

    print("Visibility Category Distribution:")
    print(df['visibility_category'].value_counts().sort_index())

    print("\nCrosstab: Visibility vs. Occlusion Reason:")
    print(pd.crosstab(df['visibility_category'], df['occlusion_reason']))
    print(f"\n{'='*100}")
    print("EXPERIMENT 1A: Calibration Curves Per Occlusion Reason")
    print(f"{'='*100}\n")
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows\n")
    
    # Prepare data
    df = prepare_data(df)
    
    # Plot class distribution
    print("Plotting class distribution...")
    plot_class_distribution(df, save_dir)
    
    # Fit models and plot calibration curves per reason
    print("\nFitting isotonic regression per occlusion reason...")
    results = fit_and_evaluate_per_reason(df, save_dir, test_size, random_state)
    
    # Plot combined calibration curves
    print("\nPlotting combined calibration curves...")
    plot_combined_calibration(df, save_dir, test_size, random_state)
    
    # Print and save results
    print("\nPrinting results summary...")
    results_df = print_results_table(results, save_dir)
    
    print(f"\n{'='*100}")
    print(f"✅ EXPERIMENT COMPLETE. Results saved to: {save_dir}/")
    print(f"{'='*100}\n")
    
    return results_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description='Experiment 1A: Calibration Curves Per Occlusion Reason'
    )
    ap.add_argument('--data', required=True, help='Path to CSV file')
    ap.add_argument('--save_dir', default='./results', help='Directory to save results')
    ap.add_argument('--test-size', type=float, default=0.3, help='Test set fraction')
    ap.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = ap.parse_args()
    
    main(
        data_path=args.data,
        save_dir=args.save_dir,
        test_size=args.test_size,
        random_state=args.random_seed
    )