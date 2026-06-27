import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a premium color palette
PALETTE = {
    'primary_dark': '#1e293b',   # Slate 800
    'primary_light': '#f8fafc',  # Slate 50
    'hawaii': '#dc2626',         # Red 600 (Treatment)
    'synthetic': '#2563eb',      # Blue 600 (Counterfactual)
    'donor': '#64748b',          # Slate 500
    'accent': '#10b981',         # Emerald 500
    'shaded_red': '#fee2e2',     # Red 100
    'shaded_blue': '#dbeafe',    # Blue 100
    'grid': '#e2e8f0'            # Slate 200
}

def set_style():
    """Sets a clean, modern aesthetic for matplotlib plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'Inter', 'DejaVu Sans'],
        'axes.edgecolor': PALETTE['donor'],
        'axes.grid': True,
        'grid.color': PALETTE['grid'],
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'text.color': PALETTE['primary_dark'],
        'axes.labelcolor': PALETTE['primary_dark'],
        'xtick.color': PALETTE['donor'],
        'ytick.color': PALETTE['donor'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'none',
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def plot_raw_distributions(df, target_hours=range(10, 31)):
    """
    Plots the raw distribution of hours worked for Hawaii vs. Donor states.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Calculate proportions for Hawaii
    hi_df = df[df['state'] == 'HI']
    hi_counts = hi_df['hours'].value_counts(normalize=True).reindex(target_hours, fill_value=0)
    
    # Calculate average proportions for Donor states
    donor_df = df[df['state'] != 'HI']
    donor_counts = donor_df.groupby('state')['hours'].value_counts(normalize=True).unstack(fill_value=0)
    donor_avg = donor_counts.mean(axis=0).reindex(target_hours, fill_value=0)
    donor_std = donor_counts.std(axis=0).reindex(target_hours, fill_value=0)
    
    # Plot Donor States baseline (with standard deviation shading)
    ax.plot(target_hours, donor_avg, color=PALETTE['donor'], linewidth=2.5, 
            label='Donor States (Avg: CA, NV, FL, AK)', linestyle='-', marker='o', markersize=4)
    ax.fill_between(target_hours, donor_avg - donor_std, donor_avg + donor_std, 
                    color=PALETTE['donor'], alpha=0.15, label='Donor Variation (±1 SD)')
    
    # Plot Hawaii (highlighted)
    ax.plot(target_hours, hi_counts, color=PALETTE['hawaii'], linewidth=3.5, 
            label='Hawaii (Actual)', marker='o', markersize=6)
    
    # Threshold indicator line
    ax.axvline(20, color=PALETTE['primary_dark'], linestyle=':', linewidth=1.5)
    ax.text(19.8, ax.get_ylim()[1] * 0.9, 'Prepaid Health Care Act\nThreshold (20 Hours)', 
            ha='right', va='top', color=PALETTE['primary_dark'], fontsize=9, fontweight='bold')
    
    # Highlighting the distortion zone
    ax.axvspan(18.5, 19.5, color=PALETTE['shaded_red'], alpha=0.3, label='Hours Bunching Spike')
    ax.axvspan(19.5, 23.5, color=PALETTE['shaded_blue'], alpha=0.2, label='Hours Deficit (Cliff)')
    
    ax.set_title('Labor Schedule Distortions: Hawaii vs. Donor States', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Weekly Hours Worked', fontsize=11, labelpad=10)
    ax.set_ylabel('Proportion of Workforce', fontsize=11, labelpad=10)
    ax.set_xticks(target_hours)
    ax.set_xticklabels(target_hours)
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_bunching_results(hours, actual_counts, counterfactual_samples, exclude_range=(18, 23)):
    """
    Plots actual counts vs. Bayesian Bunching counterfactual predictions.
    """
    set_style()
    actual_counts = np.asarray(actual_counts)
    hours = np.asarray(hours)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Compute summary statistics of the counterfactual predictions
    cf_median = np.median(counterfactual_samples, axis=1)
    cf_lower = np.percentile(counterfactual_samples, 2.5, axis=1)
    cf_upper = np.percentile(counterfactual_samples, 97.5, axis=1)
    
    # Plot actual Hawaii counts
    ax.bar(hours, actual_counts, color=PALETTE['donor'], alpha=0.4, label='Hawaii Actual Counts', width=0.8)
    
    # Highlight the bunching spike at 19 hours
    ax.bar([19], [actual_counts[hours == 19][0]], color=PALETTE['hawaii'], alpha=0.8, label='Bunching Spike (19h)', width=0.8)
    
    # Plot counterfactual curve
    ax.plot(hours, cf_median, color=PALETTE['synthetic'], linewidth=2.5, 
            linestyle='--', label='Bayesian Bunching Counterfactual (Median)')
    ax.fill_between(hours, cf_lower, cf_upper, color=PALETTE['synthetic'], alpha=0.15, label='95% Credible Interval')
    
    # Highlight the excluded policy-distortion region
    ax.axvspan(exclude_range[0]-0.5, exclude_range[1]+0.5, color='#f1f5f9', alpha=0.5, label='Excluded Fit Zone', zorder=-1)
    
    # Threshold indicator line
    ax.axvline(20, color=PALETTE['primary_dark'], linestyle=':', linewidth=1.5)
    
    ax.set_title('Bayesian Bunching Estimation (Hawaii)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Weekly Hours Worked', fontsize=11, labelpad=10)
    ax.set_ylabel('Worker Count', fontsize=11, labelpad=10)
    ax.set_xticks(hours)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_synthetic_control_results(hours, actual_dist, synthetic_samples, exclude_range=(15, 24)):
    """
    Plots actual Hawaii distribution vs. Bayesian Synthetic Control distribution.
    """
    set_style()
    actual_dist = np.asarray(actual_dist)
    hours = np.asarray(hours)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Compute summary statistics of the synthetic predictions
    syn_median = np.median(synthetic_samples, axis=1)
    syn_lower = np.percentile(synthetic_samples, 2.5, axis=1)
    syn_upper = np.percentile(synthetic_samples, 97.5, axis=1)
    
    # Plot actual Hawaii distribution
    ax.plot(hours, actual_dist, color=PALETTE['hawaii'], linewidth=3.5, 
            label='Hawaii (Actual)', marker='o', markersize=5)
    
    # Plot Synthetic Hawaii
    ax.plot(hours, syn_median, color=PALETTE['synthetic'], linewidth=2.5, 
            linestyle='--', label='Synthetic Hawaii (Counterfactual)', marker='s', markersize=4)
    ax.fill_between(hours, syn_lower, syn_upper, color=PALETTE['synthetic'], alpha=0.15, label='95% Credible Interval')
    
    # Highlight policy exclusion region
    ax.axvspan(exclude_range[0]-0.5, exclude_range[1]+0.5, color='#f8fafc', alpha=0.6, label='Treated Zone (Excluded from Fit)', zorder=-1)
    
    # Threshold indicator line
    ax.axvline(20, color=PALETTE['primary_dark'], linestyle=':', linewidth=1.5)
    
    # Shading the difference in the treated region to show net causal effect
    fit_mask = (hours >= exclude_range[0]) & (hours <= exclude_range[1])
    ax.fill_between(
        hours[fit_mask], 
        actual_dist[fit_mask], 
        syn_median[fit_mask], 
        where=(actual_dist[fit_mask] > syn_median[fit_mask]),
        color=PALETTE['accent'], alpha=0.3, label='Excess Labor (Bunching)'
    )
    ax.fill_between(
        hours[fit_mask], 
        actual_dist[fit_mask], 
        syn_median[fit_mask], 
        where=(actual_dist[fit_mask] < syn_median[fit_mask]),
        color=PALETTE['hawaii'], alpha=0.2, label='Suppressed Labor (Deficit)'
    )
    
    ax.set_title('Synthetic Control Counterfactual: Hawaii vs. Synthetic Hawaii', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Weekly Hours Worked', fontsize=11, labelpad=10)
    ax.set_ylabel('Proportion of Workforce', fontsize=11, labelpad=10)
    ax.set_xticks(hours)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_donor_weights(donor_names, w_samples):
    """
    Plots the posterior distribution of donor weights for the Synthetic Control.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    
    # Compute median and credible intervals
    weights_df = pd.DataFrame(w_samples, columns=donor_names)
    medians = weights_df.median().sort_values(ascending=True)
    lower = weights_df.quantile(0.025)[medians.index]
    upper = weights_df.quantile(0.975)[medians.index]
    
    y_pos = np.arange(len(medians))
    
    # Plot horizontal bar chart with error bars
    ax.barh(y_pos, medians, xerr=[medians - lower, upper - medians], 
            color=PALETTE['synthetic'], alpha=0.8, edgecolor=PALETTE['primary_dark'], height=0.6,
            error_kw=dict(ecolor=PALETTE['primary_dark'], lw=1.5, capsize=4))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(medians.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Posterior Weight', fontsize=11, labelpad=10)
    ax.set_title('Bayesian Synthetic Control: Donor State Weights', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    
    # Add values on the bars
    for i, v in enumerate(medians):
        ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=10, color=PALETTE['primary_dark'], fontweight='bold')
        
    plt.tight_layout()
    return fig
