import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from src.data_fetcher import fetch_cps_data
from src.models import fit_bayesian_bunching, fit_bayesian_synthetic_control
from utils.plotting import (
    plot_raw_distributions,
    plot_bunching_results,
    plot_synthetic_control_results,
    plot_donor_weights,
    plot_causal_treatment_effects,
    set_style
)

def main():
    print("Loading data...")
    df = fetch_cps_data()
    if df is None:
        raise ValueError("Data could not be loaded!")
        
    output_dir = "/Users/sam/Locals Only/sam-tritto.github.io/projects/Hawaii Benefit Cliff"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}")
    
    # 1. Raw distributions
    print("Plotting raw distributions...")
    fig1 = plot_raw_distributions(df)
    fig1.savefig(os.path.join(output_dir, "raw_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Fit Bunching
    print("Fitting Bunching model...")
    hi_df = df[df['state'] == 'HI']
    hi_counts = hi_df['hours'].value_counts().reindex(range(10, 31), fill_value=0)
    
    idata_bunching, lambda_pred = fit_bayesian_bunching(
        hi_counts,
        exclude_range=(18, 23),
        poly_degree=5,
        draws=1000,
        tune=1000
    )
    
    print("Plotting bunching results...")
    fig2 = plot_bunching_results(hi_counts.index.values, hi_counts.values, lambda_pred)
    fig2.savefig(os.path.join(output_dir, "bunching_results.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Fit Synthetic Control
    print("Fitting Synthetic Control model...")
    target_hours = range(10, 31)
    dists = df.groupby(['state', 'hours']).size().unstack(fill_value=0)
    dists = dists.reindex(columns=target_hours, fill_value=0)
    dists = dists.div(dists.sum(axis=1), axis=0) # Normalize within range (10-30 hours)
    
    hi_dist = dists.loc['HI']
    donor_dists = dists.drop('HI').T
    donor_names = donor_dists.columns.tolist()
    
    idata_sc, synthetic_hi_pred = fit_bayesian_synthetic_control(
        hi_dist,
        donor_dists,
        exclude_range=(15, 24),
        draws=1000,
        tune=1000
    )
    
    print("Plotting synthetic control results...")
    fig3 = plot_synthetic_control_results(target_hours, hi_dist, synthetic_hi_pred)
    fig3.savefig(os.path.join(output_dir, "synthetic_control_results.png"), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print("Plotting donor weights...")
    w_samples = az.extract(idata_sc, var_names="w").values.T
    fig4 = plot_donor_weights(donor_names, w_samples)
    fig4.savefig(os.path.join(output_dir, "donor_weights.png"), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    print("Plotting causal treatment effects...")
    fig5 = plot_causal_treatment_effects(target_hours, hi_dist, synthetic_hi_pred)
    fig5.savefig(os.path.join(output_dir, "causal_treatment_effects.png"), dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    print("All plots saved successfully!")

if __name__ == "__main__":
    main()
