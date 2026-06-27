import numpy as np
import pymc as pm
import arviz as az
import pandas as pd

def fit_bayesian_bunching(hours_counts, exclude_range=(18, 23), poly_degree=4, draws=1000, tune=1000):
    """
    Fits a Bayesian Poisson polynomial regression model to estimate the counterfactual
    distribution of hours worked, excluding the policy-distorted region.
    
    Parameters:
    -----------
    hours_counts : pd.Series
        Series where index is hours (int) and values are counts.
    exclude_range : tuple of (int, int)
        The range of hours to exclude from the fit (inclusive).
    poly_degree : int
        The degree of the polynomial to fit.
        
    Returns:
    --------
    idata : az.InferenceData
        The MCMC sampling results.
    lambda_pred : np.ndarray
        Posterior samples of the expected counts for all hours. Shape: (len(hours), n_samples)
    """
    h = hours_counts.index.values.astype(float)
    y = hours_counts.values
    
    # Define mask for non-excluded region
    mask = (h < exclude_range[0]) | (h > exclude_range[1])
    
    # Normalize running variable h to prevent numerical issues in polynomial power calculations
    h_mean = h.mean()
    h_std = h.std()
    h_scaled = (h - h_mean) / h_std
    
    # Construct polynomial design matrix
    X = np.column_stack([h_scaled**k for k in range(poly_degree + 1)])
    
    X_fit = X[mask]
    y_fit = y[mask]
    
    with pm.Model() as bunching_model:
        # Priors for polynomial coefficients
        # Using a relatively tight prior on higher-order terms for regularization
        beta = pm.Normal("beta", mu=0, sigma=10, shape=poly_degree + 1)
        
        # Expected counts (log link)
        log_mu = pm.math.dot(X_fit, beta)
        mu = pm.math.exp(log_mu)
        
        # Likelihood
        pm.Poisson("obs", mu=mu, observed=y_fit)
        
        # Run sampler
        idata = pm.sample(
            draws=draws, 
            tune=tune, 
            target_accept=0.95, 
            random_seed=42, 
            return_inferencedata=True,
            progressbar=False
        )
        
    # Extract posterior samples of beta
    beta_samples = az.extract(idata, var_names="beta").values.T # Shape: (n_samples, poly_degree + 1)
    
    # Predict counterfactual lambda for all hours
    log_lambda_pred = np.dot(X, beta_samples.T) # Shape: (len(h), n_samples)
    lambda_pred = np.exp(log_lambda_pred)
    
    return idata, lambda_pred


def fit_bayesian_synthetic_control(hi_dist, donor_dists, exclude_range=(15, 24), draws=1000, tune=1000):
    """
    Fits a Bayesian Synthetic Control model using the hours-worked distribution.
    Matches Hawaii's distribution to a weighted combination of donor states,
    excluding the policy-affected hours range.
    
    Parameters:
    -----------
    hi_dist : pd.Series
        Hawaii's hours worked distribution (proportions or counts).
    donor_dists : pd.DataFrame
        Donor states' hours worked distributions (columns = states, index = hours).
    exclude_range : tuple of (int, int)
        The hours range to exclude from the fitting process (inclusive).
        
    Returns:
    --------
    idata : az.InferenceData
        The MCMC sampling results.
    synthetic_hi_pred : np.ndarray
        Posterior predictions of Synthetic Hawaii's distribution. Shape: (len(hours), n_samples)
    """
    hours = hi_dist.index.values
    y_hi = hi_dist.values
    X_donors = donor_dists.values # Shape: (len(hours), n_donors)
    n_donors = X_donors.shape[1]
    
    # Mask for non-excluded hours
    mask = (hours < exclude_range[0]) | (hours > exclude_range[1])
    
    y_hi_fit = y_hi[mask]
    X_donors_fit = X_donors[mask]
    
    with pm.Model() as sc_model:
        # Dirichlet prior for weights guarantees:
        # 1. Weights are non-negative
        # 2. Weights sum to 1
        w = pm.Dirichlet("w", a=np.ones(n_donors))
        
        # Expected value for Hawaii's distribution in the non-excluded region
        mu = pm.math.dot(X_donors_fit, w)
        
        # Likelihood noise parameter
        sigma = pm.HalfNormal("sigma", sigma=0.05)
        
        # Likelihood
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_hi_fit)
        
        # Run sampler
        idata = pm.sample(
            draws=draws, 
            tune=tune, 
            target_accept=0.95, 
            random_seed=42, 
            return_inferencedata=True,
            progressbar=False
        )
        
    # Extract posterior weights
    w_samples = az.extract(idata, var_names="w").values.T # Shape: (n_samples, n_donors)
    
    # Predict Synthetic Hawaii for all hours
    # X_donors is shape (len(hours), n_donors)
    # w_samples.T is shape (n_donors, n_samples)
    synthetic_hi_pred = np.dot(X_donors, w_samples.T) # Shape: (len(hours), n_samples)
    
    return idata, synthetic_hi_pred
