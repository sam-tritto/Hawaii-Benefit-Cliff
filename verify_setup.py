import sys
import os

def main():
    print("=== Diagnostic Verification Script ===")
    print(f"Python Version: {sys.version}")
    
    # 1. Check imports
    try:
        import numpy as np
        import pandas as pd
        import matplotlib as mpl
        import seaborn as sns
        import arviz as az
        import pymc as pm
        print("✅ Core data science and Bayesian packages imported successfully!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
        
    # 2. Run data fetcher check
    try:
        from src.data_fetcher import fetch_cps_data
        print("Checking/fetching dataset...")
        df = fetch_cps_data(year=2023)
        print("✅ Data fetcher script completed successfully!")
    except Exception as e:
        print(f"❌ Data fetching check failed: {e}")
        sys.exit(1)
        
    # 3. Test tiny PyMC sampling to verify pytensor and C compiler setup
    try:
        print("Testing a tiny PyMC model sampling run...")
        with pm.Model() as test_model:
            x = pm.Normal("x", mu=0, sigma=1)
            # Use small draws and tune to speed up verification
            idata = pm.sample(draws=100, tune=100, chains=1, progressbar=False, random_seed=42)
        print("✅ PyMC successfully compiled and sampled (1 chain, 100 draws)!")
    except Exception as e:
        print(f"❌ PyMC sampling failed: {e}")
        sys.exit(1)
        
    print("=== Verification Successful! Environment is fully ready. ===")

if __name__ == "__main__":
    main()
