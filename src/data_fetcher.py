import os
import pandas as pd
import numpy as np
import glob
import io

def pool_local_zips(raw_zips_dir='data/raw_zips', cache_path='data/hawaii_hours_data.csv'):
    """
    Finds all downloaded NBER CPS zip files in raw_zips_dir, processes them,
    and pools them into a single CSV.
    """
    fips_map = {
        15: 'HI',
        32: 'NV',
        12: 'FL',
        2: 'AK',
        6: 'CA'
    }
    target_fips = list(fips_map.keys())
    
    zip_files = glob.glob(os.path.join(raw_zips_dir, "cpsb*.zip"))
    if not zip_files:
        return None
        
    print(f"Found {len(zip_files)} downloaded zip files in '{raw_zips_dir}'. Pooling them...")
    
    pooled_data = []
    usecols_func = lambda col: col.lower() in ['gestfips', 'pehrusl1', 'hrmis', 'prcow1']
    
    for zip_path in sorted(zip_files):
        try:
            df = pd.read_csv(
                zip_path,
                usecols=usecols_func,
                compression='zip'
            )
            df.columns = [c.lower() for c in df.columns]
            
            # Filter: Target states, valid hours, Month-in-Sample 1, Private Sector
            df_filtered = df[
                df['gestfips'].isin(target_fips) & 
                (df['pehrusl1'] >= 1) & 
                (df['hrmis'] == 1) & 
                (df['prcow1'] == 4)
            ].copy()
            
            df_filtered['state'] = df_filtered['gestfips'].map(fips_map)
            df_filtered['hours'] = df_filtered['pehrusl1'].astype(int)
            
            pooled_data.append(df_filtered[['state', 'hours']])
            print(f"  Processed {os.path.basename(zip_path)}: {len(df_filtered):,} matching records.")
            
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(zip_path)}: {e}")
            
    if not pooled_data:
        print("❌ No data was successfully pooled from the zip files.")
        return None
        
    final_df = pd.concat(pooled_data, ignore_index=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    final_df.to_csv(cache_path, index=False)
    print(f"\n✅ Successfully pooled and saved {len(final_df):,} records to {cache_path}")
    return final_df

def fetch_cps_data(raw_zips_dir='data/raw_zips', cache_path='data/hawaii_hours_data.csv', overwrite=False, **kwargs):
    """
    Checks if a pooled CSV exists. If not, attempts to pool downloaded raw zip files.
    If no zip files are present, prints download instructions.
    """
    if os.path.exists(cache_path) and not overwrite:
        print(f"Dataset already cached at {cache_path}. Loading cached version...")
        return pd.read_csv(cache_path)
        
    # Attempt to pool local zips
    final_df = pool_local_zips(raw_zips_dir, cache_path)
    if final_df is not None:
        return final_df
        
    # If no local zips and cache is not valid, guide the user
    os.makedirs(raw_zips_dir, exist_ok=True)
    print(f"\n=== DATA DOWNLOAD REQUIRED ===")
    print("NBER's download servers are protected by Akamai CDN and block command-line scripts.")
    print("To bypass this, please download the monthly CPS basic data zip files for 2021-2023 manually.")
    print(f"Place all downloaded ZIP files into the folder: '{raw_zips_dir}/'\n")
    print("URLs to download (36 files total):")
    for year in [2021, 2022, 2023]:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            print(f"  https://data.nber.org/cps-basic3/csv/{year}/cpsb{year}{month_str}_csv.zip")
            
    print(f"\nOnce downloaded, run `data_fetcher.py` or re-run the verification/notebook to pool the data.")
    return None

if __name__ == "__main__":
    fetch_cps_data(overwrite=True)
