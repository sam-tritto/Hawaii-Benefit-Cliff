import os
import pandas as pd
import numpy as np

def fetch_cps_data(year=2023, cache_path='data/hawaii_hours_data.csv'):
    """
    Downloads and pools CPS Basic Monthly CSV data from the NBER public repository.
    Filters for Hawaii and control states (NV, FL, AK, CA) and extracts usual weekly hours.
    """
    if os.path.exists(cache_path):
        print(f"Dataset already cached at {cache_path}. Loading cached version...")
        return pd.read_csv(cache_path)
        
    print(f"Fetching raw CPS Basic Monthly files for year {year} from NBER...")
    
    # State FIPS mappings:
    # 15: Hawaii (HI)
    # 32: Nevada (NV)
    # 12: Florida (FL)
    # 2:  Alaska (AK)
    # 6:  California (CA)
    fips_map = {
        15: 'HI',
        32: 'NV',
        12: 'FL',
        2: 'AK',
        6: 'CA'
    }
    
    target_fips = list(fips_map.keys())
    pooled_data = []
    
    # Iterate through all 12 months of the year
    for m in range(1, 13):
        month_str = f"{m:02d}"
        url = f"https://data.nber.org/cps-basic3/csv/{year}/cpsb{year}{month_str}_csv.zip"
        
        print(f"Downloading and reading {year}-{month_str}...")
        try:
            # Read columns of interest directly from zip file
            df = pd.read_csv(
                url, 
                usecols=['gestfips', 'pehrusl1', 'hrmis', 'prcow1'],
                compression='zip'
            )
            
            # Filter: Target states, valid usual hours, Month-in-Sample 1 (unique), and Private Sector (prcow1 == 4)
            df_filtered = df[
                df['gestfips'].isin(target_fips) & 
                (df['pehrusl1'] >= 1) & 
                (df['hrmis'] == 1) & 
                (df['prcow1'] == 4)
            ].copy()
            
            # Map FIPS to state codes
            df_filtered['state'] = df_filtered['gestfips'].map(fips_map)
            df_filtered['hours'] = df_filtered['pehrusl1'].astype(int)
            
            # Keep only the columns we need
            df_clean = df_filtered[['state', 'hours']]
            pooled_data.append(df_clean)
            print(f"  Successfully processed {len(df_clean):,} records.")
            
        except Exception as e:
            print(f"  ❌ Error downloading/processing month {month_str}: {e}")
            raise e
            
    # Concatenate all monthly slices
    final_df = pd.concat(pooled_data, ignore_index=True)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to file
    final_df.to_csv(cache_path, index=False)
    print(f"\n✅ Successfully pooled and saved {len(final_df):,} records to {cache_path}")
    
    return final_df

if __name__ == "__main__":
    fetch_cps_data()
