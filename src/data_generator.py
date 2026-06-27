import os
import numpy as np
import pandas as pd

def generate_hours_data(n_workers_per_state=15000, seed=42):
    """
    Generates a realistic worker-level survey dataset of hours worked for Hawaii (HI)
    and four donor states (NV, FL, AK, CA).
    
    In Hawaii, the Prepaid Health Care Act creates a benefit cliff at 20 hours.
    Employers bunch hours at 19 to avoid the mandate.
    """
    np.random.seed(seed)
    states = ['HI', 'NV', 'FL', 'AK', 'CA']
    industries = ['Hospitality', 'Retail', 'Healthcare', 'Other']
    industry_probs = [0.35, 0.25, 0.15, 0.25]  # Highly tourism/service focused
    
    data = []
    
    # Base mixture parameters for hours worked (desired hours)
    # Mixture components:
    # 1. Full-time peak around 40 hours
    # 2. Part-time peak 1 around 20 hours
    # 3. Part-time peak 2 around 30 hours
    # 4. Uniform background noise (random part-time/overtime)
    
    for state in states:
        # Slightly vary mixture weights by state for realistic diversity
        if state == 'NV':
            weights = [0.50, 0.25, 0.12, 0.13]  # More part-time in Vegas
        elif state == 'HI':
            weights = [0.52, 0.20, 0.15, 0.13]
        elif state == 'AK':
            weights = [0.60, 0.15, 0.10, 0.15]  # More full-time/seasonal
        elif state == 'FL':
            weights = [0.55, 0.20, 0.13, 0.12]
        else:  # CA
            weights = [0.58, 0.18, 0.14, 0.10]
            
        n_workers = n_workers_per_state
        component = np.random.choice(4, size=n_workers, p=weights)
        
        desired_hours = np.zeros(n_workers)
        
        # Draw from components
        comp_0 = (component == 0)
        desired_hours[comp_0] = np.random.normal(40, 2.0, size=sum(comp_0))
        
        comp_1 = (component == 1)
        desired_hours[comp_1] = np.random.normal(20, 1.5, size=sum(comp_1))
        
        comp_2 = (component == 2)
        desired_hours[comp_2] = np.random.normal(30, 2.5, size=sum(comp_2))
        
        comp_3 = (component == 3)
        desired_hours[comp_3] = np.random.uniform(5, 55, size=sum(comp_3))
        
        # Round to integers as hours are reported in whole numbers in surveys like CPS
        hours = np.round(desired_hours).astype(int)
        hours = np.clip(hours, 1, 80)
        
        # Apply the benefit cliff distortion ONLY to Hawaii
        if state == 'HI':
            distorted_hours = hours.copy()
            for i in range(len(hours)):
                h = hours[i]
                # If a worker falls in the range [20, 23] hours, there is a high probability
                # that their employer caps their hours at 19 to avoid the healthcare mandate.
                if 20 <= h <= 23:
                    # Probability of distortion decays as we move away from 20 hours
                    prob_distortion = 0.75 - (h - 20) * 0.15
                    if np.random.rand() < prob_distortion:
                        distorted_hours[i] = 19
                # Some employers might even cut hours from 24-25 down to 19, but with lower probability
                elif 24 <= h <= 25:
                    if np.random.rand() < 0.15:
                        distorted_hours[i] = 19
            hours = distorted_hours
            
        # Simulate worker characteristics
        worker_industries = np.random.choice(industries, size=n_workers, p=industry_probs)
        
        # Hourly wage: base wage + industry effect + random noise
        base_wage = 18.0 if state in ['HI', 'CA'] else 15.0
        industry_wage_effects = {
            'Hospitality': -2.0,
            'Retail': -1.5,
            'Healthcare': 4.0,
            'Other': 1.0
        }
        wages = np.array([base_wage + industry_wage_effects[ind] for ind in worker_industries])
        # Add hours dependency (part-time workers might make slightly less per hour) and noise
        wages += (hours < 20) * -1.0 + np.random.normal(0, 3.0, size=n_workers)
        wages = np.round(np.clip(wages, 12.0, 75.0), 2)
        
        for idx in range(n_workers):
            data.append({
                'worker_id': f"{state}_{idx:05d}",
                'state': state,
                'hours': int(hours[idx]),
                'industry': worker_industries[idx],
                'wage': wages[idx]
            })
            
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating simulated worker-level dataset...")
    df = generate_hours_data()
    
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/hawaii_hours_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")
    print(f"Total records generated: {len(df)}")
    print("\nHours distribution summary:")
    print(df.groupby('state')['hours'].describe())
