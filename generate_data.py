import pandas as pd
import numpy as np

def generate_house_data():
    np.random.seed(42)
    n_samples = 500

    data = {
        'sqft': np.random.normal(2000, 500, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'neighborhood': np.random.choice(['North', 'South', 'East', 'West', 'Downtown'], n_samples),
        'has_pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        # target variable with some noise
        'price': 0 
    }

    df = pd.DataFrame(data)

    # Logic for price calculation (The "Truth" the models must find)
    df['price'] = (
        (df['sqft'] * 150) + 
        (df['bedrooms'] * 25000) + 
        (df['year_built'] - 1950) * 1000 + 
        np.random.normal(0, 15000, n_samples)
    ).astype(int)

    # Intentionally add some NULLs for the Wrangler Agent to find
    df.loc[np.random.choice(df.index, 20), 'sqft'] = np.nan
    df.loc[np.random.choice(df.index, 15), 'neighborhood'] = np.nan

    df.to_csv('data/raw/house_prices.csv', index=False)
    print("Success: 'data/raw/house_prices.csv' created with 500 samples.")

if __name__ == "__main__":
    import os
    os.makedirs('data/raw', exist_ok=True)
    generate_house_data()