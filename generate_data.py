import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_rows = 50000
neighborhoods = ['Downtown', 'Riverside', 'Westside', 'Northgate', 'Lakefront', 
                 'Hillside', 'Eastwood', 'Midtown', 'Parkside', 'Sunnyville', 
                 'Craftsman', 'Victorian']

data = {
    'neighborhood': np.random.choice(neighborhoods, n_rows),
    'zipcode': np.random.choice([10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008], n_rows),
    'sqft': np.random.normal(2000, 600, n_rows).astype(int).clip(500, 6000),
    'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], n_rows, p=[0.05, 0.15, 0.35, 0.3, 0.12, 0.03]),
    'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_rows, p=[0.08, 0.12, 0.3, 0.2, 0.2, 0.07, 0.03]),
    'lot_size': np.random.normal(0.25, 0.15, n_rows).clip(0.05, 2.0),
    'year_built': np.random.choice(range(1920, 2024), n_rows),
    'stories': np.random.choice([1, 1.5, 2, 2.5, 3], n_rows, p=[0.2, 0.05, 0.5, 0.1, 0.15]),
    'has_pool': np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    'has_garage': np.random.choice([0, 1], n_rows, p=[0.1, 0.9]),
    'garage_spaces': np.random.choice([0, 1, 2, 3, 4], n_rows, p=[0.1, 0.4, 0.35, 0.1, 0.05]),
    'has_basement': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
    'basement_sqft': np.random.choice([0, 500, 1000, 1500, 2000], n_rows, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
    'has_patio': np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
    'has_deck': np.random.choice([0, 1], n_rows, p=[0.5, 0.5]),
    'has_hot_tub': np.random.choice([0, 1], n_rows, p=[0.95, 0.05]),
    'has_fireplace': np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
    'fireplace_count': np.random.choice([0, 1, 2, 3], n_rows, p=[0.4, 0.45, 0.1, 0.05]),
    'has_sprinkler': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
    'has_security_system': np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
    'condition': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent', 'Like New'],
                                  n_rows, p=[0.05, 0.1, 0.4, 0.35, 0.1]),
    'roof_condition': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'],
                                       n_rows, p=[0.1, 0.2, 0.5, 0.2]),
    'last_renovated': np.random.choice(range(1980, 2024), n_rows),
    'has_new_roof': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    'has_new_hvac': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
    'has_new_electrical': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    'has_new_plumbing': np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    'has_updated_kitchen': np.random.choice([0, 1], n_rows, p=[0.5, 0.5]),
    'has_updated_bathrooms': np.random.choice([0, 1], n_rows, p=[0.6, 0.4]),
    'has_central_ac': np.random.choice([0, 1], n_rows, p=[0.2, 0.8]),
    'has_central_heat': np.random.choice([0, 1], n_rows, p=[0.15, 0.85]),
    'has_gas': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
    'has_solar': np.random.choice([0, 1], n_rows, p=[0.95, 0.05]),
    'water_heater_type': np.random.choice(['Gas', 'Electric', 'Tankless', 'Solar'],
                                          n_rows, p=[0.5, 0.3, 0.15, 0.05]),
    'dist_to_downtown_mi': np.random.exponential(2.0, n_rows).clip(0, 20),
    'dist_to_school_mi': np.random.exponential(0.5, n_rows).clip(0, 5),
    'dist_to_transit_mi': np.random.exponential(1.0, n_rows).clip(0, 10),
    'dist_to_highway_mi': np.random.exponential(1.5, n_rows).clip(0, 15),
    'walkability_score': np.random.choice(range(0, 101), n_rows),
    'days_on_market': np.random.exponential(30, n_rows).astype(int).clip(1, 365),
    'num_beds_requested': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.2, 0.35, 0.25, 0.1]),
}

df = pd.DataFrame(data)

# Base price — linear part
base_price = (
    50000 + df['sqft'] * 150 + df['bedrooms'] * 50000 + df['bathrooms'] * 40000 +
    df['lot_size'] * 100000 + (2024 - df['year_built']) * -500 + df['has_pool'] * 30000 +
    df['garage_spaces'] * 25000 + df['basement_sqft'] * 50 + df['has_updated_kitchen'] * 35000 +
    df['has_updated_bathrooms'] * 25000 + df['has_fireplace'] * 10000 + df['has_solar'] * 20000 +
    (100 - df['dist_to_downtown_mi']) * 5000 + (100 - df['dist_to_school_mi']) * 10000 +
    df['walkability_score'] * 200 + (100 - df['days_on_market']) * 100
)

# Non-linear interactions LR can't fit
base_price = base_price * np.log1p(df['sqft'] / 1000)
base_price = base_price * (df['year_built'] / 2024) ** 2
base_price += df['sqft'] * df['bedrooms'] * 50
base_price += np.where(df['has_pool'] & df['has_hot_tub'], 80000, 0)

# Neighborhood multipliers
neighborhood_multipliers = {
    'Downtown': 1.3, 'Riverside': 0.85, 'Westside': 1.0, 'Northgate': 0.9,
    'Lakefront': 1.4, 'Hillside': 1.1, 'Eastwood': 0.8, 'Midtown': 1.2,
    'Parkside': 1.15, 'Sunnyville': 0.95, 'Craftsman': 1.05, 'Victorian': 1.25
}
df['neighborhood_multiplier'] = df['neighborhood'].map(neighborhood_multipliers)
df['price'] = base_price * df['neighborhood_multiplier']

# ── MESSINESS ────────────────────────────────────────────────

# Lakefront outliers
lakefront_outliers = (df['neighborhood'] == 'Lakefront') & (np.random.rand(len(df)) < 0.15)
df.loc[lakefront_outliers, 'price'] *= np.random.uniform(1.3, 1.8, lakefront_outliers.sum())

# Victorian historical premium
victorian_old = (df['neighborhood'] == 'Victorian') & (df['year_built'] < 1950) & (df['condition'].isin(['Good', 'Excellent']))
df.loc[victorian_old, 'price'] *= np.random.uniform(1.4, 2.0, victorian_old.sum())

# Bidding wars on cheap homes
cheap_mask = (df['price'] < df['price'].quantile(0.3)) & (np.random.rand(len(df)) < 0.1)
df.loc[cheap_mask, 'price'] *= np.random.uniform(1.5, 2.5, cheap_mask.sum())

# Distressed sales
distressed_mask = (df['price'] > df['price'].quantile(0.7)) & (np.random.rand(len(df)) < 0.08)
df.loc[distressed_mask, 'price'] *= np.random.uniform(0.5, 0.85, distressed_mask.sum())

# Pool + hot tub combo
pool_spa = (df['has_pool'] == 1) & (df['has_hot_tub'] == 1)
df.loc[pool_spa, 'price'] *= 1.5

# Pre-1950 age anomaly
very_old = (df['year_built'] < 1950) & (df['condition'].isin(['Good', 'Excellent']))
df.loc[very_old, 'price'] *= np.random.uniform(0.8, 1.3, very_old.sum())

# 20% noise
df['price'] += np.random.normal(0, df['price'] * 0.2, len(df))
df['price'] = df['price'].clip(50000, 3000000).astype(int)

# Missing condition data
condition_missing = np.random.choice(df.index, int(len(df) * 0.12), replace=False)
df.loc[condition_missing, 'condition'] = np.nan

# Other missing values
missing_columns = {
    'basement_sqft': 0.15, 'last_renovated': 0.2, 'fireplace_count': 0.1,
    'has_solar': 0.25, 'water_heater_type': 0.05, 'walkability_score': 0.08,
    'dist_to_transit_mi': 0.12,
}
for col, rate in missing_columns.items():
    idx = np.random.choice(df.index, int(len(df) * rate), replace=False)
    df.loc[idx, col] = np.nan

df = df.drop(['neighborhood_multiplier'], axis=1)

cols_order = ['neighborhood', 'zipcode', 'sqft', 'bedrooms', 'bathrooms', 'lot_size',
              'year_built', 'stories', 'condition', 'roof_condition', 'last_renovated',
              'has_pool', 'has_garage', 'garage_spaces', 'has_basement', 'basement_sqft',
              'has_patio', 'has_deck', 'has_hot_tub', 'has_fireplace', 'fireplace_count',
              'has_sprinkler', 'has_security_system', 'has_new_roof', 'has_new_hvac',
              'has_new_electrical', 'has_new_plumbing', 'has_updated_kitchen',
              'has_updated_bathrooms', 'has_central_ac', 'has_central_heat', 'has_gas',
              'has_solar', 'water_heater_type', 'dist_to_downtown_mi', 'dist_to_school_mi',
              'dist_to_transit_mi', 'dist_to_highway_mi', 'walkability_score',
              'days_on_market', 'num_beds_requested', 'price']

df = df[cols_order]

os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/house_prices_complex.csv', index=False)

print(f"Generated {len(df):,} rows, {len(df.columns)} features")
print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"Price mean:  ${df['price'].mean():,.0f}")