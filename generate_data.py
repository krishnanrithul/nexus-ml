import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

n_rows = 50000
neighborhoods = ['Downtown', 'Riverside', 'Westside', 'Northgate', 'Lakefront', 
                 'Hillside', 'Eastwood', 'Midtown', 'Parkside', 'Sunnyville', 
                 'Craftsman', 'Victorian']

# Base features
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
    'list_price': np.random.normal(500000, 200000, n_rows).clip(50000, 2000000),
    'sale_date': [datetime(2023, 1, 1) + timedelta(days=int(x)) 
                  for x in np.random.uniform(0, 365, n_rows)],
    'num_beds_requested': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.2, 0.35, 0.25, 0.1]),
}

df = pd.DataFrame(data)

# Add temporal trend
df['months_since_start'] = (df['sale_date'] - df['sale_date'].min()).dt.days / 30
df['price_trend_adjustment'] = 1 + (df['months_since_start'] / 12) * 0.05

# Base price calculation
base_price = (
    50000 + df['sqft'] * 150 + df['bedrooms'] * 50000 + df['bathrooms'] * 40000 +
    df['lot_size'] * 100000 + (2024 - df['year_built']) * -500 + df['has_pool'] * 30000 +
    df['garage_spaces'] * 25000 + df['basement_sqft'] * 50 + df['has_updated_kitchen'] * 35000 +
    df['has_updated_bathrooms'] * 25000 + df['has_fireplace'] * 10000 + df['has_solar'] * 20000 +
    (100 - df['dist_to_downtown_mi']) * 5000 + (100 - df['dist_to_school_mi']) * 10000 +
    df['walkability_score'] * 200 + (100 - df['days_on_market']) * 100
)

# Neighborhood multipliers
neighborhood_multipliers = {
    'Downtown': 1.3, 'Riverside': 0.85, 'Westside': 1.0, 'Northgate': 0.9,
    'Lakefront': 1.4, 'Hillside': 1.1, 'Eastwood': 0.8, 'Midtown': 1.2,
    'Parkside': 1.15, 'Sunnyville': 0.95, 'Craftsman': 1.05, 'Victorian': 1.25
}
df['neighborhood_multiplier'] = df['neighborhood'].map(neighborhood_multipliers)

# Calculate price
df['price'] = base_price * df['price_trend_adjustment'] * df['neighborhood_multiplier']

# ========== ADD REAL-WORLD MESSINESS ==========

# 1. NEIGHBORHOOD MARKET ANOMALY: Downtown crashed mid-year
downtown_mask = (df['neighborhood'] == 'Downtown') & (df['months_since_start'] > 6)
df.loc[downtown_mask, 'price'] = df.loc[downtown_mask, 'price'] * 0.75

# 2. LOCATION PREMIUM OUTLIERS: Some expensive locations don't follow the formula
lakefront_outliers = (df['neighborhood'] == 'Lakefront') & (np.random.rand(len(df)) < 0.15)
df.loc[lakefront_outliers, 'price'] = df.loc[lakefront_outliers, 'price'] * np.random.uniform(1.3, 1.8, lakefront_outliers.sum())

# 3. HISTORICAL SIGNIFICANCE: Old Victorian homes with character command premium
victorian_old = (df['neighborhood'] == 'Victorian') & (df['year_built'] < 1950) & (df['condition'].isin(['Good', 'Excellent']))
df.loc[victorian_old, 'price'] = df.loc[victorian_old, 'price'] * np.random.uniform(1.4, 2.0, victorian_old.sum())

# 4. MARKET INEFFICIENCY: Some cheap houses sell for way more (bidding wars)
cheap_mask = (df['price'] < df['price'].quantile(0.3)) & (np.random.rand(len(df)) < 0.1)
df.loc[cheap_mask, 'price'] = df.loc[cheap_mask, 'price'] * np.random.uniform(1.5, 2.5, cheap_mask.sum())

# 5. DISTRESSED SALES: Some houses sell below expected (foreclosure, quick sale)
distressed_mask = (df['price'] > df['price'].quantile(0.7)) & (np.random.rand(len(df)) < 0.08)
df.loc[distressed_mask, 'price'] = df.loc[distressed_mask, 'price'] * np.random.uniform(0.5, 0.85, distressed_mask.sum())

# 6. MISSING CONDITION DATA causes imputation errors
condition_missing = np.random.choice(df.index, int(len(df) * 0.12), replace=False)
df.loc[condition_missing, 'condition'] = np.nan

# 7. FEATURE INTERACTIONS: Pool + hot_tub combo is worth way more
pool_spa = (df['has_pool'] == 1) & (df['has_hot_tub'] == 1)
df.loc[pool_spa, 'price'] = df.loc[pool_spa, 'price'] * 1.5

# 8. AGE CUTOFF: Pre-1950 homes have different valuation logic
very_old = (df['year_built'] < 1950) & (df['condition'].isin(['Good', 'Excellent']))
df.loc[very_old, 'price'] = df.loc[very_old, 'price'] * np.random.uniform(0.8, 1.3, very_old.sum())

# 9. SEASONAL VARIATION: Spring premium, winter discount
df['month'] = df['sale_date'].dt.month
spring_mask = df['month'].isin([3, 4, 5])
winter_mask = df['month'].isin([12, 1, 2])
df.loc[spring_mask, 'price'] = df.loc[spring_mask, 'price'] * 1.08
df.loc[winter_mask, 'price'] = df.loc[winter_mask, 'price'] * 0.93
df = df.drop('month', axis=1)

# 10. GENERAL NOISE: 20% random variation
df['price'] = df['price'] + np.random.normal(0, df['price'] * 0.2, len(df))

# Clip to realistic range
df['price'] = df['price'].clip(50000, 3000000).astype(int)

# Add realistic missing values
missing_columns = {
    'basement_sqft': 0.15,
    'last_renovated': 0.2,
    'fireplace_count': 0.1,
    'has_solar': 0.25,
    'water_heater_type': 0.05,
    'walkability_score': 0.08,
    'dist_to_transit_mi': 0.12,
}

for col, missing_rate in missing_columns.items():
    missing_idx = np.random.choice(df.index, int(len(df) * missing_rate), replace=False)
    df.loc[missing_idx, col] = np.nan

df = df.drop(['months_since_start', 'price_trend_adjustment', 'neighborhood_multiplier'], axis=1)

cols_order = ['neighborhood', 'zipcode', 'sqft', 'bedrooms', 'bathrooms', 'lot_size', 
              'year_built', 'stories', 'condition', 'roof_condition', 'last_renovated',
              'has_pool', 'has_garage', 'garage_spaces', 'has_basement', 'basement_sqft',
              'has_patio', 'has_deck', 'has_hot_tub', 'has_fireplace', 'fireplace_count',
              'has_sprinkler', 'has_security_system', 'has_new_roof', 'has_new_hvac',
              'has_new_electrical', 'has_new_plumbing', 'has_updated_kitchen', 
              'has_updated_bathrooms', 'has_central_ac', 'has_central_heat', 'has_gas',
              'has_solar', 'water_heater_type', 'dist_to_downtown_mi', 'dist_to_school_mi',
              'dist_to_transit_mi', 'dist_to_highway_mi', 'walkability_score', 
              'days_on_market', 'list_price', 'sale_date', 'num_beds_requested', 'price']

df = df[cols_order]

os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/house_prices_complex.csv', index=False)

print(f"✅ Generated {len(df):,} rows with {len(df.columns)} features")
print(f"📊 Saved to: data/raw/house_prices_complex.csv")
print(f"\nDataset chaos added:")
print(f"  ✓ Downtown market crash (25% drop mid-year)")
print(f"  ✓ Lakefront location premium outliers")
print(f"  ✓ Victorian historical significance premium")
print(f"  ✓ Market inefficiency (cheap houses sell high)")
print(f"  ✓ Distressed sales (expensive houses sell low)")
print(f"  ✓ Missing condition data (12%)")
print(f"  ✓ Feature interaction (pool + spa combo)")
print(f"  ✓ Age-based valuation anomalies")
print(f"  ✓ Seasonal price variations")
print(f"  ✓ 20% random noise (market inefficiency)")
print(f"\nPrice stats:")
print(f"  Range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"  Mean: ${df['price'].mean():,.0f}")
print(f"  Std: ${df['price'].std():,.0f}")
print(f"\nThis dataset will stress your model:")
print(f"  → Downtown RMSE will be much worse")
print(f"  → Lakefront will have high variance")
print(f"  → Victorian will confuse the model")
print(f"  → Missing condition data will hurt all neighborhoods")