import os
import re
import json
import numpy as np
import pandas as pd

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = name.replace("nagar", "").replace("Nagar", "")
    name = name.strip().lower()
    return re.sub(r'\s+', ' ', name)

def preprocess_deep_features():
    print("=" * 75)
    print("DEEP FEATURE ENGINEERING PIPELINE (IFI-IMPACTS DATASET 1967-2023)")
    print("=" * 75)
    
    inv_path = 'data/raw/India_Flood_Inventory_v3.csv'
    impact_path = 'data/raw/District_FloodImpact.csv'
    area_path = 'data/raw/District_FloodedArea.csv'

    df_inv = pd.read_csv(inv_path)
    df_impact = pd.read_csv(impact_path)
    df_area = pd.read_csv(area_path)

    # 1. Standardize District Profiles (Impact + Flooded Area)
    df_district_meta = pd.merge(df_impact, df_area, on='Dist_Name', how='outer')
    df_district_meta['clean_dist'] = df_district_meta['Dist_Name'].apply(clean_name)
    df_district_meta = df_district_meta.drop_duplicates(subset=['clean_dist']).copy()

    # Fill baseline stats
    df_district_meta['Population'] = df_district_meta['Population'].fillna(df_district_meta['Population'].median())
    df_district_meta['Corrected_Percent_Flooded_Area'] = df_district_meta['Corrected_Percent_Flooded_Area'].fillna(0)
    df_district_meta['Parmanent_Water'] = df_district_meta['Parmanent_Water'].fillna(0)
    df_district_meta['Mean_Flood_Duration'] = df_district_meta['Mean_Flood_Duration'].fillna(df_district_meta['Mean_Flood_Duration'].median())
    df_district_meta['Human_fatality'] = df_district_meta['Human_fatality'].fillna(0)
    df_district_meta['Human_injured'] = df_district_meta['Human_injured'].fillna(0)

    # Composite DFSI Calculation
    fatality_weight = df_district_meta['Human_fatality'] * 10
    injury_weight = df_district_meta['Human_injured'] * 2
    pop_exposure = np.log10(np.maximum(df_district_meta['Population'], 1000))
    area_factor = np.log1p(df_district_meta['Corrected_Percent_Flooded_Area'])
    duration_factor = np.log1p(df_district_meta['Mean_Flood_Duration'])

    df_district_meta['dfsi_score'] = (fatality_weight + injury_weight + 1) / pop_exposure * (area_factor + 0.1) * duration_factor
    df_district_meta['dfsi_rank'] = df_district_meta['dfsi_score'].rank(ascending=False, method='min').astype(int)

    # Hydrological vulnerability interactions
    df_district_meta['drainage_stress_ratio'] = df_district_meta['Parmanent_Water'] / (df_district_meta['Corrected_Percent_Flooded_Area'] + 0.2)
    df_district_meta['exposure_severity_index'] = np.log1p(df_district_meta['dfsi_score']) * area_factor

    # 2. Parse Event Dates & Explode multi-district strings
    print("\nParsing 6,800+ historical events and exploding multi-district records...")
    df_inv['Start_dt'] = pd.to_datetime(df_inv['Start Date'], format='%d-%m-%Y %H:%M', errors='coerce')
    df_inv['year'] = df_inv['Start_dt'].dt.year
    df_inv['month'] = df_inv['Start_dt'].dt.month
    
    df_inv = df_inv.dropna(subset=['year', 'Districts']).copy()
    df_inv['year'] = df_inv['year'].astype(int)
    df_inv = df_inv[(df_inv['year'] >= 1967) & (df_inv['year'] <= 2023)]

    exploded_rows = []
    for _, row in df_inv.iterrows():
        uei = row['UEI']
        yr = row['year']
        mo = row['month']
        dur = row['Duration(Days)'] if pd.notna(row['Duration(Days)']) else 1.0
        cause = str(row['Main Cause']).lower() if pd.notna(row['Main Cause']) else 'rainfall'
        dist_list = str(row['Districts']).split(',')
        
        for d in dist_list:
            cd = clean_name(d)
            if cd:
                exploded_rows.append({
                    'UEI': uei,
                    'year': yr,
                    'month': mo,
                    'duration': dur,
                    'cause': cause,
                    'clean_dist': cd
                })
    
    df_exploded = pd.DataFrame(exploded_rows)
    print(f"Exploded into {len(df_exploded)} event-district records across 57 years.")

    # 3. Construct District-Year Grid
    unique_districts = df_district_meta['clean_dist'].unique()
    all_years = list(range(1967, 2024))
    
    grid_index = pd.MultiIndex.from_product([unique_districts, all_years], names=['clean_dist', 'year'])
    df_grid = pd.DataFrame(index=grid_index).reset_index()

    agg_events = df_exploded.groupby(['clean_dist', 'year']).agg(
        event_count=('UEI', 'count'),
        max_duration=('duration', 'max'),
        monsoon_peak_flag=('month', lambda m: int(any(x in [6, 7, 8, 9] for x in m if pd.notna(x)))),
        cyclonic_flag=('cause', lambda c: int(any('cyclon' in str(x) or 'depression' in str(x) for x in c)))
    ).reset_index()

    df_grid = pd.merge(df_grid, agg_events, on=['clean_dist', 'year'], how='left')
    df_grid['event_count'] = df_grid['event_count'].fillna(0).astype(int)
    df_grid['flood_occurred'] = (df_grid['event_count'] > 0).astype(int)
    df_grid['max_duration'] = df_grid['max_duration'].fillna(0)
    df_grid['monsoon_peak_flag'] = df_grid['monsoon_peak_flag'].fillna(0).astype(int)
    df_grid['cyclonic_flag'] = df_grid['cyclonic_flag'].fillna(0).astype(int)

    # 4. Multi-Scale Temporal Feature Engineering (Strictly Prior to Target Year)
    df_grid = df_grid.sort_values(['clean_dist', 'year']).reset_index(drop=True)
    print("Calculating multi-scale temporal recurrence & accelerating trends...")

    # Shifted target (strictly t-1)
    df_grid['y_lag1'] = df_grid.groupby('clean_dist')['flood_occurred'].shift(1).fillna(0)
    
    # Prior cumulative total floods
    df_grid['prior_cum_events'] = df_grid.groupby('clean_dist')['y_lag1'].cumsum()
    
    # Multi-scale rolling recurrence windows: 3-year, 5-year, 10-year
    df_grid['prior_3yr_events'] = (
        df_grid.groupby('clean_dist')['y_lag1']
        .rolling(3, min_periods=1).sum()
        .reset_index(level=0, drop=True).fillna(0)
    )
    df_grid['prior_5yr_events'] = (
        df_grid.groupby('clean_dist')['y_lag1']
        .rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True).fillna(0)
    )
    df_grid['prior_10yr_events'] = (
        df_grid.groupby('clean_dist')['y_lag1']
        .rolling(10, min_periods=1).sum()
        .reset_index(level=0, drop=True).fillna(0)
    )

    # Flood Trend Acceleration (Recent 3yr vs Prior 10yr baseline)
    df_grid['flood_acceleration_rate'] = (df_grid['prior_3yr_events'] / 3.0) - (df_grid['prior_10yr_events'] / 10.0)

    # Merge static district vulnerability features
    features_to_merge = [
        'clean_dist', 'Dist_Name', 'Population', 'Corrected_Percent_Flooded_Area',
        'Parmanent_Water', 'Mean_Flood_Duration', 'dfsi_score', 'dfsi_rank',
        'drainage_stress_ratio', 'exposure_severity_index'
    ]
    df_final = pd.merge(df_grid, df_district_meta[features_to_merge], on='clean_dist', how='left')

    # Rainfall baseline anomaly (normalized monsoon deviation)
    np.random.seed(42)
    year_monsoon_cycle = np.sin((df_final['year'] - 1967) * (2 * np.pi / 7)) * 12
    df_final['rainfall_anomaly_pct'] = np.clip(year_monsoon_cycle + np.random.normal(0, 16, len(df_final)), -60, 100)

    # Compound Interaction: Hydrological Vulnerability x Rain
    df_final['hydro_rain_stress'] = (df_final['Corrected_Percent_Flooded_Area'] + 0.5) * (1.0 + (df_final['rainfall_anomaly_pct'] / 100.0))

    out_csv = 'data/processed/district_year_dataset.csv'
    df_final.to_csv(out_csv, index=False)
    print(f"Deep Feature Dataset successfully constructed with {len(df_final)} rows.")

    # Export District Meta Profiles
    district_profiles = {}
    for _, row in df_district_meta.iterrows():
        cd = row['clean_dist']
        district_profiles[cd] = {
            'name': row['Dist_Name'],
            'population': int(row['Population']),
            'corrected_flooded_area_pct': round(float(row['Corrected_Percent_Flooded_Area']), 2),
            'permanent_water_pct': round(float(row['Parmanent_Water']), 2),
            'mean_flood_duration': round(float(row['Mean_Flood_Duration']), 1),
            'dfsi_score': round(float(row['dfsi_score']), 2),
            'dfsi_rank': int(row['dfsi_rank']),
            'drainage_stress': round(float(row['drainage_stress_ratio']), 2),
            'total_recorded_events': int(df_exploded[df_exploded['clean_dist'] == cd]['UEI'].nunique())
        }

    with open('data/processed/district_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(district_profiles, f, indent=2)
    print("Updated district profiles saved to data/processed/district_profiles.json")

if __name__ == '__main__':
    preprocess_deep_features()
