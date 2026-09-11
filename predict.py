import sys
import re
import joblib
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = name.replace("nagar", "").replace("Nagar", "")
    name = name.strip().lower()
    return re.sub(r'\s+', ' ', name)

def predict_district(district_query):
    # Load trained model artifacts
    clf = joblib.load('ml/models/champion_classifier.joblib')
    reg = joblib.load('ml/models/severity_regressor.joblib')
    
    # Load latest district feature baseline (2023)
    df = pd.read_csv('data/processed/district_year_dataset.csv')
    df_latest = df[df['year'] == 2023].copy()
    
    clean_q = clean_name(district_query)
    
    match = df_latest[df_latest['clean_dist'] == clean_q]
    if match.empty:
        match = df_latest[df_latest['clean_dist'].str.contains(clean_q, na=False)]
    if match.empty:
        match = df_latest[df_latest['Dist_Name'].str.lower().str.contains(clean_q, na=False)]
        
    if match.empty:
        print(f"\n[!] District '{district_query}' not found in the dataset.")
        print("Tip: Try common districts like 'Patna', 'Wayanad', 'Darbhanga', 'Gorakhpur', 'Murshidabad', 'Mumbai', 'Cuttack'.")
        return None

    row = match.iloc[0]
    dist_name = row['Dist_Name']
    
    feature_cols = [
        'prior_cum_events',
        'prior_3yr_events',
        'prior_5yr_events',
        'prior_10yr_events',
        'flood_acceleration_rate',
        'dfsi_score',
        'dfsi_rank',
        'Corrected_Percent_Flooded_Area',
        'Parmanent_Water',
        'Mean_Flood_Duration',
        'Population',
        'drainage_stress_ratio',
        'exposure_severity_index',
        'hydro_rain_stress',
        'rainfall_anomaly_pct'
    ]
    
    x_base = row[feature_cols].copy()
    
    scenarios = {
        "Drought / Deficit Rainfall (-30% Rain)": -30.0,
        "Normal Baseline (0% Anomaly)": 0.0,
        "Heavy Monsoon Surge (+20% Rain)": 20.0,
        "Severe Cloudburst / Cyclone (+45% Rain)": 45.0
    }
    
    print("=" * 75)
    print(f"MULTI-SOURCE FLOOD PROBABILITY REPORT: {dist_name.upper()}")
    print("=" * 75)
    print(f"* Historical DFSI Severity Rank:   #{int(row['dfsi_rank'])} of 640 districts")
    print(f"* Cumulative Recorded Floods:      {int(row['prior_cum_events'])} events (1967-2023 IFI-Impacts)")
    print(f"* Recent 3-Year Flood Recurrence:  {int(row['prior_3yr_events'])} events")
    print(f"* Recent 5-Year Flood Recurrence:  {int(row['prior_5yr_events'])} events")
    print(f"* Recurrence Acceleration Rate:    {row['flood_acceleration_rate']:+.2f} / yr trend")
    print(f"* Drainage Stress / Water Ratio:   {row['drainage_stress_ratio']:.2f}")
    print(f"* Exposed District Population:     {int(row['Population']):,}")
    print("-" * 75)
    print("AI MULTI-SCALE RISK PREDICTIONS:")
    print("-" * 75)

    for sc_name, rain_val in scenarios.items():
        x = x_base.copy()
        x['rainfall_anomaly_pct'] = rain_val
        x['hydro_rain_stress'] = (x['Corrected_Percent_Flooded_Area'] + 0.5) * (1.0 + (rain_val / 100.0))
        x_vec = np.nan_to_num(x.values.astype(float)).reshape(1, -1)
        
        prob = clf.predict_proba(x_vec)[0, 1]
        est_sev = reg.predict(x_vec)[0]
        
        if prob >= 0.70:
            badge = "[SEVERE ALERT]"
            action = "Immediate evacuation of low-lying floodplains; stage NDRF"
        elif prob >= 0.45:
            badge = "[HIGH RISK]"
            action = "Alert district disaster teams & monitor reservoir sluices"
        elif prob >= 0.25:
            badge = "[MODERATE WATCH]"
            action = "Routine monsoon monitoring & river gauge tracking"
        else:
            badge = "[LOW RISK]"
            action = "Normal drainage conditions expected"

        print(f"\n>> {sc_name}")
        print(f"   Calculated Probability:  {prob * 100:.2f}%  {badge}")
        print(f"   Estimated Inundation:    {est_sev:.1f} days duration")
        print(f"   Operational Directive:   {action}")
        
    print("=" * 75)

if __name__ == '__main__':
    query = sys.argv[1] if len(sys.argv) > 1 else "Patna"
    predict_district(query)
