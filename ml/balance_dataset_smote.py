import os
import argparse
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

NUMERIC_FEATURE_COLS = [
    'event_count', 'max_duration', 'monsoon_peak_flag', 'cyclonic_flag',
    'y_lag1', 'prior_cum_events', 'prior_3yr_events', 'prior_5yr_events',
    'prior_10yr_events', 'flood_acceleration_rate', 'Population',
    'Corrected_Percent_Flooded_Area', 'Parmanent_Water', 'Mean_Flood_Duration',
    'dfsi_score', 'dfsi_rank', 'drainage_stress_ratio', 'exposure_severity_index',
    'rainfall_anomaly_pct', 'hydro_rain_stress'
]

TARGET_COL = 'flood_occurred'

def balance_dataset_with_smote(
    input_csv='data/processed/district_year_dataset.csv',
    output_csv='data/processed/district_year_dataset_balanced_smote.csv',
    k_neighbors=5,
    random_state=42,
    overwrite_original=False
):
    print("=" * 70)
    print("AQUA HORIZON — SMOTE DATASET BALANCING PIPELINE")
    print("=" * 70)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(base_dir, input_csv)
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(base_dir, output_csv)

    print(f"Loading raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    n_orig = len(df)
    
    orig_counts = df[TARGET_COL].value_counts().to_dict()
    orig_n0 = orig_counts.get(0, 0)
    orig_n1 = orig_counts.get(1, 0)
    print(f"Original Records: {n_orig:,}")
    print(f"  Class 0 (Non-Flood):  {orig_n0:,} ({orig_n0/n_orig*100:.2f}%)")
    print(f"  Class 1 (Flood Event): {orig_n1:,} ({orig_n1/n_orig*100:.2f}%)")
    print(f"  Imbalance Ratio:       {orig_n0 / max(orig_n1, 1):.2f} : 1")
    
    # Clean numeric features
    features_present = [col for col in NUMERIC_FEATURE_COLS if col in df.columns]
    df[features_present] = df[features_present].fillna(0).replace([np.inf, -np.inf], 0)
    
    X = df[features_present]
    y = df[TARGET_COL]

    print(f"\nFitting SMOTE (k_neighbors={k_neighbors}, random_state={random_state})...")
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    
    n_total = len(X_res)
    n_synth = n_total - n_orig
    print(f"SMOTE Synthesis Complete:")
    print(f"  Synthesized Flood Samples: +{n_synth:,}")
    print(f"  Total Balanced Records:     {n_total:,}")
    
    # Nearest neighbor mapping for synthesized points to inherit metadata
    minority_mask = (y == 1).values
    minority_indices = np.where(minority_mask)[0]
    X_minority = X.iloc[minority_indices].values
    
    print("Mapping synthetic instances to parent district profiles...")
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_minority)
    X_synth = X_res.iloc[n_orig:].values
    _, nearest_idx = nn.kneighbors(X_synth)
    nearest_orig_indices = minority_indices[nearest_idx.flatten()]
    
    meta_cols = ['clean_dist', 'Dist_Name', 'year']
    available_meta = [c for c in meta_cols if c in df.columns]
    synth_meta = df.iloc[nearest_orig_indices][available_meta].copy().reset_index(drop=True)
    
    df_synth = pd.concat([synth_meta, X_res.iloc[n_orig:].reset_index(drop=True)], axis=1)
    df_synth[TARGET_COL] = 1
    
    # Assemble full balanced dataframe matching original column order
    df_balanced = pd.concat([df, df_synth[df.columns]], ignore_index=True)
    
    bal_counts = df_balanced[TARGET_COL].value_counts().to_dict()
    bal_n0 = bal_counts.get(0, 0)
    bal_n1 = bal_counts.get(1, 0)
    
    print(f"\nFinal Balanced Dataset Verification:")
    print(f"  Total Rows:           {len(df_balanced):,}")
    print(f"  Class 0 (Non-Flood):  {bal_n0:,} ({bal_n0/len(df_balanced)*100:.1f}%)")
    print(f"  Class 1 (Flood Event): {bal_n1:,} ({bal_n1/len(df_balanced)*100:.1f}%)")
    print(f"  Status:               PERFECTLY BALANCED (50:50)")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_balanced.to_csv(output_csv, index=False)
    print(f"\nSaved balanced dataset to: {output_csv}")
    
    if overwrite_original:
        backup_path = input_csv.replace('.csv', '_unbalanced_backup.csv')
        df.to_csv(backup_path, index=False)
        df_balanced.to_csv(input_csv, index=False)
        print(f"Overwrote original file ({input_csv}) and backed up original to {backup_path}")

    return df_balanced

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Balance dataset using SMOTE")
    parser.add_argument('--input', default='data/processed/district_year_dataset.csv', help="Input CSV path")
    parser.add_argument('--output', default='data/processed/district_year_dataset_balanced_smote.csv', help="Output CSV path")
    parser.add_argument('--k-neighbors', type=int, default=5, help="Number of nearest neighbors for SMOTE")
    parser.add_argument('--overwrite-original', action='store_true', help="Also overwrite original CSV file with backup")
    
    args = parser.parse_args()
    balance_dataset_with_smote(
        input_csv=args.input,
        output_csv=args.output,
        k_neighbors=args.k_neighbors,
        overwrite_original=args.overwrite_original
    )
