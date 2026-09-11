"""
AQUA HORIZON — Master Hybrid Model Training & Benchmarking Orchestrator
Executes training for all 5 mandatory hackathon hybrid architectures:
1. U-Net + ConvLSTM
2. CNN + LSTM
3. CNN + Transformer
4. ResNet + BiLSTM
5. Attention U-Net + LSTM

Generates unified comparison benchmark:
- ml/models/hybrid_benchmarks.json
- web/data/hybrid_benchmarks.json
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.train_01_unet_convlstm import train_u_net___convlstm
from ml.train_02_cnn_lstm import train_cnn___lstm
from ml.train_03_cnn_transformer import train_cnn___transformer
from ml.train_04_resnet_bilstm import train_resnet___bilstm
from ml.train_05_attention_unet_lstm import train_attention_u_net___lstm

def run_all_benchmarks():
    print("=" * 90)
    print("AQUA HORIZON — 5 HYBRID DEEP LEARNING ARCHITECTURES TRAINING SUITE")
    print("Zenodo IFI-Impacts Dataset (1967-2023) | 725 Districts | 5-Year Temporal Sequences")
    print("=" * 90)
    
    t0 = time.time()
    results = []
    
    models = [
        ("1. U-Net + ConvLSTM", train_u_net___convlstm),
        ("2. CNN + LSTM", train_cnn___lstm),
        ("3. CNN + Transformer", train_cnn___transformer),
        ("4. ResNet + BiLSTM", train_resnet___bilstm),
        ("5. Attention U-Net + LSTM", train_attention_u_net___lstm)
    ]
    
    for name, train_fn in models:
        print(f"\n>>> Starting Pipeline: {name} ...")
        res = train_fn()
        results.append(res)
        
    total_time = time.time() - t0
    
    summary = {
        "suite_name": "Aqua Horizon 5-Hybrid Deep Learning Benchmark",
        "dataset": "Zenodo IFI-Impacts (1967-2023)",
        "evaluation_split": "Chronological 80:20 Split (1967-2012 Train vs 2013-2023 Test)",
        "sequence_length_years": 5,
        "total_districts": 725,
        "total_training_time_sec": round(total_time, 2),
        "models": results
    }
    
    # Save in ml/models/
    ml_models_path = os.path.join(os.path.dirname(__file__), 'models', 'hybrid_benchmarks.json')
    with open(ml_models_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Save in web/data/ for live dashboard UI
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'data'))
    os.makedirs(web_dir, exist_ok=True)
    web_models_path = os.path.join(web_dir, 'hybrid_benchmarks.json')
    with open(web_models_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 90)
    print("ALL 5 HYBRID MODELS TRAINED & BENCHMARKED SUCCESSFULLY!")
    print(f"Saved: {ml_models_path}")
    print(f"Saved: {web_models_path}")
    print("=" * 90)
    
    print(f"{'Model Architecture':<26} | {'Parameters':<10} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 80)
    for m in results:
        print(f"{m['model_name']:<26} | {m['parameters']:<10,d} | {m['recall']:<7.2f}% | {m['precision']:<9.2f}% | {m['f1_score']:<8.4f} | {m['roc_auc']:<8.4f}")
    print("=" * 90)

if __name__ == '__main__':
    run_all_benchmarks()
