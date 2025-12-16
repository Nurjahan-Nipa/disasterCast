#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -p gpu2
#SBATCH -A loni_hdr_llm01
#SBATCH --gres=gpu:1
#SBATCH --job-name=temporal_h3_graph_ablation
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -e

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# Force UTF-8 encoding
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8

# ==========================================
# CONFIGURATION
# ==========================================
CITIES=("tampa" "cape_coral" "jacksonville" "miami" "orlando")
HORIZONS=("d14" "d21")
ENCODERS=("gru" "lstm" "cnn" "gru_cnn")
MODES=("temp_only" "temp_h3")  # Each encoder tested with and without H3
SEED=42
H3_EMBED_PATH="data/h3_embeddings_all.pkl"  # ← NEW: Path to graph embeddings

TOTAL_MODELS=$((${#CITIES[@]} * ${#HORIZONS[@]} * ${#ENCODERS[@]} * ${#MODES[@]}))

echo "=========================================="
echo "TEMPORAL ENCODER ABLATION - GRAPH H3"
echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Cities: ${CITIES[@]}"
echo "Horizons: ${HORIZONS[@]}"
echo "Encoders: ${ENCODERS[@]}"
echo "Modes: ${MODES[@]} (with and without H3)"
echo "H3 Embeddings: $H3_EMBED_PATH"
echo "Total models: $TOTAL_MODELS"
echo "Estimated time: ~20 hours"
echo "Using: train2.py (log-space + graph H3 embeddings)"
echo ""
echo "=========================================="
echo ""

# Check if H3 embeddings exist
if [ ! -f "$H3_EMBED_PATH" ]; then
    echo "ERROR: H3 embeddings not found at $H3_EMBED_PATH"
    echo "Please run: python step1b_create_h3_embeddings.py first"
    exit 1
fi

current=0

# ==========================================
# LOOP THROUGH ALL CITIES
# ==========================================

for city in "${CITIES[@]}"; do
    echo ""
    echo "========================================"
    echo "CITY: ${city^^}"
    echo "========================================"
    
    # ==========================================
    # LOOP THROUGH BOTH HORIZONS
    # ==========================================
    
    for horizon in "${HORIZONS[@]}"; do
        echo ""
        echo "  Horizon: ${horizon}"
        echo "  ----------------------------------------"
        
        BASE="data/splits/${city}_${horizon}"
        OUT_BASE="results/${city}_${horizon}_temporal_ablation_graph_h3"  # ← NEW: Different output dir
        mkdir -p "$OUT_BASE"
        
        # Check if data exists
        if [ ! -f "${BASE}_train.jsonl" ]; then
            echo "  WARNING: Data not found for ${city} ${horizon}, skipping..."
            continue
        fi
        
        # Train all encoder+mode combinations for this city+horizon
        for encoder in "${ENCODERS[@]}"; do
            for mode in "${MODES[@]}"; do
                current=$((current + 1))
                
                echo ""
                echo "  [${current}/${TOTAL_MODELS}] ${city} + ${horizon} + ${encoder} + ${mode}"
                
                # Set H3 flag and hidden dim based on mode and encoder
                if [ "$mode" == "temp_h3" ]; then
                    USE_H3="--use-h3 --h3-embed-path $H3_EMBED_PATH"  # ← NEW: Add path
                    MODE_NAME="${encoder}+H3"
                else
                    USE_H3=""
                    MODE_NAME="${encoder}"
                fi
                
                # Use 256 hidden dim for CNN, 64 for others
                if [ "$encoder" == "cnn" ]; then
                    HIDDEN_DIM=256
                else
                    HIDDEN_DIM=64
                fi
                
                OUT_DIR="${OUT_BASE}/${mode}_${encoder}"
                
                # Training - CHANGED to train2.py
                echo "    Training: ${MODE_NAME} (hidden_dim=${HIDDEN_DIM})..."
                python train2.py \
                    --train ${BASE}_train.jsonl \
                    --val ${BASE}_val.jsonl \
                    --horizon $horizon \
                    --arch-mode $mode \
                    $USE_H3 \
                    --temp-encoder $encoder \
                    --temp-hidden-dim $HIDDEN_DIM \
                    --h3-embed-dim 64 \
                    --batch-size 8 \
                    --epochs 3 \
                    --lr 0.001 \
                    --output-dir $OUT_DIR \
                    --seed $SEED || { echo "    ERROR: Training failed!"; continue; }
                
                # Evaluation - CHANGED to evaluate2.py
                echo "    Evaluating..."
                python evaluate2.py \
                    --test ${BASE}_test.jsonl \
                    --horizon $horizon \
                    --checkpoint $OUT_DIR \
                    --batch-size 8 \
                    --output ${OUT_DIR}/test_pred.csv \
                    --use-best || { echo "    ERROR: Evaluation failed!"; continue; }
                
                # Quick display
                if [ -f ${OUT_DIR}/test_pred.metrics.json ]; then
                    python -c "
import json
m = json.load(open('${OUT_DIR}/test_pred.metrics.json'))
print('    SUCCESS: ${MODE_NAME} - MAE={:.4f}, R2={:.4f}'.format(m['MAE'], m['R2']))
" 2>/dev/null || echo "    SUCCESS: Results saved"
                fi
            done
        done
        
        # Save results for this city+horizon
        export CITY=$city
        export HORIZON=$horizon
        export OUT_BASE=$OUT_BASE
        
        python << 'PYEOF'
import json
import pandas as pd
from pathlib import Path
import sys
import os

# Force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

city = os.environ.get("CITY", "unknown")
horizon = os.environ.get("HORIZON", "d14")
out_base = Path(os.environ.get("OUT_BASE", "."))

encoders = ["gru", "lstm", "cnn", "gru_cnn"]
modes = ["temp_only", "temp_h3"]

results = []

for encoder in encoders:
    for mode in modes:
        metrics_path = out_base / f"{mode}_{encoder}" / "test_pred.metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    m = json.load(f)
                
                # Create model name
                if mode == "temp_h3":
                    model_name = f"{encoder.upper().replace('_', '+')}+H3"
                else:
                    model_name = encoder.upper().replace('_', '+')
                
                results.append({
                    "City": city,
                    "Horizon": horizon,
                    "Encoder": encoder,
                    "H3": "Graph" if mode == "temp_h3" else "No",  # ← Changed to indicate graph
                    "Model": model_name,
                    "MAE": round(m["MAE"], 4),
                    "RMSE": round(m["RMSE"], 4),
                    "R2": round(m["R2"], 4),
                    "sMAPE": round(m.get("sMAPE(%)", 0), 4),
                    "RMSLE": round(m.get("RMSLE", 0), 4)
                })
            except Exception as e:
                print(f"  Error loading {mode}_{encoder}: {e}", file=sys.stderr)

if results:
    df = pd.DataFrame(results)
    
    # Save individual summary
    try:
        df.to_csv(out_base / "summary.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(f"  Error saving summary: {e}", file=sys.stderr)
    
    # Save to shared directory
    try:
        shared_dir = Path("results/temporal_ablation_graph_h3")  # ← NEW: Different shared dir
        shared_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(shared_dir / f"{city}_{horizon}_results.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(f"  Error saving to shared: {e}", file=sys.stderr)
    
    # Print comparison
    try:
        print(f"\n  Results for {city} {horizon}:")
        print(f"  {'Model':<20s} {'MAE':>8s} {'R2':>8s} {'Change':>12s}")
        print(f"  {'-'*52}")
        
        # Group by encoder
        for encoder in encoders:
            df_enc = df[df['Encoder'] == encoder].sort_values('H3')
            if len(df_enc) == 0:
                continue
            
            # Without H3 (base)
            base = df_enc[df_enc['H3'] == 'No']
            if len(base) > 0:
                base = base.iloc[0]
                print(f"  {base['Model']:<20s} {base['MAE']:>8.4f} {base['R2']:>8.4f}")
            
            # With H3
            h3 = df_enc[df_enc['H3'] == 'Graph']
            if len(h3) > 0 and len(base) > 0:
                h3 = h3.iloc[0]
                mae_diff = h3['MAE'] - base['MAE']
                
                if mae_diff < 0:
                    diff_str = f"↓ {abs(mae_diff):.4f}"
                else:
                    diff_str = f"↑ {mae_diff:.4f}"
                
                print(f"  {h3['Model']:<20s} {h3['MAE']:>8.4f} {h3['R2']:>8.4f} {diff_str:>12s}")
        
        print()
        
        # Best overall
        best = df.loc[df['MAE'].idxmin()]
        print(f"  Best: {best['Model']} (MAE={best['MAE']:.4f}, R2={best['R2']:.4f})")
        
    except Exception as e:
        print(f"  Error in comparison: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
PYEOF

    done
    
    echo ""
    echo "  ${city^^} complete!"
    
done

# ==========================================
# FINAL SUMMARY
# ==========================================
echo ""
echo "========================================"
echo "GENERATING FINAL SUMMARY"
echo "========================================"
echo ""

python << 'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

results_dir = Path("results/temporal_ablation_graph_h3")  # ← NEW: Different dir
output_file = Path("results/temporal_ablation_graph_h3_final.csv")  # ← NEW: Different output

result_files = sorted(results_dir.glob("*_results.csv"))

if not result_files:
    print("ERROR: No result files found!")
    sys.exit(1)

print(f"Found {len(result_files)} result files\n")

all_results = []
for f in result_files:
    try:
        df = pd.read_csv(f, encoding='utf-8')
        all_results.append(df)
    except Exception as e:
        print(f"Error reading {f.name}: {e}")

if not all_results:
    print("ERROR: No valid results!")
    sys.exit(1)

df_final = pd.concat(all_results, ignore_index=True)
df_final = df_final.sort_values(["City", "Horizon", "Encoder", "H3"])

# Save
df_final.to_csv(output_file, index=False, encoding='utf-8')

print("="*100)
print("FINAL RESULTS SUMMARY - TEMPORAL ENCODERS WITH GRAPH H3 EMBEDDINGS")
print("="*100)
print()

# Summary by model
print("AVERAGE PERFORMANCE BY MODEL:")
print()
summary = df_final.groupby("Model").agg({
    "MAE": ["mean", "std", "min", "max"],
    "R2": ["mean", "std", "min", "max"]
}).round(4)
print(summary)
print()

# H3 Impact Analysis
print("="*100)
print("GRAPH H3 SPATIAL EMBEDDING IMPACT")
print("="*100)
print()

for encoder in ["gru", "lstm", "cnn", "gru_cnn"]:
    df_enc = df_final[df_final['Encoder'] == encoder]
    
    no_h3 = df_enc[df_enc['H3'] == 'No']
    with_h3 = df_enc[df_enc['H3'] == 'Graph']
    
    if len(no_h3) > 0 and len(with_h3) > 0:
        # Merge for comparison
        merged = pd.merge(
            no_h3[['City', 'Horizon', 'MAE', 'R2']],
            with_h3[['City', 'Horizon', 'MAE', 'R2']],
            on=['City', 'Horizon'],
            suffixes=('_base', '_h3')
        )
        
        mae_improvements = merged['MAE_base'] - merged['MAE_h3']
        r2_improvements = merged['R2_h3'] - merged['R2_base']
        
        better_mae = (mae_improvements > 0).sum()
        better_r2 = (r2_improvements > 0).sum()
        total = len(merged)
        
        avg_mae_change = mae_improvements.mean()
        avg_r2_change = r2_improvements.mean()
        
        enc_name = encoder.upper().replace('_', '+')
        print(f"{enc_name}:")
        print(f"  Graph H3 improves MAE:  {better_mae}/{total} cases ({better_mae/total*100:.1f}%)")
        print(f"  Graph H3 improves R2:   {better_r2}/{total} cases ({better_r2/total*100:.1f}%)")
        print(f"  Avg MAE change:   {avg_mae_change:+.4f}")
        print(f"  Avg R2 change:    {avg_r2_change:+.4f}")
        print()

# Best encoder overall
print("="*100)
print("BEST ENCODER BY CRITERION")
print("="*100)
print()

print("By Average MAE:")
best_mae = df_final.groupby("Model")["MAE"].mean().sort_values()
for i, (model, mae) in enumerate(best_mae.head(5).items(), 1):
    print(f"  {i}. {model:<20s}: {mae:.4f}")

print("\nBy Average R2:")
best_r2 = df_final.groupby("Model")["R2"].mean().sort_values(ascending=False)
for i, (model, r2) in enumerate(best_r2.head(5).items(), 1):
    print(f"  {i}. {model:<20s}: {r2:.4f}")

# By horizon
print("\n" + "="*100)
print("BEST BY HORIZON")
print("="*100)

for h in sorted(df_final["Horizon"].unique()):
    print(f"\n{h.upper()}:")
    df_h = df_final[df_final["Horizon"] == h]
    
    print("  By MAE:")
    for i, (model, mae) in enumerate(df_h.groupby("Model")["MAE"].mean().sort_values().head(3).items(), 1):
        print(f"    {i}. {model:<20s}: {mae:.4f}")
    
    print("  By R2:")
    for i, (model, r2) in enumerate(df_h.groupby("Model")["R2"].mean().sort_values(ascending=False).head(3).items(), 1):
        print(f"    {i}. {model:<20s}: {r2:.4f}")

print("\n" + "="*100)
print(f"Total models trained: {len(df_final)}")
print(f"Saved: {output_file}")
print("="*100)
PYEOF

echo ""
echo "========================================"
echo "ALL CITIES COMPLETE"
echo "========================================"
echo "Job finished: $(date)"
echo ""
echo "Final results: results/temporal_ablation_graph_h3_final.csv"
echo ""