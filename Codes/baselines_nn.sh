#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -p gpu2
#SBATCH -A loni_hdr_llm01
#SBATCH --gres=gpu:1
#SBATCH --job-name=nn_baselines_50

echo "========================================================================"
echo "MASTER NN BASELINE RUNNER"
echo "5 Models x 5 Cities x 2 Horizons = 50 Experiments"
echo "========================================================================"
echo ""

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Data directory containing per-city train/test files
DATA_DIR="data/splits"

# Output directory
OUTPUT_DIR="results/nn_baselines_complete"

# Horizons to process
HORIZONS=("d14" "d21")

# Cities to process
CITIES=("tampa" "orlando" "cape_coral" "miami" "jacksonville")

# Models to run
MODELS=("RNN" "LSTM" "GRU" "CNN" "GRU+CNN")

# Training hyperparameters
EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=0.001
PATIENCE=10
SEED=42

# ============================================================================
# FILE NAMING PATTERNS - ADJUST IF NEEDED
# ============================================================================

# Pattern 1: {city}_{horizon}_train.jsonl / {city}_{horizon}_val.jsonl / {city}_{horizon}_test.jsonl
get_train_file() {
    local city=$1
    local horizon=$2
    echo "$DATA_DIR/${city}_${horizon}_train.jsonl"
}

get_val_file() {
    local city=$1
    local horizon=$2
    echo "$DATA_DIR/${city}_${horizon}_val.jsonl"
}

get_test_file() {
    local city=$1
    local horizon=$2
    echo "$DATA_DIR/${city}_${horizon}_test.jsonl"
}

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/run_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Configuration:" | tee "$LOG_FILE"
echo "  Data directory: $DATA_DIR" | tee -a "$LOG_FILE"
echo "  Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Horizons: ${HORIZONS[@]}" | tee -a "$LOG_FILE"
echo "  Cities: ${CITIES[@]}" | tee -a "$LOG_FILE"
echo "  Models: ${MODELS[@]}" | tee -a "$LOG_FILE"
echo "  Epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "  Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Learning rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "  Patience: $PATIENCE" | tee -a "$LOG_FILE"
echo "  Seed: $SEED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if Python script exists
if [ ! -f "baselines_nn.py" ]; then
    echo "[ERROR] baselines_nn.py not found!" | tee -a "$LOG_FILE"
    echo "Make sure you're running this from the correct directory." | tee -a "$LOG_FILE"
    exit 1
fi

# ============================================================================
# VERIFY DATA FILES
# ============================================================================

echo "Verifying data files..." | tee -a "$LOG_FILE"
MISSING_FILES=0

for HORIZON in "${HORIZONS[@]}"; do
    for CITY in "${CITIES[@]}"; do
        TRAIN_FILE=$(get_train_file "$CITY" "$HORIZON")
        VAL_FILE=$(get_val_file "$CITY" "$HORIZON")
        TEST_FILE=$(get_test_file "$CITY" "$HORIZON")
        
        if [ ! -f "$TRAIN_FILE" ]; then
            echo "  [MISSING] $TRAIN_FILE" | tee -a "$LOG_FILE"
            MISSING_FILES=$((MISSING_FILES + 1))
        fi
        
        if [ ! -f "$VAL_FILE" ]; then
            echo "  [MISSING] $VAL_FILE" | tee -a "$LOG_FILE"
            MISSING_FILES=$((MISSING_FILES + 1))
        fi
        
        if [ ! -f "$TEST_FILE" ]; then
            echo "  [MISSING] $TEST_FILE" | tee -a "$LOG_FILE"
            MISSING_FILES=$((MISSING_FILES + 1))
        fi
    done
done

if [ $MISSING_FILES -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[ERROR] $MISSING_FILES data file(s) not found!" | tee -a "$LOG_FILE"
    echo "Please update the DATA_DIR and file naming pattern in this script." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Looking for JSONL files in $DATA_DIR:" | tee -a "$LOG_FILE"
    find "$DATA_DIR" -name "*.jsonl" -type f 2>/dev/null | head -10 | tee -a "$LOG_FILE"
    exit 1
fi

echo "[OK] All data files found" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

START_TIME=$(date +%s)
TOTAL_EXPERIMENTS=$((${#HORIZONS[@]} * ${#MODELS[@]} * ${#CITIES[@]}))
CURRENT_EXP=0
SUCCESSFUL=0
FAILED=0

echo "========================================================================"
echo "STARTING $TOTAL_EXPERIMENTS EXPERIMENTS"
echo "========================================================================"
echo ""

# Track results for summary
declare -A RESULTS_MAE
declare -A RESULTS_R2

for HORIZON in "${HORIZONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for CITY in "${CITIES[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            echo "========================================================================"
            echo "Experiment $CURRENT_EXP/$TOTAL_EXPERIMENTS"
            echo "Model: $MODEL | City: $CITY | Horizon: $HORIZON"
            echo "========================================================================"
            
            # Get file paths
            TRAIN_FILE=$(get_train_file "$CITY" "$HORIZON")
            VAL_FILE=$(get_val_file "$CITY" "$HORIZON")
            TEST_FILE=$(get_test_file "$CITY" "$HORIZON")
            OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_${CITY}_${HORIZON}.json"
            
            echo "Train: $TRAIN_FILE"
            echo "Val: $VAL_FILE"
            echo "Test: $TEST_FILE"
            echo "Output: $OUTPUT_FILE"
            echo ""
            
            # Log to file
            echo "[$CURRENT_EXP/$TOTAL_EXPERIMENTS] $MODEL - $CITY - $HORIZON - Started: $(date)" >> "$LOG_FILE"
            
            # Run experiment
            python baselines_nn.py \
                --train "$TRAIN_FILE" \
                --val "$VAL_FILE" \
                --test "$TEST_FILE" \
                --horizon "$HORIZON" \
                --encoder "$MODEL" \
                --hidden-size 64 \
                --num-layers 2 \
                --dropout 0.3 \
                --epochs $EPOCHS \
                --batch-size $BATCH_SIZE \
                --lr $LEARNING_RATE \
                --patience $PATIENCE \
                --seed $SEED \
                --output "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"
            
            EXIT_CODE=${PIPESTATUS[0]}
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "[SUCCESS] $MODEL - $CITY - $HORIZON" | tee -a "$LOG_FILE"
                SUCCESSFUL=$((SUCCESSFUL + 1))
                
                # Extract metrics for summary
                if [ -f "$OUTPUT_FILE" ]; then
                    MAE=$(python3 -c "import json; print(json.load(open('$OUTPUT_FILE'))['metrics']['MAE'])" 2>/dev/null)
                    R2=$(python3 -c "import json; print(json.load(open('$OUTPUT_FILE'))['metrics']['R2'])" 2>/dev/null)
                    RESULTS_MAE["${MODEL}_${CITY}_${HORIZON}"]=$MAE
                    RESULTS_R2["${MODEL}_${CITY}_${HORIZON}"]=$R2
                fi
            else
                echo "[FAILED] $MODEL - $CITY - $HORIZON (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
                FAILED=$((FAILED + 1))
            fi
            
            echo "Progress: $SUCCESSFUL successful, $FAILED failed" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
            
            # Small delay between experiments
            sleep 1
        done
    done
done

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Average time per experiment: $((ELAPSED / TOTAL_EXPERIMENTS))s"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Generate comprehensive summary table
echo "========================================================================"
echo "RESULTS SUMMARY"
echo "========================================================================"

python3 << 'EOFPYTHON'
import json
import sys
from pathlib import Path

output_dir = Path("results/baselines_nn_complete")
models = ["RNN", "LSTM", "GRU", "CNN", "GRU+CNN"]
cities = ["tampa", "orlando", "cape_coral", "miami", "jacksonville"]
horizons = ["d14", "d21"]

for horizon in horizons:
    print(f"\n{'='*90}")
    print(f"HORIZON: {horizon.upper()} - MAE (Mean Absolute Error)")
    print(f"{'='*90}")
    
    print(f"{'Model':<12}", end="")
    for city in cities:
        print(f"{city:<15}", end="")
    print("Mean")
    print("-" * 90)
    
    for model in models:
        print(f"{model:<12}", end="")
        model_maes = []
        
        for city in cities:
            result_file = output_dir / f"{model}_{city}_{horizon}.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        results = json.load(f)
                        mae = results["metrics"]["MAE"]
                        model_maes.append(mae)
                        print(f"{mae:<15.2f}", end="")
                except:
                    print(f"{'ERROR':<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        
        if model_maes:
            avg_mae = sum(model_maes) / len(model_maes)
            print(f"{avg_mae:.2f}")
        else:
            print("N/A")
    
    print()
    print(f"{'='*90}")
    print(f"HORIZON: {horizon.upper()} - R2 Score")
    print(f"{'='*90}")
    
    print(f"{'Model':<12}", end="")
    for city in cities:
        print(f"{city:<15}", end="")
    print("Mean")
    print("-" * 90)
    
    for model in models:
        print(f"{model:<12}", end="")
        model_r2s = []
        
        for city in cities:
            result_file = output_dir / f"{model}_{city}_{horizon}.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        results = json.load(f)
                        r2 = results["metrics"]["R2"]
                        model_r2s.append(r2)
                        print(f"{r2:<15.4f}", end="")
                except:
                    print(f"{'ERROR':<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        
        if model_r2s:
            avg_r2 = sum(model_r2s) / len(model_r2s)
            print(f"{avg_r2:.4f}")
        else:
            print("N/A")
    
    # Find best model per city for this horizon
    print(f"\nBEST MODEL PER CITY - {horizon.upper()} (by MAE):")
    print("-" * 50)
    
    for city in cities:
        best_model = None
        best_mae = float('inf')
        
        for model in models:
            result_file = output_dir / f"{model}_{city}_{horizon}.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        results = json.load(f)
                        mae = results["metrics"]["MAE"]
                        if mae < best_mae:
                            best_mae = mae
                            best_model = model
                except:
                    pass
        
        if best_model:
            print(f"  {city:<15}: {best_model:<10} (MAE: {best_mae:.2f})")
        else:
            print(f"  {city:<15}: N/A")
    
    print()

EOFPYTHON

echo ""
echo "========================================================================"
echo "GENERATING CSV OUTPUT"
echo "========================================================================"

# Generate CSV with all 50 results
python results_to_csv.py \
    --results-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/all_results.csv"

if [ -f "$OUTPUT_DIR/all_results.csv" ]; then
    echo ""
    echo "[SUCCESS] CSV file created: $OUTPUT_DIR/all_results.csv"
    NUM_ROWS=$(tail -n +2 "$OUTPUT_DIR/all_results.csv" | wc -l)
    echo "Total rows: $NUM_ROWS/50"
else
    echo "[WARNING] CSV generation failed"
fi

echo ""
echo "========================================================================"
echo "DONE!"
echo "========================================================================"