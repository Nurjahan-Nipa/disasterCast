#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu2
#SBATCH -A loni_hdr_llm01
#SBATCH --gres=gpu:1
#SBATCH --job-name=llm_ablation
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N

set -e

echo "=========================================="
echo "LLM ABLATION STUDY - 3 PROMPT TEMPLATES"
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

# ==========================================
# CONFIGURATION
# ==========================================

MODEL="meta-llama/Meta-Llama-3-8B"
QUANT="4bit"
BATCH_SIZE=8
SEED=42

# Cities to process
CITIES=("tampa" "orlando" "cape_coral" "miami" "jacksonville")

# Horizons to process
HORIZONS=("d14" "d21")

# Templates for ablation study
TEMPLATES=("A" "B" "C")

echo "Global Configuration:"
echo "  Model: $MODEL"
echo "  Quantization: $QUANT"
echo "  Batch Size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo "  Cities: ${CITIES[@]}"
echo "  Horizons: ${HORIZONS[@]}"
echo "  Templates: ${TEMPLATES[@]}"
echo ""
echo "Template Descriptions:"
echo "  A = Minimalist (temporal-only baseline)"
echo "  B = Hurricane-aware (temporal + event context)"
echo "  C = Full context (semantic + temporal + hurricane)"
echo ""

# Create output directories
mkdir -p results/llm_ablation/template_A
mkdir -p results/llm_ablation/template_B
mkdir -p results/llm_ablation/template_C

# ==========================================
# RUN PREDICTIONS FOR ALL COMBINATIONS
# ==========================================

total_runs=$((${#CITIES[@]} * ${#HORIZONS[@]} * ${#TEMPLATES[@]}))
current_run=0

for TEMPLATE in "${TEMPLATES[@]}"; do
    echo ""
    echo "######################################################"
    echo "# TEMPLATE $TEMPLATE ABLATION"
    echo "######################################################"
    echo ""
    
    for CITY in "${CITIES[@]}"; do
        for HORIZON in "${HORIZONS[@]}"; do
            current_run=$((current_run + 1))
            
            echo "=========================================="
            echo "Run $current_run/$total_runs"
            echo "Template: $TEMPLATE | City: $CITY | Horizon: $HORIZON"
            echo "=========================================="
            
            BASE="data/splits/${CITY}_${HORIZON}_test"
            OUTPUT="results/llm_ablation/template_${TEMPLATE}/${CITY}_${HORIZON}_template${TEMPLATE}.json"
            
            echo "Input file: ${BASE}.jsonl"
            echo "Output file: ${OUTPUT}"
            echo ""
            
            # Check if input file exists
            if [ ! -f "${BASE}.jsonl" ]; then
                echo "??  WARNING: File ${BASE}.jsonl not found! Skipping..."
                echo ""
                continue
            fi
            
            # Run prediction with selected template
            python llm_only_3_prompt.py \
                --test ${BASE}.jsonl \
                --horizon ${HORIZON} \
                --template ${TEMPLATE} \
                --model-name ${MODEL} \
                --quantization ${QUANT} \
                --batch-size ${BATCH_SIZE} \
                --seed ${SEED} \
                --output ${OUTPUT}
            
            echo ""
            echo "? Completed: Template $TEMPLATE - $CITY - $HORIZON"
            echo "  Saved to: ${OUTPUT}"
            echo ""
            
        done
    done
    
    echo ""
    echo "??? Template $TEMPLATE complete!"
    echo ""
done

echo "=========================================="
echo "ALL ABLATION EXPERIMENTS COMPLETED"
echo "Job finished: $(date)"
echo "=========================================="
echo ""

# ==========================================
# SUMMARY BY TEMPLATE
# ==========================================

echo "=========================================="
echo "RESULTS SUMMARY BY TEMPLATE"
echo "=========================================="
echo ""

for TEMPLATE in "${TEMPLATES[@]}"; do
    echo "Template $TEMPLATE Results:"
    echo "-------------------------"
    
    for CITY in "${CITIES[@]}"; do
        for HORIZON in "${HORIZONS[@]}"; do
            OUTPUT="results/llm_ablation/template_${TEMPLATE}/${CITY}_${HORIZON}_template${TEMPLATE}.json"
            if [ -f "${OUTPUT}" ]; then
                echo "  ? ${CITY}_${HORIZON}: $(ls -lh ${OUTPUT} | awk '{print $5}')"
            else
                echo "  ? ${CITY}_${HORIZON}: NOT FOUND"
            fi
        done
    done
    echo ""
done

# ==========================================
# QUICK METRICS EXTRACTION (if jq available)
# ==========================================

if command -v jq &> /dev/null; then
    echo "=========================================="
    echo "QUICK METRICS COMPARISON (R² scores)"
    echo "=========================================="
    echo ""
    
    for TEMPLATE in "${TEMPLATES[@]}"; do
        echo "Template $TEMPLATE:"
        for CITY in "${CITIES[@]}"; do
            for HORIZON in "${HORIZONS[@]}"; do
                OUTPUT="results/llm_ablation/template_${TEMPLATE}/${CITY}_${HORIZON}_template${TEMPLATE}.json"
                if [ -f "${OUTPUT}" ]; then
                    R2=$(jq -r '.metrics.R2' ${OUTPUT} 2>/dev/null || echo "N/A")
                    MAE=$(jq -r '.metrics.MAE' ${OUTPUT} 2>/dev/null || echo "N/A")
                    printf "  %s_%s: R²=%.4f, MAE=%.2f\n" "$CITY" "$HORIZON" "$R2" "$MAE" 2>/dev/null || echo "  ${CITY}_${HORIZON}: Error reading metrics"
                fi
            done
        done
        echo ""
    done
else
    echo "Note: Install 'jq' for automatic metrics extraction"
fi

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="