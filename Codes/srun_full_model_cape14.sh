#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -p gpu2
#SBATCH -A loni_hdr_llm02
#SBATCH --gres=gpu:1
#SBATCH --job-name=log_full_cnn_cape_coral_d14
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

export TF_ENABLE_ONEDNN_OPTS=0
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8

CITY="cape_coral"
HORIZON="d14"
MODEL="meta-llama/Meta-Llama-3-8B"
QUANT="4bit"
EPOCHS=3

BASE="data/splits/${CITY}_${HORIZON}"
OUT="results/${CITY}_${HORIZON}_full_cnn_train_64_log"

echo "=========================================="
echo "FULL MODEL (LOG-SPACE): ${CITY^^} ${HORIZON}"
echo "=========================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

mkdir -p $OUT

echo "TRAINING..."
python train.py \
    --train ${BASE}_train.jsonl \
    --val ${BASE}_val.jsonl \
    --horizon $HORIZON \
    --arch-mode full \
    --use-h3 \
    --model-name $MODEL \
    --quantization $QUANT \
    --temp-encoder cnn \
    --temp-hidden-dim 64 \
    --h3-embed-dim 64 \
    --lambda-mse 1.0 \
    --batch-size 4 \
    --epochs $EPOCHS \
    --lr 0.0001 \
    --output-dir $OUT \
    --seed 42

if [ $? -ne 0 ]; then echo "ERROR: Training failed!"; exit 1; fi
echo "SUCCESS: Training complete!"
echo ""

echo "EVALUATING..."
python evaluate.py \
    --test ${BASE}_test.jsonl \
    --horizon $HORIZON \
    --checkpoint $OUT \
    --model-name $MODEL \
    --quantization $QUANT \
    --batch-size 4 \
    --output ${OUT}/predictions.csv \
    --use-best

if [ $? -ne 0 ]; then echo "ERROR: Evaluation failed!"; exit 1; fi
echo "SUCCESS: Evaluation complete!"
echo ""

if [ -f ${OUT}/predictions.metrics.json ]; then
    python -c "import json; m=json.load(open('${OUT}/predictions.metrics.json')); print('MAE: {:.4f}, R2: {:.4f}'.format(m['MAE'], m['R2']))"
fi

echo ""
echo "Job finished: $(date)"