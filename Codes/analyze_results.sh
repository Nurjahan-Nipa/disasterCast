#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -p gpu2
#SBATCH -A loni_hdr_llm02
#SBATCH --gres=gpu:1
#SBATCH --job-name=complete_ablation
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err



# ============================
# CAPE CORAL
# ============================
python analyze_results.py --csv results/cape_coral_d14_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/cape_coral_d14_mistral_cnn_64_train/predictions.csv

python analyze_results.py --csv results/cape_coral_d21_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/cape_coral_d21_mistral_cnn_64_train/predictions.csv


# ============================
# JACKSONVILLE
# ============================
python analyze_results.py --csv results/jacksonville_d14_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/jacksonville_d14_mistral_cnn_64_train/predictions.csv

python analyze_results.py --csv results/jacksonville_d21_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/jacksonville_d21_mistral_cnn_64_train/predictions.csv


# ============================
# MIAMI
# ============================
python analyze_results.py --csv results/miami_d14_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/miami_d14_mistral_cnn_64_train/predictions.csv

python analyze_results.py --csv results/miami_d21_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/miami_d21_mistral_cnn_64_train/predictions.csv


# ============================
# ORLANDO
# ============================
python analyze_results.py --csv results/orlando_d14_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/orlando_d14_mistral_cnn_64_train/predictions.csv

python analyze_results.py --csv results/orlando_d21_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/orlando_d21_mistral_cnn_64_train/predictions.csv


# ============================
# TAMPA
# ============================
python analyze_results.py --csv results/tampa_d14_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/tampa_d14_mistral_cnn_64_train/predictions.csv

python analyze_results.py --csv results/tampa_d21_full_cnn_64_train/predictions.csv
python analyze_results.py --csv results/tampa_d21_mistral_cnn_64_train/predictions.csv



echo "Job finished: $(date)"
