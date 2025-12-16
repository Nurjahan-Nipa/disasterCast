#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_results_no_metrics.py

Post-hoc analysis for visit forecasting results WITHOUT recomputing global metrics.

Input: CSV produced by evaluate.py with columns such as:
    y_true, y_pred_raw, y_pred_int, abs_error,
    location_name, business_category, city, placekey,
    prev_values (stringified list of prev 13/20 counts)

This script:
  - Finds top 10 best and worst individual predictions
  - Computes error by business category (with a min-count threshold)
  - Adds percentage of samples per category
  - Saves summary CSVs and simple matplotlib visualizations
  - Uses prev_values to plot example time series for TOP 5 BEST & WORST cases
  - Plots with explicit dates on X-axis (Start: Sep 19)
  - Adds a vertical red dotted line for Landfall (Sep 28)

Usage:
    python analyze_results_no_metrics.py \
        --csv results/cape_coral_d14_full/predictions.csv \
        --out-dir analysis/cape_coral_d14_full
"""

import argparse
from pathlib import Path
import ast
import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        required=True,
        help="Path to predictions CSV produced by evaluate.py",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save analysis outputs (default: <csv_dir>/analysis)",
    )
    p.add_argument(
        "--min-cat-count",
        type=int,
        default=20,
        help="Minimum number of samples per category for category-level stats",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if args.out_dir is None:
        out_dir = csv_path.parent / "analysis"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LOADING PREDICTIONS: {csv_path}")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    required_cols = {
        "y_true",
        "y_pred_int",
        "abs_error",
        "location_name",
        "business_category",
        "city",
        "placekey",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Parse prev_values column (stringified list) if present
    if "prev_values" in df.columns:
        def _safe_parse(x):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        df["prev_values_parsed"] = df["prev_values"].apply(_safe_parse)
    else:
        df["prev_values_parsed"] = [[] for _ in range(len(df))]

    # ============================
    # Top 10 best / worst POIs
    # ============================

    print("\nTOP 10 BEST PREDICTIONS (LOWEST ABS ERROR)")
    print("-" * 70)
    best_10 = df.nsmallest(10, "abs_error").copy()
    print(
        best_10[
            [
                "city",
                "business_category",
                "location_name",
                "y_true",
                "y_pred_int",
                "abs_error",
            ]
        ]
    )

    print("\nTOP 10 WORST PREDICTIONS (HIGHEST ABS ERROR)")
    print("-" * 70)
    worst_10 = df.nlargest(10, "abs_error").copy()
    print(
        worst_10[
            [
                "city",
                "business_category",
                "location_name",
                "y_true",
                "y_pred_int",
                "abs_error",
            ]
        ]
    )

    best_path = out_dir / "top10_best_predictions.csv"
    worst_path = out_dir / "top10_worst_predictions.csv"
    best_10.to_csv(best_path, index=False)
    worst_10.to_csv(worst_path, index=False)
    print(f"\nSaved best 10 to : {best_path}")
    print(f"Saved worst 10 to: {worst_path}")

    # ============================
    # Error by business category (with percentages)
    # ============================

    print("\nERROR BY BUSINESS CATEGORY (WITH PERCENTAGES)")
    print("-" * 70)

    total_samples = len(df)

    cat_stats = (
        df.groupby("business_category")["abs_error"]
        .agg(["count", "mean", "median"])
        .sort_values("mean")
    )

    # Add percentage of total samples
    cat_stats["percent"] = (cat_stats["count"] / total_samples) * 100.0

    # Filter categories with enough samples
    cat_stats_filtered = cat_stats[cat_stats["count"] >= args.min_cat_count]

    # Sort for best and worst views
    best_20 = cat_stats_filtered.nsmallest(20, "mean")
    worst_20 = cat_stats_filtered.nlargest(20, "mean")

    print(
        "\nTOP 20 BEST CATEGORIES (lowest mean abs_error, "
        f"count >= {args.min_cat_count}):"
    )
    print(best_20.to_string(float_format="%.2f"))

    print(
        "\nTOP 20 WORST CATEGORIES (highest mean abs_error, "
        f"count >= {args.min_cat_count}):"
    )
    print(worst_20.to_string(float_format="%.2f"))

    best_cat_path = out_dir / "category_best20.csv"
    worst_cat_path = out_dir / "category_worst20.csv"
    full_cat_path = out_dir / "category_error_stats_full.csv"

    best_20.to_csv(best_cat_path)
    worst_20.to_csv(worst_cat_path)
    cat_stats_filtered.to_csv(full_cat_path)

    print(f"\nSaved best 20 categories to : {best_cat_path}")
    print(f"Saved worst 20 categories to: {worst_cat_path}")
    print(f"Full category stats saved to : {full_cat_path}")

    # ============================
    # Simple Visualizations
    # ============================

    print("\nGenerating plots...")

    # 1) Histogram of absolute errors
    plt.figure()
    plt.hist(df["abs_error"], bins=50)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Absolute Errors")
    hist_path = out_dir / "hist_abs_error.png"
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # 2) Scatter: y_true vs y_pred_int
    plt.figure()
    plt.scatter(df["y_true"], df["y_pred_int"], s=8, alpha=0.6)
    plt.xlabel("Ground Truth Visits")
    plt.ylabel("Predicted Visits")
    plt.title("True vs Predicted Visit Counts")

    # Diagonal reference line
    lo = min(df["y_true"].min(), df["y_pred_int"].min())
    hi = max(df["y_true"].max(), df["y_pred_int"].max())
    plt.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    scatter_path = out_dir / "scatter_true_vs_pred.png"
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # 3) Bar chart: mean abs_error by business category (top 10 frequent)
    cat_stats_full = (
        df.groupby("business_category")["abs_error"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )

    top_cats = cat_stats_full.head(10)

    plt.figure()
    plt.bar(range(len(top_cats)), top_cats["mean"])
    plt.xticks(range(len(top_cats)), top_cats.index, rotation=45, ha="right")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE by Business Category (Top 10 by sample count)")
    plt.tight_layout()
    bar_path = out_dir / "bar_mae_top_categories.png"
    plt.savefig(bar_path)
    plt.close()

    # ============================
    # Example time series plots for TOP 5 BEST & WORST
    # ============================

    if "prev_values_parsed" in df.columns:
        
        # --- Helper for plotting ---
        def plot_series_with_dates(row, seq, title_prefix, filename):
            if len(seq) == 0:
                return None
            
            plt.figure(figsize=(12, 6))
            
            # --- Date Generation ---
            # Start Date: Sep 19, 2022
            start_date = datetime.date(2022, 9, 19)
            
            # Generate date objects for the history + 1 target day
            dates = [start_date + datetime.timedelta(days=i) for i in range(len(seq) + 1)]
            
            # Split into history dates and target date
            history_dates = dates[:-1]
            target_date = dates[-1]

            # --- Plot History ---
            plt.plot(history_dates, seq, label="History", marker='o', linestyle='-', color='tab:blue', markersize=6)
            
            # --- Plot Landfall (Sep 28) ---
            landfall_date = datetime.date(2022, 9, 28)
            
            # Draw vertical dotted line for Landfall
            plt.axvline(x=landfall_date, color='red', linestyle=':', linewidth=2, alpha=0.8, label="Landfall Date (Sep 28)")

            # Check if landfall is within the history range for the specific dot
            if start_date <= landfall_date <= history_dates[-1]:
                idx = (landfall_date - start_date).days
                val = seq[idx]
                plt.plot([landfall_date], [val], marker='o', color='red', markersize=10, zorder=10)
                plt.text(landfall_date, val, " Landfall", verticalalignment='bottom', color='red', fontweight='bold')
            
            # --- Plot Target/Prediction ---
            # True value (Green Circle)
            plt.scatter(
                [target_date], [row['y_true']], 
                color='green', s=150, zorder=5, label='Ground Truth'
            )
            
            # Predicted value (Red X)
            plt.scatter(
                [target_date], [row['y_pred_int']], 
                color='red', marker='x', s=150, zorder=6, linewidth=3, label='Prediction'
            )

            # Draw a faint dotted line connecting last history to truth
            plt.plot([history_dates[-1], target_date], [seq[-1], row['y_true']], 'k:', alpha=0.3)

            # --- Formatting ---
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.gcf().autofmt_xdate()

            plt.ylabel("Visits")
            
            plt.title(
                f"{title_prefix}\n"
                f"Placekey: {row['placekey']}\n"
                f"{row['city']} - {row['business_category']} ({row['location_name']})\n"
                f"Target Date: {target_date.strftime('%b %d')} | y_true={row['y_true']}, y_pred={row['y_pred_int']}, abs_error={row['abs_error']}"
            )
            
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            out_p = out_dir / filename
            plt.tight_layout()
            plt.savefig(out_p)
            plt.close()
            return out_p

        print("\nGenerating Top 5 BEST Prediction Plots...")
        # Loop through top 5 best
        count_best = 0
        for idx, row in best_10.iterrows():
            if count_best >= 5:
                break
            seq = row["prev_values_parsed"]
            # Sanitize placekey for filename
            clean_pk = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(row['placekey']))
            fname = f"best_{count_best+1}_{clean_pk}.png"
            
            out_path = plot_series_with_dates(
                row, seq, 
                f"Best Prediction Rank #{count_best+1}", 
                fname
            )
            if out_path:
                print(f"  - Saved: {out_path}")
            count_best += 1

        print("\nGenerating Top 5 WORST Prediction Plots...")
        # Loop through top 5 worst
        count_worst = 0
        for idx, row in worst_10.iterrows():
            if count_worst >= 5:
                break
            seq = row["prev_values_parsed"]
            clean_pk = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(row['placekey']))
            fname = f"worst_{count_worst+1}_{clean_pk}.png"
            
            out_path = plot_series_with_dates(
                row, seq, 
                f"Worst Prediction Rank #{count_worst+1}", 
                fname
            )
            if out_path:
                print(f"  - Saved: {out_path}")
            count_worst += 1

    print("\nSaved summary plots to:")
    print(f"  - {hist_path}")
    print(f"  - {scatter_path}")
    print(f"  - {bar_path}")

    print("\nDONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()