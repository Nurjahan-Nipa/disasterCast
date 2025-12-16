#!/usr/bin/env python3
"""
Per-POI Prophet Baseline for Hurricane Recovery Prediction

Usage:
    python prophet_baseline.py \
        --test-jsonl data/splits/cape_coral_d14_test.jsonl \
        --horizon d14 \
        --output-dir results/prophet_baseline/cape_coral_d14 \
        --city "Cape Coral"
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time
from datetime import datetime, timedelta

# Check Prophet availability
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("="*70)
    print("ERROR: Prophet not installed")
    print("="*70)
    print("\nInstall with:")
    print("  pip install prophet --break-system-packages")
    exit(1)


def load_test_data(test_jsonl_path, horizon):
    """
    Load test data from JSONL file.
    
    Args:
        test_jsonl_path: Path to test JSONL file
        horizon: "d14" or "d21"
    
    Returns:
        List of dicts with placekey, sequence, target
    """
    data = []
    
    # Determine expected sequence length and key names based on horizon
    if horizon == "d14":
        expected_length = 13
        sequence_key_options = ['prev_13_values_raw', 'sequence', 'sequences']
        target_key_options = ['y_true_d14_raw', 'y_true_d14', 'target', 'y_true', 'targets']
    elif horizon == "d21":
        expected_length = 20
        sequence_key_options = ['prev_20_values_raw', 'sequence', 'sequences']
        target_key_options = ['y_true_d21_raw', 'y_true_d21', 'target', 'y_true', 'targets']
    else:
        raise ValueError(f"Unknown horizon: {horizon}")
    
    print(f"Loading test data from {test_jsonl_path}...")
    
    with open(test_jsonl_path, 'r') as f:
        first_line = f.readline()
        first_record = json.loads(first_line)
        
        # Auto-detect sequence key name
        sequence_key = None
        for key in sequence_key_options:
            if key in first_record:
                sequence_key = key
                break
        
        if sequence_key is None:
            print(f"ERROR: Could not find sequence key in JSONL.")
            print(f"Available keys: {list(first_record.keys())}")
            print(f"Tried: {sequence_key_options}")
            raise KeyError(f"No sequence key found. Available keys: {list(first_record.keys())}")
        
        # Auto-detect target key name
        target_key = None
        for key in target_key_options:
            if key in first_record:
                target_key = key
                break
        
        if target_key is None:
            print(f"ERROR: Could not find target key in JSONL.")
            print(f"Available keys: {list(first_record.keys())}")
            print(f"Tried: {target_key_options}")
            raise KeyError(f"No target key found. Available keys: {list(first_record.keys())}")
        
        print(f"  Detected sequence key: '{sequence_key}'")
        print(f"  Detected target key: '{target_key}'")
        
        # Process first record
        placekey = first_record['placekey']
        target = first_record[target_key]
        sequence = first_record[sequence_key]
        
        if len(sequence) == expected_length:
            data.append({
                'placekey': placekey,
                'sequence': sequence,
                'target': target
            })
        else:
            print(f"⚠️  Warning: First POI {placekey} has sequence length {len(sequence)}, expected {expected_length}")
        
        # Process remaining records
        for line in f:
            record = json.loads(line)
            
            placekey = record['placekey']
            target = record[target_key]
            sequence = record[sequence_key]
            
            # Validate sequence length
            if len(sequence) != expected_length:
                print(f"⚠️  Warning: POI {placekey} has sequence length {len(sequence)}, expected {expected_length}")
                continue
            
            data.append({
                'placekey': placekey,
                'sequence': sequence,
                'target': target
            })
    
    print(f"  Loaded {len(data)} test samples")
    return data


def compute_metrics(y_true, y_pred):
    """Compute all evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # sMAPE
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_val = 100 * np.mean(numerator / (denominator + 1e-8))
    
    # RMSLE
    rmsle_val = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'sMAPE(%)': smape_val,
        'RMSLE': rmsle_val
    }


def predict_with_prophet(history, hurricane_date='2022-09-28'):
    """Fit Prophet and predict next value"""
    start_time = time.time()
    
    try:
        hurricane_dt = pd.to_datetime(hurricane_date)
        dates = [hurricane_dt + timedelta(days=i+1) for i in range(len(history))]
        
        df_prophet = pd.DataFrame({
            'ds': dates,
            'y': history
        })
        
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95,
            uncertainty_samples=0
        )
        
        # Suppress output
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        
        prediction = forecast['yhat'].iloc[-1]
        
        components = {
            'trend': float(forecast['trend'].iloc[-1]) if 'trend' in forecast else None,
            'yhat_lower': float(forecast['yhat_lower'].iloc[-1]) if 'yhat_lower' in forecast else None,
            'yhat_upper': float(forecast['yhat_upper'].iloc[-1]) if 'yhat_upper' in forecast else None
        }
        
        fit_time = time.time() - start_time
        return prediction, True, fit_time, components
        
    except Exception:
        prediction = np.mean(history)
        fit_time = time.time() - start_time
        return prediction, False, fit_time, None


def main():
    parser = argparse.ArgumentParser(description="Per-POI Prophet Baseline")
    parser.add_argument("--test-jsonl", required=True, help="Test JSONL file")
    parser.add_argument("--horizon", choices=["d14", "d21"], required=True, help="Forecast horizon")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--city", required=True, help="City name")
    parser.add_argument("--hurricane-date", type=str, default="2022-09-28", help="Hurricane landfall date")
    parser.add_argument("--max-pois", type=int, default=None, help="Limit number of POIs (for testing)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("PER-POI PROPHET BASELINE")
    print("="*70)
    print(f"City:     {args.city}")
    print(f"Horizon:  {args.horizon}")
    print(f"Hurricane Date: {args.hurricane_date}")
    print("="*70)
    print()
    
    # Load test data
    test_data = load_test_data(args.test_jsonl, args.horizon)
    
    if args.max_pois:
        test_data = test_data[:args.max_pois]
        print(f"\n⚠️  Limited to first {args.max_pois} POIs for testing")
    
    # Run Prophet
    print(f"\nProcessing {len(test_data)} test samples...")
    print("⚠️  Prophet is VERY slow (~5-10 sec per POI)")
    print(f"   Estimated time: {len(test_data) * 7 / 60:.1f} minutes")
    print()
    
    results = []
    prophet_predictions = []
    y_true_list = []
    prophet_success_count = 0
    prophet_fail_count = 0
    total_fit_time = 0
    
    start_time = time.time()
    
    for sample in tqdm(test_data, desc="Fitting Prophet models"):
        history = np.array(sample['sequence'])
        y_true = sample['target']
        y_true_list.append(y_true)
        
        # Prophet prediction
        prophet_pred_raw, success, fit_time, components = predict_with_prophet(
            history, hurricane_date=args.hurricane_date
        )
        
        # Round to integer and ensure non-negative (visit counts must be integers)
        prophet_pred = int(max(0, round(prophet_pred_raw)))
        prophet_predictions.append(prophet_pred)
        total_fit_time += fit_time
        
        if success:
            prophet_success_count += 1
        else:
            prophet_fail_count += 1
        
        results.append({
            'placekey': sample['placekey'],
            'y_true': y_true,
            'prophet_pred': prophet_pred,
            'prophet_success': success,
            'prophet_trend': components['trend'] if components else None,
            'fit_time_seconds': fit_time
        })
    
    elapsed_time = time.time() - start_time
    
    print()
    print("="*70)
    print("PROPHET FITTING COMPLETE")
    print("="*70)
    print(f"Success: {prophet_success_count}/{len(test_data)} ({100*prophet_success_count/len(test_data):.1f}%)")
    print(f"Failed:  {prophet_fail_count}/{len(test_data)} ({100*prophet_fail_count/len(test_data):.1f}%)")
    print(f"Time:    {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print()
    
    # Compute metrics
    print("="*70)
    print("METRICS")
    print("="*70)
    print()
    
    y_true = np.array(y_true_list)
    prophet_metrics = compute_metrics(y_true, np.array(prophet_predictions))
    
    print(f"{'Metric':<15} {'Value':>10}")
    print("-"*30)
    print(f"{'MAE':<15} {prophet_metrics['MAE']:>10.2f}")
    print(f"{'RMSE':<15} {prophet_metrics['RMSE']:>10.2f}")
    print(f"{'R²':<15} {prophet_metrics['R2']:>10.4f}")
    print(f"{'sMAPE(%)':<15} {prophet_metrics['sMAPE(%)']:>10.2f}")
    print(f"{'RMSLE':<15} {prophet_metrics['RMSLE']:>10.4f}")
    print()
    
    # Save results
    print("="*70)
    print("SAVING RESULTS")
    print("="*70)
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame(results)
    pred_file = output_dir / 'predictions.csv'
    results_df.to_csv(pred_file, index=False)
    print(f"✓ Predictions: {pred_file}")
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(prophet_metrics, f, indent=2)
    print(f"✓ Metrics:     {metrics_file}")
    
    # Save detailed summary
    summary = {
        'city': args.city,
        'horizon': args.horizon,
        'test_jsonl': args.test_jsonl,
        'test_samples': len(test_data),
        'prophet_success_count': prophet_success_count,
        'prophet_fail_count': prophet_fail_count,
        'prophet_success_rate': prophet_success_count / len(test_data),
        'total_time_seconds': elapsed_time,
        'avg_time_per_poi_seconds': elapsed_time / len(test_data),
        'metrics': {k: float(v) for k, v in prophet_metrics.items()}
    }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary:     {summary_file}")
    
    print()
    print("="*70)
    print("PROPHET BASELINE COMPLETE")
    print("="*70)
    print(f"Results: {output_dir}")
    print(f"Prophet MAE: {prophet_metrics['MAE']:.4f}")
    print(f"Prophet R²:  {prophet_metrics['R2']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()