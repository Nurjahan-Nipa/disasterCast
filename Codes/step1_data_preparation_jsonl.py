#!/usr/bin/env python3
# step1_data_prep_enhanced.py
# Prepare 13->14 or 20->21 daily windows from 21-day format
# FIXED: No data leakage - H3 stats will be computed in training script

import argparse, json, ast, sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# --- H3 and placekey imports ---
try:
    import h3
    import placekey as pk
except ImportError:
    print("[ERROR] Missing required libraries. Please run: pip install h3 placekey", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare Dx forecasting data from 21-day format (NO LEAKAGE)."
    )
    p.add_argument("--input", required=True, help="Raw CSV with 21-day visits_by_day.")
    p.add_argument("--city", required=True, help="City name to filter (e.g., 'Jacksonville', 'Cape Coral').")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument(
        "--horizon",
        choices=["14", "21"],
        default="14",
        help="Which day to predict: 14 (13->14) or 21 (20->21). Default: 14.",
    )
    p.add_argument(
        "--landfall-date",
        default="2022-09-28",
        help="Landfall date (YYYY-MM-DD). Default: 2022-09-28.",
    )
    p.add_argument(
        "--series-start",
        default="2022-09-19",
        help="Expected series start date (YYYY-MM-DD). Default: 2022-09-19.",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Read only first N input rows (speedy trial).")
    p.add_argument("--print-samples", type=int, default=3, help="Print first N prepared records to stdout.")
    return p.parse_args()


def norm_city(s: str) -> str:
    """Normalize city name for comparison."""
    return str(s).strip().upper().replace("  ", " ")


def process_naics(naics_code, top_cat, sub_cat) -> dict:
    """
    Extract hierarchical NAICS codes and create business description.
    """
    naics_code = pd.to_numeric(naics_code, errors="coerce")

    if pd.isna(naics_code) or naics_code == 0:
        naics_2 = None
        naics_4 = None
        naics_6 = None
    else:
        s = str(int(naics_code))
        naics_2 = s[:2] if len(s) >= 2 else None
        naics_4 = s[:4] if len(s) >= 4 else None
        naics_6 = s[:6] if len(s) == 6 else None

    # Create business description for semantic LLM
    if pd.notna(sub_cat) and str(sub_cat).strip():
        business_category = str(sub_cat).strip()
    elif pd.notna(top_cat) and str(top_cat).strip():
        business_category = str(top_cat).strip()
    else:
        business_category = "Unknown Business Type"

    return {
        "naics_2": naics_2,
        "naics_4": naics_4,
        "naics_6": naics_6,
        "business_category": business_category,
    }


def process_h3(placekey_str: str) -> dict:
    """Extract hierarchical H3 hexagons from placekey."""
    if pd.isna(placekey_str):
        return {"h3_6": None, "h3_7": None, "h3_8": None}
    try:
        h3_10_hex = pk.placekey_to_h3(placekey_str)
        return {
            "h3_6": h3.cell_to_parent(h3_10_hex, 6),
            "h3_7": h3.cell_to_parent(h3_10_hex, 7),
            "h3_8": h3.cell_to_parent(h3_10_hex, 8),
        }
    except Exception:
        return {"h3_6": None, "h3_7": None, "h3_8": None}


def parse_vbd(v) -> Optional[List[float]]:
    """Parse visits_by_day column (can be string or list)."""
    if isinstance(v, (list, tuple)):
        return list(map(float, v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        # Try JSON parsing
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, (list, tuple)):
                    return list(map(float, obj))
            except Exception:
                pass
        # Try comma-separated
        try:
            return [float(x) for x in s.split(",")]
        except Exception:
            return None
    return None


def add_calendar_features(record, target_date, landfall_date):
    """
    Add calendar and temporal context features to a record.
    These are deterministic and don't cause data leakage.

    Here we pass target_date explicitly so it works for both D14 and D21.
    """
    target_date = pd.to_datetime(target_date)

    # Day of week (0=Monday, 6=Sunday)
    record["dow_target"] = int(target_date.dayofweek)

    # Hurricane phase (clipped to -14 to +14)
    days_after = (target_date - landfall_date).days
    record["target_days_after_landfall"] = days_after
    record["hurricane_phase"] = max(-14, min(14, days_after))

    # Is weekend
    record["is_weekend"] = 1 if target_date.dayofweek >= 5 else 0

    return record


def main():
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        print(f"[ERROR] Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    landfall_date = pd.to_datetime(args.landfall_date, errors="coerce")
    if pd.isna(landfall_date):
        print(f"[ERROR] Invalid --landfall-date: {args.landfall_date}", file=sys.stderr)
        sys.exit(1)

    series_start = pd.to_datetime(args.series_start, errors="coerce")
    if pd.isna(series_start):
        print(f"[ERROR] Invalid --series-start: {args.series_start}", file=sys.stderr)
        sys.exit(1)

    # Horizon logic
    horizon_int = int(args.horizon)   # 14 or 21
    target_suffix = f"d{horizon_int}"  # "d14" or "d21"
    prev_len = horizon_int - 1        # 13 or 20

    print("=" * 70)
    print(f"ENHANCED DATA PREPARATION (21-day format) - NO LEAKAGE - D{horizon_int}")
    print("=" * 70)
    print("\n??  H3 neighborhood stats will be computed during training")
    print("    to prevent data leakage between train/val/test sets.\n")

    # Read data
    print(f"Reading data from {inp}...")
    df = pd.read_csv(inp, nrows=args.max_rows)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Parse visits_by_day
    print("\nParsing visits_by_day...")
    df["vbd_list"] = df["visits_by_day"].apply(parse_vbd)

    # Filter for rows with exactly 21 days
    df = df[df["vbd_list"].apply(lambda x: isinstance(x, list) and len(x) == 21)].copy()
    print(f"  Rows with 21-day visits_by_day: {len(df)}")

    # Parse date_range_start
    df["date_range_start"] = pd.to_datetime(df["date_range_start"], errors="coerce")

    # Filter by city
    target_city = norm_city(args.city)
    if "city" in df.columns:
        df["city_norm"] = df["city"].map(norm_city)
        df = df[df["city_norm"] == target_city].copy()
        print(f"  Filtered to {len(df)} POIs in {args.city}")
    else:
        print("[WARN] No 'city' column found; processing all rows.")

    if df.empty:
        print(f"[WARN] No rows for city='{args.city}'. Exiting.")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("")
        sys.exit(0)

    # Ensure optional metadata columns exist
    optional_cols = ["location_name", "top_category", "sub_category", "latitude", "longitude", "naics_code"]
    for c in optional_cols:
        if c not in df.columns:
            df[c] = np.nan
            print(f"[WARN] Optional column '{c}' not found. Will output 'null'.")

    # Prepare records
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES (NO AGGREGATION ACROSS POIS)")
    print("=" * 70)

    prepared = []
    stats = {
        "total_pois": len(df),
        "pois_processed": 0,
        "pois_skipped": 0,
    }

    window_start_idx = 0
    window_end_idx = horizon_int - 1  # 13 for D14, 20 for D21

    print(f"\n  Window: days {window_start_idx}-{window_end_idx} ({horizon_int} days total)")
    print(f"  Series start: {series_start.date()}")
    print(f"  Target date: {(series_start + pd.Timedelta(days=window_end_idx)).date()}")

    # Process each POI independently (no cross-POI aggregation)
    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"  Processing POI {idx+1}/{len(df)}...")

        placekey_val = row.get("placekey")
        if pd.isna(placekey_val):
            stats["pois_skipped"] += 1
            continue

        vbd_list = row.get("vbd_list")
        if not vbd_list or len(vbd_list) != 21:
            stats["pois_skipped"] += 1
            continue

        # Extract horizon days (indices 0 .. horizon-1)
        visits_h_days = vbd_list[window_start_idx : window_end_idx + 1]

        if len(visits_h_days) != horizon_int:
            stats["pois_skipped"] += 1
            continue

        # Apply log1p transformation
        visits_raw = np.array(visits_h_days)
        visits_log = np.log1p(visits_raw)

        prev_log = visits_log[:prev_len].tolist()
        y_log = float(visits_log[prev_len])

        prev_raw = visits_raw[:prev_len].tolist()
        y_raw = float(visits_raw[prev_len])

        # Extract NAICS and H3 features (NO aggregation)
        naics_features = process_naics(
            row.get("naics_code"),
            row.get("top_category"),
            row.get("sub_category"),
        )
        h3_features = process_h3(placekey_val)

        # Calculate target date
        target_date = series_start + pd.Timedelta(days=window_end_idx)

        if target_date < landfall_date:
            time_period = "before"
        elif target_date == landfall_date:
            time_period = "landfall"
        else:
            time_period = "after"

        # Key names depend on horizon
        prev_key = f"prev_{prev_len}_values"
        prev_key_raw = f"prev_{prev_len}_values_raw"
        y_key = f"y_true_{target_suffix}"
        y_key_raw = f"y_true_{target_suffix}_raw"
        target_date_key = f"target_date_{target_suffix}"

        # Create record with ONLY per-POI features
        record = {
            "placekey": placekey_val,
            "city": row.get("city", args.city),
            "location_name": row.get("location_name", np.nan),
            "business_category": naics_features["business_category"],
            # H3 cell IDs
            "h3_6": h3_features["h3_6"],
            "h3_7": h3_features["h3_7"],
            "h3_8": h3_features["h3_8"],
            # NAICS hierarchy
            "naics_code": row.get("naics_code", np.nan),
            "naics_2": naics_features["naics_2"],
            "naics_4": naics_features["naics_4"],
            "naics_6": naics_features["naics_6"],
            # Geographic coordinates
            "latitude": row.get("latitude", np.nan),
            "longitude": row.get("longitude", np.nan),
            # Temporal metadata
            "series_start_date": str(series_start.date()),
            "landfall_date": str(landfall_date.date()),
            target_date_key: str(target_date.date()),
            "actual_target_date": str(target_date.date()),
            "time_periods_used": time_period,
            # Target variable (log + raw, horizon-aware keys)
            prev_key: prev_log,
            y_key: y_log,
            prev_key_raw: prev_raw,
            y_key_raw: y_raw,
        }

        # Add calendar features (horizon-agnostic)
        record = add_calendar_features(record, target_date, landfall_date)

        prepared.append(record)
        stats["pois_processed"] += 1

    print(f"\n  Extracted {len(prepared)} records")

    # Write JSONL output
    print("\n" + "=" * 70)
    print("WRITING OUTPUT")
    print("=" * 70)

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for rec in prepared:
            # Clean NaN values for JSON
            rec_cleaned = {
                k: (v if isinstance(v, list) else (None if pd.isna(v) else v))
                for k, v in rec.items()
            }
            f.write(json.dumps(rec_cleaned, ensure_ascii=False) + "\n")

    # Print summary
    print("\n" + "=" * 70)
    print("PREPARATION SUMMARY")
    print("=" * 70)

    for k, v in stats.items():
        print(f"{k}: {v}")

    # Count features
    sample_rec = prepared[0] if prepared else {}
    print(f"\nFeatures per record: {len(sample_rec)}")
    print("  - Per-POI features only (no aggregations)")
    print("  - H3 cell IDs for embedding lookup")
    print("  - Calendar features (deterministic)")
    print("  - NO neighborhood statistics (will be computed in training)")

    print(f"\nOutput file: {outp}")
    print(f"Total records: {len(prepared)}")
    print("=" * 70)

    # Show sample records
    if prepared and args.print_samples > 0:
        print("\n" + "=" * 70)
        print("SAMPLE RECORDS")
        print("=" * 70)

        for i, rec in enumerate(prepared[: args.print_samples], 1):
            print(f"\n--- Sample {i} ---")
            print(f"Placekey: {rec['placekey']}")
            print(f"Location: {rec['location_name']}")
            print(f"City: {rec['city']}")
            print(f"Business Category: {rec['business_category']}")
            print(
                f"NAICS: {rec['naics_2']} / {rec['naics_4']} / {rec['naics_6']} "
                f"(code: {rec.get('naics_code', 'N/A')})"
            )
            print(f"H3: {rec['h3_6']} / {rec['h3_7']} / {rec['h3_8']}")
            print(f"Coordinates: ({rec.get('latitude', 'N/A')}, {rec.get('longitude', 'N/A')})")

            prev_key = f"prev_{prev_len}_values_raw"
            y_key_raw = f"y_true_{target_suffix}_raw"
            target_date_key = f"target_date_{target_suffix}"

            print("\nTime series:")
            print(f"  Series start: {rec['series_start_date']}")
            print(f"  Target date: {rec[target_date_key]}")
            print(f"  Days after landfall: {rec['target_days_after_landfall']}")
            print(f"  First 3 days (raw): {rec[prev_key][:3]}")
            print(f"  Target (raw): {rec[y_key_raw]}")

            print("\nCalendar features:")
            print(
                f"  Day of week: {rec.get('dow_target', 'N/A')} "
                f"({'Weekend' if rec.get('is_weekend') else 'Weekday'})"
            )
            print(f"  Hurricane phase: {rec.get('hurricane_phase', 'N/A')}")
            print("\n??  H3 neighborhood stats will be computed during training")


if __name__ == "__main__":
    main()
