#!/usr/bin/env python3
"""
Simple LLM-only prediction script for POI visit forecasting.

Three prompt templates for ablation study:
- Template A: Minimalist (temporal-only baseline)
- Template B: Hurricane-aware (temporal + event context)
- Template C: Full context (semantic + temporal + hurricane)

REPRODUCIBILITY:
- Deterministic generation (do_sample=False)
- Fixed seed for any random operations
- Output is always integer (visitor counts)
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import numpy as np


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"✓ Random seed set to {seed}")


# ============================================================================
# Dataset
# ============================================================================

class SimpleLLMDataset(Dataset):
    """Dataset with three prompt template options."""
    
    def __init__(self, jsonl_path, horizon="d14", city_filter=None, prompt_template="C"):
        self.records = []
        self.horizon = horizon
        self.prompt_template = prompt_template
        
        # Track unique cities for debugging
        cities_found = set()
        
        with open(jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                city = rec.get("city", "")
                cities_found.add(city)
                
                # Apply city filter if specified (normalize underscores and case)
                if city_filter is None:
                    self.records.append(rec)
                else:
                    # Normalize both for comparison: replace spaces/underscores, lowercase
                    city_normalized = city.lower().replace(" ", "_").replace("-", "_")
                    filter_normalized = city_filter.lower().replace(" ", "_").replace("-", "_")
                    
                    if city_normalized == filter_normalized:
                        self.records.append(rec)
        
        # Print debug info if filtering resulted in 0 samples
        if city_filter and len(self.records) == 0:
            print(f"\n⚠️  WARNING: City filter '{city_filter}' matched 0 records!")
            print(f"Cities found in file: {sorted(cities_found)[:10]}")
            if len(cities_found) > 10:
                print(f"  ... and {len(cities_found) - 10} more")
            print()
    
    def make_prompt_A(self, rec, seq, values):
        """Template A - Minimalist (Temporal-Only Baseline)"""
        prompt = (
            "You are an expert time-series forecaster.\n"
            "Given the historical daily visit counts, predict the next day's visit count.\n\n"
            "Return the answer ONLY in this format:\n"
            "PREDICTION: <number>\n\n"
            f"DailyVisits: {values}\n\n"
            "PREDICTION:"
        )
        return prompt
    
    def make_prompt_B(self, rec, seq, values, series_start, landfall, target_date):
        """Template B - Hurricane-Aware (Temporal + Event Context)"""
        location_name = rec.get("location_name", "Unknown")
        city = rec.get("city", "Unknown")
        
        prompt = (
            "You are an expert in disaster-aware mobility forecasting.\n"
            "Predict the visit count for the target date using the provided historical daily visits\n"
            "and the hurricane landfall time.\n\n"
            "Do NOT explain your reasoning.\n"
            "Return the answer ONLY in this format:\n"
            "PREDICTION: <number>\n\n"
            f"Location: {location_name}\n"
            f"City: {city}\n"
            f"SeriesStart: {series_start}\n"
            f"HurricaneLandfall: {landfall}\n"
            f"TargetDate: {target_date}\n"
            f"DailyVisits: {values}\n\n"
            "PREDICTION:"
        )
        return prompt
    
    def make_prompt_C(self, rec, seq, values, series_start, landfall, target_date):
        """Template C - Full Context (Semantic + Temporal + Hurricane)"""
        location_name = rec.get("location_name", "Unknown")
        business_category = rec.get("business_category", "Unknown")
        city = rec.get("city", "Unknown")
        
        prompt = (
            "You are an expert in human-mobility forecasting after natural disasters.\n"
            "Predict the visit count for the target date by combining:\n"
            "- the temporal trend,\n"
            "- the hurricane landfall disruption,\n"
            "- the semantics of the business/location.\n\n"
            "Return ONLY:\n"
            "PREDICTION: <number>\n\n"
            "=== CONTEXT ===\n"
            f"LocationName: {location_name}\n"
            f"BusinessCategory: {business_category}\n"
            f"City: {city}\n"
            f"SeriesStartDate: {series_start}\n"
            f"HurricaneLandfallDate: {landfall}\n"
            f"TargetDateToPredict: {target_date}\n"
            f"PastDailyVisits (ordered): {values}\n"
            "=== END ===\n\n"
            "PREDICTION:"
        )
        return prompt
    
    def make_prompt(self, rec):
        """Create prompt based on selected template."""
        
        if self.horizon == "d14":
            seq = rec["prev_13_values_raw"]
            series_start = "2022-09-19"
            landfall = "2022-09-28 (Day 10)"
            target_date = "2022-10-02 (Day 14)"
        else:
            seq = rec["prev_20_values_raw"]
            series_start = "2022-09-19"
            landfall = "2022-09-28 (Day 10)"
            target_date = "2022-10-09 (Day 21)"
        
        # Format as space-separated integers for cleaner parsing
        values = " ".join([str(int(v)) for v in seq])
        
        # Select template
        if self.prompt_template == "A":
            return self.make_prompt_A(rec, seq, values)
        elif self.prompt_template == "B":
            return self.make_prompt_B(rec, seq, values, series_start, landfall, target_date)
        else:  # "C"
            return self.make_prompt_C(rec, seq, values, series_start, landfall, target_date)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = self.make_prompt(rec)
        
        # Get target
        if self.horizon == "d14":
            target = rec["y_true_d14_raw"]
            seq = rec["prev_13_values_raw"]
        else:
            target = rec["y_true_d21_raw"]
            seq = rec["prev_20_values_raw"]
        
        return {
            "prompt": prompt,
            "target": float(target),
            "poi_id": rec.get("safegraph_place_id", f"poi_{idx}"),
            "sequence": seq
        }
    
    def __len__(self):
        return len(self.records)


def collate_fn(batch):
    """Simple collation."""
    return {
        "prompts": [b["prompt"] for b in batch],
        "targets": torch.tensor([b["target"] for b in batch], dtype=torch.float32),
        "poi_ids": [b["poi_id"] for b in batch],
        "sequences": [b["sequence"] for b in batch]
    }


# ============================================================================
# Prediction
# ============================================================================

def extract_number_from_response(response_text, seq_context=None):
    """
    Extract number from 'PREDICTION: <number>' format.
    Handles various edge cases and applies sanity bounds.
    """
    import re
    
    # Clean up response
    text = response_text.strip()
    
    # Strategy 1: Look for "PREDICTION: 123" pattern
    match = re.search(r'PREDICTION:\s*(\d+)', text, re.IGNORECASE)
    if match:
        pred = int(match.group(1))
    else:
        # Strategy 2: First number after colon
        match = re.search(r':\s*(\d+)', text)
        if match:
            pred = int(match.group(1))
        else:
            # Strategy 3: Any standalone number
            numbers = re.findall(r'\b(\d+)\b', text)
            if numbers:
                pred = int(numbers[0])  # First number
            else:
                # Fallback: 0
                pred = 0
    
    # Apply sanity bounds if context provided
    if seq_context is not None and len(seq_context) > 0:
        max_historical = max(seq_context)
        # Don't predict more than 5x the historical max or less than 0
        pred = max(0, min(pred, int(max_historical * 5)))
    
    return pred


def predict_batch(model, tokenizer, prompts, sequences, device, max_new_tokens=30):
    """
    Generate predictions for a batch of prompts.
    Returns list of integers (visitor counts).
    """
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to(device)
    
    # Generate (deterministic with do_sample=False)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic - always same output for same input
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    predictions = []
    for i, output in enumerate(outputs):
        # Get only the generated part (skip the prompt)
        prompt_length = inputs["input_ids"][i].shape[0]
        generated_ids = output[prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract number with context-based bounds (returns integer)
        pred_value = extract_number_from_response(response, sequences[i])
        predictions.append(pred_value)
    
    return predictions


def evaluate(predictions, targets):
    """Compute evaluation metrics."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Handle empty arrays
    if len(predictions) == 0 or len(targets) == 0:
        return {
            "MAE": float('nan'),
            "RMSE": float('nan'),
            "RMSLE": float('nan'),
            "R2": float('nan'),
            "sMAPE": float('nan')
        }
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    # RMSLE - add small epsilon to avoid log(0)
    epsilon = 1e-6
    rmsle = np.sqrt(mean_squared_error(
        np.log1p(targets + epsilon),
        np.log1p(predictions + epsilon)
    ))
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = np.abs(targets) + np.abs(predictions)
    # Avoid division by zero: only compute where denominator > 0
    mask = denominator > 0
    smape = np.mean(2 * np.abs(targets[mask] - predictions[mask]) / denominator[mask]) * 100 if mask.any() else float('inf')
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "RMSLE": rmsle,
        "R2": r2,
        "sMAPE": smape
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-only POI visit prediction with ablation templates")
    
    # Data
    parser.add_argument("--test", required=True, help="Path to test JSONL")
    parser.add_argument("--horizon", choices=["d14", "d21"], default="d14")
    parser.add_argument("--city", default=None, 
                       help="Filter by city (e.g., cape_coral, miami, or leave empty for all)")
    
    # Prompt template selection
    parser.add_argument("--template", choices=["A", "B", "C"], default="C",
                       help="Prompt template: A=Minimalist, B=Hurricane-aware, C=Full context")
    
    # Model
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B",
                       help="HuggingFace model name (e.g., meta-llama/Meta-Llama-3-8B, mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none",
                       help="Quantization mode for faster inference")
    parser.add_argument("--batch-size", type=int, default=4)
    
    # Output
    parser.add_argument("--output", default="llm_predictions.json",
                       help="Output file for predictions")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Debug
    parser.add_argument("--debug-samples", type=int, default=0,
                       help="Print first N sample outputs for debugging (0 = none)")
    
    # Device - NOW DEFAULTS TO CUDA
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # ========================================================================
    # GPU AVAILABILITY CHECK
    # ========================================================================
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("="*70)
            print("❌ ERROR: CUDA requested but not available!")
            print("="*70)
            print("Diagnostics:")
            print(f"  - PyTorch version: {torch.__version__}")
            print(f"  - CUDA available: {torch.cuda.is_available()}")
            print(f"  - CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
            print("\nTroubleshooting:")
            print("  1. Run 'nvidia-smi' to check GPU visibility")
            print("  2. Check CUDA_VISIBLE_DEVICES environment variable")
            print("  3. Verify PyTorch CUDA installation")
            print("\n⚠️  FALLING BACK TO CPU (will be 50-100x slower!)")
            print("="*70 + "\n")
            args.device = "cpu"
        else:
            # GPU is available - print info
            print("="*70)
            print("✓ GPU DETECTED")
            print("="*70)
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print("="*70 + "\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Template descriptions
    template_desc = {
        "A": "Minimalist (temporal-only baseline)",
        "B": "Hurricane-aware (temporal + event context)",
        "C": "Full context (semantic + temporal + hurricane)"
    }
    
    print("="*70)
    print("LLM-ONLY PREDICTION WITH ABLATION TEMPLATES")
    print("="*70)
    print(f"Test file: {args.test}")
    print(f"Horizon: {args.horizon}")
    print(f"City filter: {args.city or 'All cities'}")
    print(f"Prompt template: {args.template} - {template_desc[args.template]}")
    print(f"Model: {args.model_name}")
    print(f"Quantization: {args.quantization}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Generation: Deterministic (do_sample=False)")
    print(f"Output type: Integer (visitor counts)")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with quantization if specified
    if args.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        
        print("  Using 4-bit quantization (faster inference, less memory)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
    elif args.quantization == "8bit":
        from transformers import BitsAndBytesConfig
        
        print("  Using 8-bit quantization (faster inference)")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
    else:
        print("  Using full precision (no quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map="auto" if args.device == "cuda" else None
        )
        
        if args.device == "cpu":
            model = model.to(args.device)
    
    model.eval()
    print(f"✓ Model loaded\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = SimpleLLMDataset(
        args.test, 
        horizon=args.horizon, 
        city_filter=args.city,
        prompt_template=args.template
    )
    
    if len(dataset) == 0:
        print("❌ ERROR: No samples loaded!")
        if args.city:
            print(f"   Try removing --city filter or check city names in the JSONL file")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"✓ {len(dataset)} samples loaded")
    if args.city:
        print(f"  (filtered for city: {args.city})")
    print()
    
    # Debug: Show sample prompts if requested
    if args.debug_samples > 0:
        print("="*70)
        print(f"DEBUG: First {args.debug_samples} sample(s) with Template {args.template}")
        print("="*70)
        
        debug_batch = next(iter(dataloader))
        for i in range(min(args.debug_samples, len(debug_batch["prompts"]))):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt:\n{debug_batch['prompts'][i]}")
            print(f"\nActual target: {debug_batch['targets'][i].item():.0f}")
            
            # Generate prediction for this sample
            inputs = tokenizer([debug_batch['prompts'][i]], return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            
            prompt_length = inputs["input_ids"][0].shape[0]
            response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            extracted = extract_number_from_response(response, debug_batch['sequences'][i])
            
            print(f"Model response: '{response}'")
            print(f"Extracted prediction: {extracted}")
            print(f"Error: {abs(extracted - debug_batch['targets'][i].item()):.0f}")
        
        print("\n" + "="*70 + "\n")
    
    # Predict
    print("Generating predictions...\n")
    all_predictions = []
    all_targets = []
    all_poi_ids = []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        prompts = batch["prompts"]
        targets = batch["targets"]
        poi_ids = batch["poi_ids"]
        sequences = batch["sequences"]
        
        # Get predictions (returns integers)
        predictions = predict_batch(
            model, tokenizer, prompts, sequences,
            args.device, max_new_tokens=30
        )
        
        all_predictions.extend(predictions)
        all_targets.extend(targets.tolist())
        all_poi_ids.extend(poi_ids)
    
    # Evaluate
    print("\n" + "="*70)
    print(f"RESULTS - Template {args.template}: {template_desc[args.template]}")
    print("="*70)
    
    metrics = evaluate(all_predictions, all_targets)
    
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.4f}")
    
    print(f"\nPrediction statistics:")
    print(f"  Min prediction: {min(all_predictions)}")
    print(f"  Max prediction: {max(all_predictions)}")
    print(f"  Mean prediction: {sum(all_predictions)/len(all_predictions):.2f}")
    print(f"  Median prediction: {sorted(all_predictions)[len(all_predictions)//2]}")
    print(f"  All predictions are integers: {all(isinstance(p, int) for p in all_predictions)}")
    
    print(f"\nTarget statistics:")
    print(f"  Min target: {min(all_targets):.0f}")
    print(f"  Max target: {max(all_targets):.0f}")
    print(f"  Mean target: {sum(all_targets)/len(all_targets):.2f}")
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "horizon": args.horizon,
        "city": args.city or "all",
        "model": args.model_name,
        "quantization": args.quantization,
        "seed": args.seed,
        "prompt_template": args.template,
        "template_description": template_desc[args.template],
        "num_samples": len(all_predictions),
        "reproducible": True,
        "deterministic_generation": True,
        "metrics": metrics,
        "predictions": [
            {
                "poi_id": poi_id,
                "predicted": pred,  # Integer
                "actual": target
            }
            for poi_id, pred, target in zip(all_poi_ids, all_predictions, all_targets)
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()