#!/usr/bin/env python3
"""
evaluate.py - Multimodal fusion version (matches train.py)
FIXED: Auxiliary modules stay in FP32 for numerical stability

UPDATED:
- Collects metadata from dataset (location_name, business_category, city, placekey)
- Collects raw temporal sequences (seq_raw)
- Writes them into the final CSV for deeper error analysis
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from dataset import VisitForecastDataset, collate_fn
from model_components import (
    build_temporal_encoder,
    H3Embedding,
    Projection,
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(2.0 * np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def rmsle(y_true, y_pred, eps=1e-8):
    y_true = np.maximum(np.array(y_true, dtype=float), 0.0)
    y_pred = np.maximum(np.array(y_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)))


def load_quantized_model(model_name: str, quantization: str, device: str):
    """Load model with optional quantization"""
    
    if quantization == "4bit":
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        return model, torch.float16
        
    elif quantization == "8bit":
        print("Loading model with 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        return model, torch.float16
        
    else:
        print("Loading model in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        return model, torch.float16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="Path to test JSONL")
    p.add_argument("--horizon", choices=["d14", "d21"], default="d14", help="Forecast horizon")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--output", required=True, help="Path to save predictions CSV")
    p.add_argument("--model-name", dest="model_name", default="meta-llama/Meta-Llama-3-8B", 
                   help="HuggingFace model name")
    p.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none",
                   help="Quantization mode")
    p.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to use")
    p.add_argument("--use-best", action="store_true",
                   help="Use best_components.pt instead of final components.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    ckpt_dir = Path(args.checkpoint)
    
    # Allow choosing between best and final checkpoint
    if args.use_best:
        comp_path = ckpt_dir / "best_components.pt"
        print(f"Using best validation checkpoint")
    else:
        comp_path = ckpt_dir / "components.pt"
        print(f"Using final checkpoint")
    
    adapter_dir = ckpt_dir / "lora_adapter"

    if not comp_path.exists():
        raise FileNotFoundError(f"{comp_path.name} not found in {ckpt_dir}")

    print("\n" + "="*70)
    print("EVALUATION CONFIGURATION (MULTIMODAL FUSION)")
    print("="*70)
    print(f"Test JSONL    : {args.test}")
    print(f"Horizon       : {args.horizon}")
    print(f"Checkpoint    : {comp_path}")
    print(f"Model         : {args.model_name}")
    print(f"Quantization  : {args.quantization}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Device        : {device}")
    print("="*70 + "\n")

    # Load test dataset
    print("Loading test dataset...")
    test_ds = VisitForecastDataset(args.test, horizon=args.horizon)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"  Test samples: {len(test_ds)}")

    # Load checkpoint metadata
    print("\nLoading checkpoint metadata...")
    comp = torch.load(comp_path, map_location=device)

    fusion_mode = comp.get("fusion_mode", "full")
    h3_used = comp.get("h3_used", False)
    temp_encoder_type = comp.get("temp_encoder_type", "gru")
    temp_hidden_dim = comp.get("temp_hidden_dim", 64)
    h3_embed_dim = comp.get("h3_embed_dim", 64)
    d_llm = comp.get("d_llm", 4096)

    print(f"  Architecture  : {fusion_mode}")
    print(f"  Temp Encoder  : {temp_encoder_type}")
    print(f"  H3 Used       : {h3_used}")

    # Decide if we need to load the LLM
    uses_llm = fusion_mode in ("full", "llm_only")

    # Define dtypes (matching train.py)
    aux_dtype = torch.float32  # Auxiliary modules always FP32
    
    if uses_llm:
        # Load LLM + tokenizer
        print("\nLoading language model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set max_length
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = 512

        # Load model and get its dtype
        base_model, model_dtype = load_quantized_model(args.model_name, args.quantization, device)

        # Load LoRA adapter
        if PEFT_AVAILABLE and adapter_dir.exists():
            print(f"  Loading LoRA adapter from {adapter_dir}...")
            llm = PeftModel.from_pretrained(base_model, adapter_dir)
        else:
            print("  No LoRA adapter found")
            llm = base_model

        llm.eval()
        print(f"  Model dtype: {model_dtype}")
    else:
        # For temp_only/temp_h3, we don't need the LLM
        print("\nArchitecture does not use LLM. Skipping LLM loading.")
        tokenizer = None
        llm = None
        model_dtype = torch.float32  # Default dtype for non-LLM models

    # Rebuild auxiliary modules (ALWAYS FP32)
    print("\nRebuilding auxiliary modules...")
    
    # Temporal encoder + projection (FP32)
    temp_encoder = build_temporal_encoder(
        mode=temp_encoder_type,
        input_dim=1,
        hidden_dim=temp_hidden_dim,
    ).to(device=device, dtype=aux_dtype)
    temp_encoder.load_state_dict(comp["temp_encoder"])
    temp_encoder.eval()

    proj_temp = Projection(
        input_dim=temp_encoder.out_dim,
        output_dim=d_llm,
    ).to(device=device, dtype=aux_dtype)
    proj_temp.load_state_dict(comp["proj_temp"])
    proj_temp.eval()

    # H3 embedding + projection (FP32)
    if h3_used and comp.get("h3_embed") is not None:
        h3_state = comp["h3_embed"]

        h3_weight_key = None
        for k in h3_state.keys():
            if "weight" in k:
                h3_weight_key = k
                break

        if h3_weight_key is None:
            raise KeyError(f"No weight key found in h3_embed")

        num_h3_cells = h3_state[h3_weight_key].shape[0]

        h3_embed = H3Embedding(
            num_h3_cells=num_h3_cells,
            embed_dim=h3_embed_dim,
        ).to(device=device, dtype=aux_dtype)
        h3_embed.load_state_dict(h3_state)
        h3_embed.eval()

        proj_h3 = Projection(
            input_dim=h3_embed_dim,
            output_dim=d_llm,
        ).to(device=device, dtype=aux_dtype)
        proj_h3.load_state_dict(comp["proj_h3"])
        proj_h3.eval()
        
        print(f"  H3 embedding: {num_h3_cells} cells")
    else:
        h3_embed = None
        proj_h3 = None
        print("  H3 embedding: disabled")

    # Regression head (FP32 - CRITICAL FOR STABILITY!)
    if "reg_head" not in comp:
        raise KeyError("reg_head not found in checkpoint")

    reg_head = nn.Linear(d_llm, 1).to(device=device, dtype=aux_dtype)
    reg_head.load_state_dict(comp["reg_head"])
    reg_head.eval()

    print(f"  LLM dtype: {model_dtype}")
    print(f"  Aux modules dtype: {aux_dtype}")

    # Evaluation loop
    print("\n" + "="*70)
    print("EVALUATING (MULTIMODAL FUSION)")
    print("="*70 + "\n")
    
    all_y_true = []
    all_y_pred_raw = []
    all_y_pred_int = []

    # NEW: metadata + raw sequence buffers
    all_location_names = []
    all_business_categories = []
    all_cities = []
    all_placekeys = []
    all_seq_raw = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            prompts = batch["prompts"]
            seqs = batch["sequences"].to(device).to(aux_dtype)  # FP32
            targets = batch["targets"].cpu().numpy()
            h3_ids = batch["h3_ids"].to(device)

            # NEW: metadata + raw sequence from batch
            location_names = batch["location_name"]
            business_categories = batch["business_category"]
            cities = batch["city"]
            placekeys = batch["placekey"]
            seq_raw_batch = batch["seq_raw"]  # list-of-lists (prev 13/20 values)

            B = seqs.size(0)

            # === MULTIMODAL FUSION: Merge embeddings BEFORE LLM ===
            
            # Get text embeddings (LLM dtype)
            if uses_llm and llm is not None:
                enc = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                
                text_embeds = llm.get_input_embeddings()(enc["input_ids"])  # [B, seq_len, d_llm] (model_dtype)
                attention_mask = enc["attention_mask"]
            else:
                text_embeds = None
                attention_mask = None

            # Get temporal embeddings (FP32, convert to model_dtype for LLM)
            if fusion_mode in ("full", "temp_only", "temp_h3"):
                temp_vec = temp_encoder(seqs)  # FP32
                temp_emb_aux = proj_temp(temp_vec)  # FP32, [B, d_llm]
            else:
                temp_emb_aux = None

            # Get H3 embeddings (FP32, convert to model_dtype for LLM)
            if h3_used and fusion_mode in ("full", "temp_h3") and h3_embed is not None:
                h3_vec = h3_embed(h3_ids)  # FP32
                h3_emb_aux = proj_h3(h3_vec)  # FP32, [B, d_llm]
            else:
                h3_emb_aux = None

            # === Forward through architecture ===
            if fusion_mode == "full":
                # Convert aux embeddings to model_dtype for LLM
                temp_emb = temp_emb_aux.unsqueeze(1).to(model_dtype)  # [B, 1, d_llm]
                h3_emb = h3_emb_aux.unsqueeze(1).to(model_dtype)      # [B, 1, d_llm]
                
                # Merge: [temp_token][h3_token][text_tokens...]
                combined_embeds = torch.cat([temp_emb, h3_emb, text_embeds], dim=1)
                combined_mask = torch.cat([
                    torch.ones(B, 2, device=device, dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)
                
                outputs = llm(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    output_hidden_states=True,
                )
                hidden_llm = outputs.hidden_states[-1][:, -1, :]  # model_dtype
                hidden_aux = hidden_llm.to(aux_dtype)  # Convert to FP32 for regression
                
            elif fusion_mode == "llm_only":
                outputs = llm(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_llm = outputs.hidden_states[-1][:, -1, :]
                hidden_aux = hidden_llm.to(aux_dtype)
                
            elif fusion_mode == "temp_only":
                hidden_aux = temp_emb_aux  # Already FP32
                
            elif fusion_mode == "temp_h3":
                hidden_aux = temp_emb_aux + h3_emb_aux  # FP32
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

            # Regression (in FP32 for numerical stability)
            preds_raw = reg_head(hidden_aux).squeeze(-1).cpu().numpy()
            preds_clipped = np.maximum(preds_raw, 0.0)
            preds_int = np.rint(preds_clipped).astype(int)

            all_y_true.extend(targets.tolist())
            all_y_pred_raw.extend(preds_raw.tolist())
            all_y_pred_int.extend(preds_int.tolist())

            # NEW: extend metadata + sequences
            all_location_names.extend(location_names)
            all_business_categories.extend(business_categories)
            all_cities.extend(cities)
            all_placekeys.extend(placekeys)
            all_seq_raw.extend(seq_raw_batch)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

    # Compute metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    metrics = {
        "MAE": mae(all_y_true, all_y_pred_int),
        "RMSE": rmse(all_y_true, all_y_pred_int),
        "R2": r2_score(all_y_true, all_y_pred_int),
        "sMAPE(%)": smape(all_y_true, all_y_pred_int),
        "RMSLE": rmsle(all_y_true, all_y_pred_int),
    }

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    # Save results
    y_true_arr = np.array(all_y_true)
    y_pred_int_arr = np.array(all_y_pred_int)
    y_pred_raw_arr = np.array(all_y_pred_raw)

    # Store prev values as stringified lists (safe for CSV)
    prev_values_col = [str(seq) for seq in all_seq_raw]

    df_out = pd.DataFrame({
        "y_true": y_true_arr,
        "y_pred_raw": y_pred_raw_arr,
        "y_pred_int": y_pred_int_arr,
        "abs_error": np.abs(y_true_arr - y_pred_int_arr),
        "location_name": all_location_names,
        "business_category": all_business_categories,
        "city": all_cities,
        "placekey": all_placekeys,
        "prev_values": prev_values_col,
    })
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPrediction Statistics:")
    print(f"  Ground truth range : [{y_true_arr.min():.0f}, {y_true_arr.max():.0f}]")
    print(f"  Predicted range    : [{y_pred_int_arr.min():.0f}, {y_pred_int_arr.max():.0f}]")
    print(f"  Mean abs error     : {df_out['abs_error'].mean():.2f}")
    print(f"  Median abs error   : {df_out['abs_error'].median():.2f}")
    print(f"  Within 10 visits   : {(df_out['abs_error'] <= 10).mean() * 100:.1f}%")

    print("\n" + "="*70)
    print(f"Predictions saved to: {out_path}")
    print(f"Metrics saved to:     {metrics_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
