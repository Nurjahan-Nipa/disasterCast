#!/usr/bin/env python3
"""
Train LLM-based POI visit forecasting model with temporal encoder + H3 + QLoRA.

MULTIMODAL FUSION VERSION - Merges embeddings BEFORE LLM fine-tuning with LoRA
OPTIMIZED: Skips LLM loading entirely for temp_only/temp_h3 architectures

Features:
- Deterministic training for reproducibility
- 4-bit/8-bit quantization support (QLoRA)
- Multimodal fusion: temporal + H3 + text embeddings
- End-to-end training with LoRA adapters
- Proper validation that matches evaluation logic
- Stable mixed-precision: aux modules in FP32, LLM in FP16
- Efficient: No LLM loading for temporal-only baselines
"""

import argparse
from pathlib import Path
import sys
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from dataset import VisitForecastDataset, collate_fn
from model_components import (
    build_temporal_encoder,
    H3Embedding,
    Projection,
)

# Optional: disable TorchDynamo/torch.compile globally (override from env if needed)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "0")

# LoRA + Quantization
try:
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    print("WARNING: 'peft' not installed. Quantization features disabled.")
    PEFT_AVAILABLE = False


# ============================================================================#
# Utilities
# ============================================================================#

def set_seed(seed: int, deterministic: bool = True):
    """Set random seed for maximum reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            print(f"✓ Random seed set to {seed} (deterministic mode enabled)")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            print(f"✓ Random seed set to {seed} (deterministic mode disabled, faster training)")
    else:
        print(f"✓ Random seed set to {seed}")


def load_quantized_model(model_name: str, quantization: str, device: str, debug: bool = False):
    """Load model with optional quantization. Fixed for cluster hangs."""
    if debug:
        print(">>> [DEBUG] Entering load_quantized_model()", flush=True)

    if quantization == "4bit":
        print("Loading model with 4-bit quantization (QLoRA)...")
        print("  - Using NF4 quantization type")
        print("  - Double quantization enabled")
        print("  - Compute dtype: float16")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Force single GPU (avoid accelerate auto-mapping issues)
        print("  - FORCE DEVICE: GPU 0 (Bypassing auto-map)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        model_dtype = torch.float16
        
    elif quantization == "8bit":
        print("Loading model with 8-bit quantization...")
        print("  - Int8 threshold: 6.0")
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        model_dtype = torch.float16
        
    else:
        raise ValueError(f"Invalid quantization: {quantization}. Use '4bit' or '8bit'.")

    if debug:
        print(">>> [DEBUG] Exiting load_quantized_model()", flush=True)
    
    return model, model_dtype


def build_lora_model(base_model, model_name: str, quantization: str, debug: bool = False):
    """Wrap the base_model with LoRA adapters."""
    if not PEFT_AVAILABLE:
        print("WARNING: PEFT not available. Using full model (not recommended).")
        return base_model

    if quantization in ("4bit", "8bit"):
        print("Preparing model for k-bit training...")
        base_model = prepare_model_for_kbit_training(base_model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if debug:
        print(">>> [DEBUG] Applying LoRA to base_model...", flush=True)
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    if debug:
        print(">>> [DEBUG] LoRA wrapping complete.", flush=True)
    return model


def get_adaptive_hyperparameters(arch_mode: str, quantization: str, base_lr: float):
    """Get adaptive hyperparameters based on architecture and quantization."""
    params = {
        'lr': base_lr,
        'lr_scale': 10.0,
        'recommended_epochs': 3,
    }
    
    if arch_mode == "full" and quantization in ("4bit", "8bit"):
        params['lr'] = base_lr * 0.5
        params['lr_scale'] = 5.0
        params['recommended_epochs'] = 5
        
        print("\n" + "="*70)
        print("ADAPTIVE HYPERPARAMETERS ACTIVATED")
        print("="*70)
        print(f"Detected: {arch_mode} architecture with {quantization} quantization")
        print(f"Adjustments made for stability:")
        print(f"  - LLM LR reduced: {base_lr:.6f} → {params['lr']:.6f}")
        print(f"  - LR scale reduced: 10.0 → {params['lr_scale']}")
        print(f"  - Recommended epochs: {params['recommended_epochs']}")
        print("="*70 + "\n")
    
    return params


def parse_args():
    p = argparse.ArgumentParser(description="Train LLM-based POI visit forecasting model")

    # Data / Task
    p.add_argument("--train", required=True, help="Path to TRAIN JSONL")
    p.add_argument("--val", default=None, help="Path to VAL JSONL (optional)")
    p.add_argument("--horizon", choices=["d14", "d21"], default="d14", help="Forecast horizon")

    # LLM Configuration (only needed for full/llm_only)
    p.add_argument("--model-name", dest="model_name", default="meta-llama/Meta-Llama-3-8B", 
                   help="HuggingFace model name (only for full/llm_only)")
    p.add_argument("--quantization", choices=["4bit", "8bit"], default="4bit",
                   help="Quantization mode (only for full/llm_only)")

    # Temporal Encoder Configuration
    p.add_argument("--temp-encoder", choices=["gru", "lstm", "cnn", "gru_cnn"], default="gru",
                   help="Type of temporal encoder")
    p.add_argument("--temp-hidden-dim", type=int, default=64, help="Hidden dimension of temporal encoder")

    # H3 Spatial Configuration
    p.add_argument("--h3-embed-dim", type=int, default=64, help="Embedding dimension for H3 cells")
    p.add_argument("--use-h3", action="store_true", help="Include H3 spatial embedding")

    # Architecture Ablations
    p.add_argument("--arch-mode", choices=["full", "llm_only", "temp_only", "temp_h3"], default="full",
                   help="Architecture ablation mode")

    # Training Hyperparameters
    p.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    p.add_argument("--lr-scale", type=float, default=None,
                   help="LR multiplier for non-LLM components (auto if not specified)")
    p.add_argument("--lambda-mse", type=float, default=10.0, help="Weight for MSE term")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Disable deterministic mode (faster but non-reproducible)")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")

    # I/O
    p.add_argument("--output-dir", default="results", help="Directory to save checkpoints")

    # Device
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device, help="Device to use")

    return p.parse_args()


def save_checkpoint(llm, temporal_encoder, h3_embed, proj_temp, proj_h3, reg_head,
                    optimizer, epoch, path, args, d_llm: int):
    """Save model checkpoint in format compatible with evaluate.py"""
    checkpoint = {
        'epoch': epoch,
        'temp_encoder': temporal_encoder.state_dict(),
        'proj_temp': proj_temp.state_dict(),
        'reg_head': reg_head.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'fusion_mode': args.arch_mode,
        'h3_used': args.use_h3,
        'temp_encoder_type': args.temp_encoder,
        'temp_hidden_dim': args.temp_hidden_dim,
        'h3_embed_dim': args.h3_embed_dim,
        'd_llm': d_llm,
        'quantization': args.quantization if hasattr(args, 'quantization') else 'none',
        'args': vars(args)
    }
    
    if h3_embed:
        checkpoint['h3_embed'] = h3_embed.state_dict()
    if proj_h3:
        checkpoint['proj_h3'] = proj_h3.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def validate(llm, temporal_encoder, h3_embed, proj_temp, proj_h3, reg_head,
             val_loader, tokenizer, args, device, model_dtype, aux_dtype):
    """
    MULTIMODAL VALIDATION - Merges embeddings before LLM.
    LLM stays in model_dtype (fp16), aux modules stay in aux_dtype (fp32).
    """
    temporal_encoder.eval()
    if h3_embed:
        h3_embed.eval()
    proj_temp.eval()
    if proj_h3:
        proj_h3.eval()
    reg_head.eval()
    if llm is not None and hasattr(llm, 'eval'):
        llm.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            prompts = batch["prompts"]
            seqs = batch["sequences"].to(device).to(aux_dtype)
            targets = batch["targets"].to(device).to(aux_dtype)
            h3_ids = batch["h3_ids"].to(device)
            B = targets.size(0)
            
            # === Text embeddings (LLM) ===
            if args.arch_mode in ("full", "llm_only"):
                enc = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                
                text_embeds = llm.get_input_embeddings()(enc["input_ids"])  # [B, seq_len, d_llm] (fp16)
                attention_mask = enc["attention_mask"]
            else:
                text_embeds = None
                attention_mask = None
            
            # === Temporal embeddings (aux_dtype) ===
            if args.arch_mode in ("full", "temp_only", "temp_h3"):
                temp_vec = temporal_encoder(seqs)            # FP32
                temp_emb_aux = proj_temp(temp_vec)           # FP32
            else:
                temp_emb_aux = None
            
            # === H3 embeddings (aux_dtype) ===
            if args.use_h3 and args.arch_mode in ("full", "temp_h3") and h3_embed:
                h3_vec = h3_embed(h3_ids)                    # FP32
                h3_emb_aux = proj_h3(h3_vec)                 # FP32
            else:
                h3_emb_aux = None
            
            # === Merge + forward through LLM (if used) ===
            if args.arch_mode == "full":
                # Cast prefixes to model_dtype for LLM
                prefix_list = [temp_emb_aux]
                if h3_emb_aux is not None:
                    prefix_list.append(h3_emb_aux)
                prefix_embeds_aux = torch.stack(prefix_list, dim=1) if len(prefix_list) == 2 else torch.cat(prefix_list, dim=1)
                # Above is messy; simpler:
                prefix_embeds_aux = torch.cat(
                    [temp_emb_aux.unsqueeze(1)] +
                    ([h3_emb_aux.unsqueeze(1)] if h3_emb_aux is not None else []),
                    dim=1
                )  # [B, n_prefix, d_llm] FP32

                prefix_embeds = prefix_embeds_aux.to(model_dtype)
                n_prefix = prefix_embeds.size(1)

                combined_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
                combined_mask = torch.cat([
                    torch.ones(B, n_prefix, device=device, dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)
                
                outputs = llm(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    output_hidden_states=True,
                )
                hidden_llm = outputs.hidden_states[-1][:, -1, :]  # fp16
                hidden_aux = hidden_llm.to(aux_dtype)
                
            elif args.arch_mode == "llm_only":
                outputs = llm(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_llm = outputs.hidden_states[-1][:, -1, :]
                hidden_aux = hidden_llm.to(aux_dtype)
                
            elif args.arch_mode == "temp_only":
                hidden_aux = temp_emb_aux  # [B, d_llm] FP32
                
            elif args.arch_mode == "temp_h3":
                if h3_emb_aux is None:
                    raise ValueError("arch_mode='temp_h3' requires --use-h3")
                hidden_aux = temp_emb_aux + h3_emb_aux  # FP32
            
            # === Regression in FP32 ===
            raw_reg = reg_head(hidden_aux).squeeze(-1)           # FP32
            loss = F.mse_loss(raw_reg, targets)                  # FP32
            
            total_loss += loss.item()
            num_batches += 1
    
    # Restore training mode
    temporal_encoder.train()
    if h3_embed:
        h3_embed.train()
    proj_temp.train()
    if proj_h3:
        proj_h3.train()
    reg_head.train()
    if llm is not None and hasattr(llm, 'train'):
        llm.train()
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


# ============================================================================#
# Main Training
# ============================================================================#

def main():
    args = parse_args()

    if args.debug:
        print(">>> [DEBUG] Arguments:", args, flush=True)
    
    # Safety: enforce consistency between arch_mode and use_h3
    if args.arch_mode in ("full", "temp_h3") and not args.use_h3:
        raise ValueError(f"arch_mode='{args.arch_mode}' requires --use-h3. "
                         f"Use 'llm_only' or 'temp_only' for ablations without H3.")

    # Quantization check
    uses_quantization = (
        args.arch_mode in ("full", "llm_only") and 
        hasattr(args, 'quantization') and 
        args.quantization in ("4bit", "8bit")
    )
    
    # Force non-deterministic mode for quantized models
    if uses_quantization and not args.no_deterministic:
        print("\n" + "="*70)
        print("AUTO-DISABLING DETERMINISTIC MODE")
        print("="*70)
        print("Reason: 4-bit/8-bit quantization uses non-deterministic operations")
        print("Setting --no-deterministic automatically for stability")
        print("="*70 + "\n")
        args.no_deterministic = True
    
    # Seed
    set_seed(args.seed, deterministic=not args.no_deterministic)
    
    uses_llm = args.arch_mode in ("full", "llm_only")
    
    # Adaptive hyperparameters only for LLM-based architectures
    if uses_llm:
        adaptive_params = get_adaptive_hyperparameters(args.arch_mode, args.quantization, args.lr)
        effective_lr = adaptive_params['lr']
    else:
        adaptive_params = {
            'lr': args.lr,
            'lr_scale': args.lr_scale if args.lr_scale else 1.0,
            'recommended_epochs': 3,
        }
        effective_lr = args.lr
    
    if args.lr_scale is None:
        args.lr_scale = adaptive_params['lr_scale']
    
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dtypes
    aux_dtype = torch.float32      # temporal/H3/regression
    model_dtype = torch.float16    # LLM (if used)

    # Print configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Train JSONL       : {args.train}")
    print(f"Val JSONL         : {args.val or 'None'}")
    print(f"Horizon           : {args.horizon}")
    print(f"Architecture      : {args.arch_mode}")
    print(f"Temporal Encoder  : {args.temp_encoder}")
    print(f"Use H3            : {args.use_h3}")
    
    if uses_llm:
        print(f"Model             : {args.model_name}")
        print(f"Quantization      : {args.quantization}")
    else:
        print(f"Model             : None (temporal-only)")
        print(f"Quantization      : N/A")
    
    print(f"Batch Size        : {args.batch_size}")
    print(f"Epochs            : {args.epochs}")
    print(f"LR (Base)         : {effective_lr}")
    print(f"LR Scale          : {args.lr_scale}")
    print(f"Lambda MSE        : {args.lambda_mse}")
    print(f"Device            : {device}")
    print(f"Deterministic     : {not args.no_deterministic}")
    print(f"Debug Mode        : {args.debug}")
    
    if uses_llm and args.epochs < adaptive_params['recommended_epochs']:
        print(f"\n⚠️  WARNING: Recommended epochs: {adaptive_params['recommended_epochs']}")
        print(f"   You're using: {args.epochs} epochs (may not converge)")
    
    print("="*70 + "\n")

    # Load Dataset
    print("Loading datasets...")
    train_ds = VisitForecastDataset(args.train, horizon=args.horizon)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = None
    if args.val:
        val_ds = VisitForecastDataset(args.val, horizon=args.horizon)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")
    else:
        print(f"  Train samples: {len(train_ds)}")

    num_h3_cells = train_ds.num_h3_cells
    print(f"  H3 vocabulary size: {num_h3_cells}")

    if args.debug:
        print(">>> [DEBUG] Data loaders ready.", flush=True)

    # Estimate mean target
    print("\nEstimating target statistics...")
    sample_targets = []
    for i, batch in enumerate(train_loader):
        sample_targets.extend(batch['targets'].tolist())
        if i >= 20:
            break
    mean_target = sum(sample_targets) / len(sample_targets) if sample_targets else 10.0
    print(f"  Mean target: {mean_target:.2f}")
    print(f"  Target range: [{min(sample_targets):.1f}, {max(sample_targets):.1f}]")

    # Load LLM (only if needed)
    if uses_llm:
        print("\nLoading language model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = 512

        if args.debug:
            print(">>> [DEBUG] About to load quantized LLM...", flush=True)
        base_model, model_dtype = load_quantized_model(args.model_name, args.quantization, device, debug=args.debug)
        if args.debug:
            print(">>> [DEBUG] Finished load_quantized_model()", flush=True)

        d_llm = base_model.config.hidden_size

        if args.debug:
            print(">>> [DEBUG] Wrapping base_model with LoRA...", flush=True)
        llm = build_lora_model(base_model, args.model_name, args.quantization, debug=args.debug)
        if args.debug:
            print(">>> [DEBUG] Finished build_lora_model()", flush=True)

        print(f"  LLM hidden dimension: {d_llm}")
        print(f"  Model dtype (LLM)    : {model_dtype}")
    else:
        print("\n✓ Architecture does not use LLM. Skipping LLM loading.")
        tokenizer = None
        llm = None
        d_llm = args.temp_hidden_dim  # Use temporal encoder output dim
        model_dtype = aux_dtype
        print(f"  Using d_llm={d_llm} (temporal encoder output dimension)")
        print(f"  Model dtype: {model_dtype} (FP32 for temporal-only)")

    # Build Auxiliary Modules (FP32)
    print("\nBuilding auxiliary modules...")
    
    temporal_encoder = build_temporal_encoder(
        mode=args.temp_encoder,
        input_dim=1,
        hidden_dim=args.temp_hidden_dim,
    ).to(device=device, dtype=aux_dtype)
    print(f"  Temporal encoder: {args.temp_encoder} (out_dim={temporal_encoder.out_dim})")

    proj_temp = Projection(
        input_dim=temporal_encoder.out_dim,
        output_dim=d_llm,
    ).to(device=device, dtype=aux_dtype)

    if args.use_h3:
        h3_embed = H3Embedding(
            num_h3_cells=num_h3_cells,
            embed_dim=args.h3_embed_dim,
        ).to(device=device, dtype=aux_dtype)
        
        proj_h3 = Projection(
            input_dim=args.h3_embed_dim,
            output_dim=d_llm,
        ).to(device=device, dtype=aux_dtype)
        print(f"  H3 embedding: {num_h3_cells} cells -> {args.h3_embed_dim}D")
    else:
        h3_embed = None
        proj_h3 = None
        print("  H3 embedding: disabled")

    reg_head = nn.Linear(d_llm, 1).to(device=device, dtype=aux_dtype)
    
    with torch.no_grad():
        reg_head.bias.fill_(mean_target)
    print(f"  Regression head: initialized bias to {mean_target:.2f}")
    print(f"  Aux modules dtype: {aux_dtype}")

    mse_loss_fn = nn.MSELoss()

    # Setup Optimizer
    print("\nSetting up optimizer...")
    params_llm = []
    params_other = []

    if uses_llm and llm is not None:
        params_llm = list(llm.parameters())

    if args.arch_mode in ("full", "temp_only", "temp_h3"):
        params_other += list(temporal_encoder.parameters())
        params_other += list(proj_temp.parameters())

    if args.use_h3 and args.arch_mode in ("full", "temp_h3"):
        params_other += list(h3_embed.parameters())
        params_other += list(proj_h3.parameters())

    params_other += list(reg_head.parameters())

    optimizer_groups = []
    if params_llm:
        optimizer_groups.append({'params': params_llm, 'lr': effective_lr})
        print(f"  LLM parameters: {sum(p.numel() for p in params_llm):,} @ lr={effective_lr}")
    if params_other:
        optimizer_groups.append({'params': params_other, 'lr': effective_lr * args.lr_scale})
        print(f"  Other parameters: {sum(p.numel() for p in params_other):,} @ lr={effective_lr * args.lr_scale}")
    
    if not optimizer_groups:
        raise ValueError("No parameters to optimize!")
    
    optimizer = torch.optim.AdamW(optimizer_groups)

    if args.debug:
        print(">>> [DEBUG] Optimizer constructed. Entering training loop...", flush=True)

    # Train mode
    temporal_encoder.train()
    if h3_embed:
        h3_embed.train()
    proj_temp.train()
    if proj_h3:
        proj_h3.train()
    reg_head.train()
    if uses_llm and llm is not None:
        llm.train()

    # Training Loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        if args.debug:
            print(f">>> [DEBUG] Starting epoch {epoch+1}/{args.epochs}", flush=True)

        total_loss = 0.0
        total_ce = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            prompts = batch["prompts"]
            seqs = batch["sequences"].to(device).to(aux_dtype)
            targets = batch["targets"].to(device).to(aux_dtype)
            h3_ids = batch["h3_ids"].to(device)
            B = targets.size(0)

            optimizer.zero_grad(set_to_none=True)
            ce_loss = torch.tensor(0.0, device=device, dtype=aux_dtype)

            # === MULTIMODAL FUSION: Merge embeddings BEFORE LLM ===
            if uses_llm:
                # For CE loss, tokenize FULL text (prompt + target)
                full_texts = []
                prefix_lengths = []
                for i in range(B):
                    prompt = prompts[i]
                    target_str = str(int(targets[i].item()))
                    prompt_text = prompt.strip()
                    if not prompt_text.rstrip().endswith("Prediction:") and \
                       not prompt_text.rstrip().endswith("Prediction"):
                        prompt_text = prompt_text.rstrip() + "\nPrediction:"
                    full = prompt_text + " " + target_str
                    full_texts.append(full)
                    prefix_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    prefix_lengths.append(len(prefix_ids))
                
                enc_full = tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                full_text_embeds = llm.get_input_embeddings()(enc_full["input_ids"])  # fp16
                
                labels = enc_full["input_ids"].clone()
                max_len = labels.size(1)
                for i in range(B):
                    L = min(prefix_lengths[i], max_len)
                    labels[i, :L] = -100
                
                enc = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                text_embeds = llm.get_input_embeddings()(enc["input_ids"])  # fp16
                attention_mask = enc["attention_mask"]
            else:
                text_embeds = None
                attention_mask = None
                full_text_embeds = None
            
            # Temporal (FP32)
            if args.arch_mode in ("full", "temp_only", "temp_h3"):
                temp_vec = temporal_encoder(seqs)
                temp_emb_aux = proj_temp(temp_vec)         # [B, d_llm] FP32
            else:
                temp_emb_aux = None
            
            # H3 (FP32)
            if args.use_h3 and args.arch_mode in ("full", "temp_h3"):
                h3_vec = h3_embed(h3_ids)
                h3_emb_aux = proj_h3(h3_vec)               # [B, d_llm] FP32
            else:
                h3_emb_aux = None
            
            # === Forward ===
            if args.arch_mode == "full":
                prefix_embeds_aux = torch.cat(
                    [temp_emb_aux.unsqueeze(1)] +
                    ([h3_emb_aux.unsqueeze(1)] if h3_emb_aux is not None else []),
                    dim=1
                )  # [B, n_prefix, d_llm] FP32
                prefix_embeds = prefix_embeds_aux.to(model_dtype)
                n_prefix = prefix_embeds.size(1)

                combined_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
                combined_mask = torch.cat([
                    torch.ones(B, n_prefix, device=device, dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)

                outputs = llm(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    output_hidden_states=True,
                )
                hidden_llm = outputs.hidden_states[-1][:, -1, :]
                hidden_aux = hidden_llm.to(aux_dtype)

                # CE path
                combined_embeds_full = torch.cat([prefix_embeds, full_text_embeds], dim=1)
                combined_mask_full = torch.cat([
                    torch.ones(B, n_prefix, device=device, dtype=enc_full["attention_mask"].dtype),
                    enc_full["attention_mask"]
                ], dim=1)
                labels_padded = torch.cat([
                    torch.full((B, n_prefix), -100, device=device, dtype=labels.dtype),
                    labels
                ], dim=1)

                outputs_ce = llm(
                    inputs_embeds=combined_embeds_full,
                    attention_mask=combined_mask_full,
                    labels=labels_padded,
                )
                ce_loss = outputs_ce.loss.to(aux_dtype)
                
            elif args.arch_mode == "llm_only":
                outputs = llm(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_llm = outputs.hidden_states[-1][:, -1, :]
                hidden_aux = hidden_llm.to(aux_dtype)

                outputs_ce = llm(
                    inputs_embeds=full_text_embeds,
                    attention_mask=enc_full["attention_mask"],
                    labels=labels,
                )
                ce_loss = outputs_ce.loss.to(aux_dtype)
                
            elif args.arch_mode == "temp_only":
                hidden_aux = temp_emb_aux        # FP32
                
            elif args.arch_mode == "temp_h3":
                if h3_emb_aux is None:
                    raise ValueError("arch_mode='temp_h3' requires --use-h3")
                hidden_aux = temp_emb_aux + h3_emb_aux  # FP32
            
            # Regression (FP32)
            raw_reg = reg_head(hidden_aux).squeeze(-1)
            mse_loss = mse_loss_fn(raw_reg, targets)

            # Total Loss
            if uses_llm:
                loss = ce_loss + args.lambda_mse * mse_loss
            else:
                loss = args.lambda_mse * mse_loss

            # NaN/inf guard
            if not torch.isfinite(loss):
                print(f">>> [DEBUG] Non-finite loss at epoch {epoch+1}, batch {batch_idx}", flush=True)
                print(f"    mse_loss={mse_loss.item()}, ce_loss={float(ce_loss)}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward
            loss.backward()
            
            # Gradient clipping
            if params_llm:
                torch.nn.utils.clip_grad_norm_(params_llm, max_norm=1.0)
            if params_other:
                torch.nn.utils.clip_grad_norm_(params_other, max_norm=1.0)

            optimizer.step()

            total_loss += float(loss)
            total_ce += float(ce_loss) if uses_llm else 0.0
            total_mse += float(mse_loss)
            num_batches += 1

            if args.debug and batch_idx % 20 == 0:
                print(f">>> [DEBUG] Epoch {epoch+1}, batch {batch_idx}/{len(train_loader)} "
                      f"loss={float(loss):.4f}", flush=True)
            
        # Epoch Summary
        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches if uses_llm else 0.0
        avg_mse = total_mse / num_batches

        if uses_llm:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, MSE: {avg_mse:.4f})", end="")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f})", end="")
        
        # Validation
        if val_loader:
            if args.debug:
                print("\n>>> [DEBUG] Running validation...", flush=True)
            val_loss = validate(
                llm, temporal_encoder, h3_embed, proj_temp, proj_h3, reg_head,
                val_loader, tokenizer, args, device, model_dtype, aux_dtype
            )
            print(f" | Val: {val_loss:.4f}", end="")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    llm, temporal_encoder, h3_embed, proj_temp, proj_h3, reg_head,
                    optimizer, epoch, output_dir / "best_components.pt", args, d_llm
                )
                print(" ✓ BEST", end="")
        
        print()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    # Save Final Model
    save_checkpoint(
        llm, temporal_encoder, h3_embed, proj_temp, proj_h3, reg_head,
        optimizer, args.epochs, output_dir / "components.pt", args, d_llm
    )
    
    # Save LoRA adapter (only if LLM was used)
    if PEFT_AVAILABLE and uses_llm and isinstance(llm, PeftModel):
        adapter_dir = output_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        llm.save_pretrained(adapter_dir)
        print(f"\n✓ Saved LoRA adapter to: {adapter_dir}")
    
    print(f"✓ Saved final checkpoint to: {output_dir / 'components.pt'}")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()