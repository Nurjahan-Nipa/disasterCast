#!/usr/bin/env python3
"""
Standalone Neural Network Baselines for POI Visit Prediction

Completely separate from existing pipeline - trains simple temporal encoders:
- RNN: Simple Recurrent Neural Network
- LSTM: Long Short-Term Memory  
- GRU: Gated Recurrent Unit
- CNN: 1D Convolutional Neural Network
- GRU+CNN: Hybrid architecture

Usage:
    python nn_baselines.py --train train.jsonl --test test.jsonl --encoder GRU --horizon d14
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random


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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset
# ============================================================================

class SimpleTemporalDataset(Dataset):
    """Simple dataset - just sequences and targets."""
    
    def __init__(self, jsonl_path, horizon="d14", city_filter=None):
        self.records = []
        self.horizon = horizon
        
        with open(jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                
                # City filter
                if city_filter is not None:
                    city = rec.get("city", "").lower().replace(" ", "_")
                    if city != city_filter.lower().replace(" ", "_"):
                        continue
                
                self.records.append(rec)
        
        print(f"Loaded {len(self.records)} samples")
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        # Get sequence and target
        if self.horizon == "d14":
            seq = rec["prev_13_values_raw"]
            target = rec["y_true_d14_raw"]
        else:
            seq = rec["prev_20_values_raw"]
            target = rec["y_true_d21_raw"]
        
        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "target": torch.tensor([target], dtype=torch.float32),
            "poi_id": rec.get("safegraph_place_id", f"poi_{idx}")
        }
    
    def __len__(self):
        return len(self.records)


# ============================================================================
# Models
# ============================================================================

class RNNModel(nn.Module):
    """Simple RNN."""
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM."""
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class GRUModel(nn.Module):
    """GRU."""
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class CNNModel(nn.Module):
    """1D CNN."""
    def __init__(self, hidden_size=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1)


class GRUCNNModel(nn.Module):
    """GRU + CNN hybrid."""
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        # CNN feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        
        # GRU
        self.gru = nn.GRU(32, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # CNN
        x = x.unsqueeze(1)  # (batch, 1, seq)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq, channels)
        
        # GRU
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        targets = batch["target"].to(device).squeeze()
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions + metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_poi_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            targets = batch["target"].squeeze().numpy()
            poi_ids = batch["poi_id"]
            
            outputs = model(sequences)
            preds = outputs.cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            all_poi_ids.extend(poi_ids)
    
    # Compute metrics
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    # Clip negative predictions
    preds = np.maximum(preds, 0)
    
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    
    # RMSLE
    epsilon = 1e-6
    rmsle = np.sqrt(mean_squared_error(
        np.log1p(targets + epsilon),
        np.log1p(preds + epsilon)
    ))
    
    # sMAPE
    denominator = np.abs(targets) + np.abs(preds)
    mask = denominator > 0
    smape = np.mean(2 * np.abs(targets[mask] - preds[mask]) / denominator[mask]) * 100 if mask.any() else float('inf')
    
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "RMSLE": float(rmsle),
        "R2": float(r2),
        "sMAPE": float(smape)
    }
    
    predictions = [
        {"poi_id": pid, "predicted": float(p), "actual": float(t)}
        for pid, p, t in zip(all_poi_ids, preds, targets)
    ]
    
    return metrics, predictions


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone NN Baselines")
    
    # Data
    parser.add_argument("--train", required=True, help="Training JSONL")
    parser.add_argument("--val", required=True, help="Validation JSONL")
    parser.add_argument("--test", required=True, help="Test JSONL")
    parser.add_argument("--horizon", choices=["d14", "d21"], default="d14")
    parser.add_argument("--city", default=None, help="Filter by city")
    
    # Model
    parser.add_argument("--encoder", choices=["RNN", "LSTM", "GRU", "CNN", "GRU+CNN"], required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Output
    parser.add_argument("--output", default="baseline_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("="*70)
    print(f"NEURAL NETWORK BASELINE: {args.encoder}")
    print("="*70)
    print(f"Train: {args.train}")
    print(f"Val: {args.val}")
    print(f"Test: {args.test}")
    print(f"Horizon: {args.horizon}")
    print(f"City: {args.city or 'All'}")
    print(f"Device: {args.device}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    train_dataset = SimpleTemporalDataset(args.train, args.horizon, args.city)
    val_dataset = SimpleTemporalDataset(args.val, args.horizon, args.city)
    test_dataset = SimpleTemporalDataset(args.test, args.horizon, args.city)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating {args.encoder} model...")
    if args.encoder == "RNN":
        model = RNNModel(args.hidden_size, args.num_layers, args.dropout)
    elif args.encoder == "LSTM":
        model = LSTMModel(args.hidden_size, args.num_layers, args.dropout)
    elif args.encoder == "GRU":
        model = GRUModel(args.hidden_size, args.num_layers, args.dropout)
    elif args.encoder == "CNN":
        model = CNNModel(args.hidden_size, args.dropout)
    else:  # GRU+CNN
        model = GRUCNNModel(args.hidden_size, args.num_layers, args.dropout)
    
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequence"].to(args.device)
                targets = batch["target"].to(args.device).squeeze()
                outputs = model(sequences)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "/tmp/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load("/tmp/best_model.pt"))
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    metrics, predictions = evaluate(model, test_loader, args.device)
    
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "encoder": args.encoder,
        "horizon": args.horizon,
        "city": args.city or "all",
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "total_params": total_params,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset),
        "metrics": metrics,
        "predictions": predictions
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()