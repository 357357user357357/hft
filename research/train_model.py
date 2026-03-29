#!/usr/bin/env python3
"""Train the math signal transformer model.

Usage:
    python3 train_model.py

Options:
    --model_size: tiny, small, medium, large
    --epochs: Number of training epochs
    --batch_size: Batch size
    --lr: Learning rate
    --data_path: Path to training_data.jsonl
    --output_path: Path to save best model

Example:
    python3 train_model.py --model_size small --epochs 50 --batch_size 32
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from math_signal_model import MathSignalTransformer, create_model


class MathSignalDataset(Dataset):
    """Dataset for math signal prediction."""

    def __init__(self, jsonl_path: Path, max_seq_len: int = 200):
        self.samples = []
        self.max_seq_len = max_seq_len

        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  Skipping invalid JSON on line {i+1}")
                    continue

        print(f"  Loaded {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        labels = sample["labels"]

        # Input: normalize prices to [0, 1]
        prices = sample["input_prices"][:self.max_seq_len]
        p_min = min(prices)
        p_max = max(prices)
        p_range = max(p_max - p_min, 1e-10)
        prices_norm = [(p - p_min) / p_range for p in prices]

        # Pad to max_seq_len
        prices_norm = prices_norm + [prices_norm[-1]] * (self.max_seq_len - len(prices_norm))

        # Convert to tensor
        prices_tensor = torch.tensor(prices_norm, dtype=torch.float32).unsqueeze(-1)

        # Build labels dict
        label_dict = {
            "prices": prices_tensor,
        }

        # Poincaré regime (3-class)
        regime_map = {"mean-reversion": 0, "trending": 1, "neutral": 2}
        label_dict["poincare_regime"] = torch.tensor(
            regime_map.get(labels["poincare"]["regime"], 2), dtype=torch.long
        )

        # Poincaré score (continuous)
        label_dict["poincare_score"] = torch.tensor(
            labels["poincare"]["score"], dtype=torch.float32
        ).unsqueeze(0)

        # Whitehead signal (2-class)
        whitehead_map = {"regime_change": 0, "same_regime": 1}
        label_dict["whitehead_signal"] = torch.tensor(
            whitehead_map.get(labels["whitehead"]["signal"], 1), dtype=torch.long
        )

        # Torsion ratio (continuous)
        label_dict["torsion_ratio"] = torch.tensor(
            labels["whitehead"]["torsion_ratio"], dtype=torch.float32
        ).unsqueeze(0)

        # Hecke L-value (complex = 2 real numbers)
        label_dict["hecke_l_real"] = torch.tensor(
            labels["hecke"]["l_value_real"], dtype=torch.float32
        ).unsqueeze(0)
        label_dict["hecke_l_imag"] = torch.tensor(
            labels["hecke"]["l_value_imag"], dtype=torch.float32
        ).unsqueeze(0)

        # Zeta signal (3-class)
        zeta_map = {"strong": 0, "moderate": 1, "weak": 2}
        label_dict["zeta_signal"] = torch.tensor(
            zeta_map.get(labels["hecke"]["zeta_signal"], 1), dtype=torch.long
        )

        # Future return (continuous)
        label_dict["future_return"] = torch.tensor(
            labels["future_return"], dtype=torch.float32
        ).unsqueeze(0)

        return label_dict


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    result = {}
    for key in batch[0].keys():
        if key == "prices":
            # Stack prices: [batch, seq_len, 1]
            result[key] = torch.stack([item[key] for item in batch])
        else:
            # Stack other labels
            result[key] = torch.stack([item[key] for item in batch])
    return result


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move to device
        prices = batch["prices"].to(device)

        # Forward pass
        outputs = model(prices)

        # Compute multi-task loss
        loss = 0.0
        loss_components = {}

        # === Classification losses (higher weight) ===

        # Poincaré regime
        loss_poincare_regime = ce_loss(outputs["poincare_regime"], batch["poincare_regime"].to(device))
        loss += 1.0 * loss_poincare_regime
        loss_components["poincare_regime"] = loss_poincare_regime.item()

        # Whitehead signal
        loss_whitehead = ce_loss(outputs["whitehead_signal"], batch["whitehead_signal"].to(device))
        loss += 1.0 * loss_whitehead
        loss_components["whitehead"] = loss_whitehead.item()

        # Zeta signal
        loss_zeta = ce_loss(outputs["zeta_signal"], batch["zeta_signal"].to(device))
        loss += 1.0 * loss_zeta
        loss_components["zeta"] = loss_zeta.item()

        # === Regression losses (lower weight) ===

        # Poincaré score
        loss_poincare_score = mse_loss(
            outputs["poincare_score"].squeeze(),
            batch["poincare_score"].squeeze().to(device)
        )
        loss += 0.1 * loss_poincare_score
        loss_components["poincare_score"] = loss_poincare_score.item()

        # Torsion ratio
        loss_torsion = mse_loss(
            outputs["torsion_ratio"].squeeze(),
            batch["torsion_ratio"].squeeze().to(device)
        )
        loss += 0.1 * loss_torsion
        loss_components["torsion"] = loss_torsion.item()

        # Hecke L-value (complex)
        hecke_pred = outputs["hecke_l_value"]
        hecke_true = torch.stack([
            batch["hecke_l_real"].squeeze(),
            batch["hecke_l_imag"].squeeze(),
        ], dim=-1).to(device)
        loss_hecke = mse_loss(hecke_pred, hecke_true)
        loss += 0.1 * loss_hecke
        loss_components["hecke"] = loss_hecke.item()

        # Future return (supervised learning)
        loss_return = mse_loss(
            outputs["future_return"].squeeze(),
            batch["future_return"].squeeze().to(device)
        )
        loss += 0.5 * loss_return
        loss_components["future_return"] = loss_return.item()

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | "
                  f"PReg: {loss_components.get('poincare_regime', 0):.3f} | "
                  f"WH: {loss_components.get('whitehead', 0):.3f} | "
                  f"Zeta: {loss_components.get('zeta', 0):.3f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Accuracy counters
    correct = {"poincare": 0, "whitehead": 0, "zeta": 0}
    total = 0

    # MSE counters
    mse = {"poincare_score": 0.0, "torsion": 0.0, "hecke": 0.0, "return": 0.0}

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    for batch in dataloader:
        prices = batch["prices"].to(device)
        outputs = model(prices)

        # Classification accuracy
        correct["poincare"] += (outputs["poincare_regime"].argmax(1) == batch["poincare_regime"].to(device)).sum().item()
        correct["whitehead"] += (outputs["whitehead_signal"].argmax(1) == batch["whitehead_signal"].to(device)).sum().item()
        correct["zeta"] += (outputs["zeta_signal"].argmax(1) == batch["zeta_signal"].to(device)).sum().item()
        total += prices.size(0)

        # Regression MSE
        mse["poincare_score"] += mse_loss(outputs["poincare_score"].squeeze(), batch["poincare_score"].squeeze().to(device)).item()
        mse["torsion"] += mse_loss(outputs["torsion_ratio"].squeeze(), batch["torsion_ratio"].squeeze().to(device)).item()
        mse["return"] += mse_loss(outputs["future_return"].squeeze(), batch["future_return"].squeeze().to(device)).item()

        hecke_pred = outputs["hecke_l_value"]
        hecke_true = torch.stack([
            batch["hecke_l_real"].squeeze(),
            batch["hecke_l_imag"].squeeze(),
        ], dim=-1).to(device)
        mse["hecke"] += mse_loss(hecke_pred, hecke_true).item()

        num_batches += 1

    return {
        "poincare_accuracy": correct["poincare"] / max(total, 1),
        "whitehead_accuracy": correct["whitehead"] / max(total, 1),
        "zeta_accuracy": correct["zeta"] / max(total, 1),
        "poincare_score_mse": mse["poincare_score"] / max(num_batches, 1),
        "torsion_mse": mse["torsion"] / max(num_batches, 1),
        "hecke_mse": mse["hecke"] / max(num_batches, 1),
        "future_return_mse": mse["return"] / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train math signal transformer")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--data_path", type=str, default="research/training_data.jsonl")
    parser.add_argument("--output_path", type=str, default="research/math_signal_model.pt")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Setup
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training Math Signal Transformer ===")
    print(f"Device: {device}")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print()

    # Check data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run 'python3 research/generate_training_data.py' first")
        return

    # Load dataset
    dataset = MathSignalDataset(data_path, max_seq_len=200)

    # Train/val split
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create model
    model = create_model(args.model_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {args.model_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10
    )

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        start_time = datetime.now()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        # Compute combined validation accuracy
        val_acc = (
            val_metrics["poincare_accuracy"] +
            val_metrics["whitehead_accuracy"] +
            val_metrics["zeta_accuracy"]
        ) / 3

        # Learning rate step
        scheduler.step()

        elapsed = (datetime.now() - start_time).total_seconds()

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val accuracy: {val_acc:.3f}")
        print(f"    Poincaré: {val_metrics['poincare_accuracy']:.3f}")
        print(f"    Whitehead: {val_metrics['whitehead_accuracy']:.3f}")
        print(f"    Zeta: {val_metrics['zeta_accuracy']:.3f}")
        print(f"  Val MSE:")
        print(f"    Poincaré score: {val_metrics['poincare_score_mse']:.4f}")
        print(f"    Torsion: {val_metrics['torsion_mse']:.4f}")
        print(f"    Hecke: {val_metrics['hecke_mse']:.4f}")
        print(f"    Return: {val_metrics['future_return_mse']:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, output_path)
            print(f"  → Saved best model (val_acc={val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement)")
                break

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {output_path}")

    # Load best model for final inference test
    print(f"\nLoading best model for inference test...")
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test inference
    print("\n=== Inference Test ===")
    test_prices = torch.randn(1, 200, 1).to(device) * 0.1 + 0.5
    with torch.no_grad():
        outputs = model(test_prices)

    regime_classes = ["mean-reversion", "trending", "neutral"]
    whitehead_classes = ["regime_change", "same_regime"]
    zeta_classes = ["strong", "moderate", "weak"]

    print(f"Poincaré regime: {regime_classes[outputs['poincare_regime'][0].argmax().item()]}")
    print(f"Poincaré score: {outputs['poincare_score'][0, 0].item():.3f}")
    print(f"Whitehead signal: {whitehead_classes[outputs['whitehead_signal'][0].argmax().item()]}")
    print(f"Torsion ratio: {outputs['torsion_ratio'][0, 0].item():.3f}")
    print(f"Zeta signal: {zeta_classes[outputs['zeta_signal'][0].argmax().item()]}")


if __name__ == "__main__":
    main()
