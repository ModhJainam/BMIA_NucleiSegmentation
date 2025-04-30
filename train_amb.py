import os
import argparse
from typing import Tuple
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data import DataLoader, CSVDataset
from monai.metrics import DiceMetric
from monai.optimizers import WarmupCosineSchedule

from model_amb import get_model_and_loss, get_transforms

# ----------------------------------------------------------------------------
#  Training / validation routine  
# ----------------------------------------------------------------------------

def run_epoch(model, loader, criterion, device, train: bool = True) -> Tuple[float, dict]:
    phase = "train" if train else "val"
    model.train() if train else model.eval()

    total_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    comp = {"seg": 0.0, "amb": 0.0}

    for batch in loader:
        imgs = batch["image_path"].to(device)
        nuc_gt = batch["nucleus_gt"].to(device)
        amb_gt = batch["amb_gt"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(imgs)
            loss, parts = criterion(outputs, {"nucleus_gt": nuc_gt, "amb_gt": amb_gt})
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        comp["seg"] += parts["seg"].item()
        comp["amb"] += parts["amb"].item()

        with torch.no_grad():
            preds_bin = (torch.sigmoid(outputs["nucleus_pred"]) > 0.5).float()
            dice_metric(y_pred=preds_bin, y=nuc_gt)

    avg_loss = total_loss / len(loader)
    avg_seg = comp["seg"] / len(loader)
    avg_amb = comp["amb"] / len(loader)
    dice = dice_metric.aggregate().item()
    return avg_loss, avg_seg, avg_amb, dice

def set_seed(seed):
    """
    Set random seed for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


# ----------------------------------------------------------------------------
#  Main entry‑point  
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ambiguity‑Aware UNet")
    parser.add_argument("--data_dir", default="processed_data", help="Path to processed dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_tfms, val_tfms = get_transforms()
    train_csv = os.path.join(args.data_dir, "splits", "train.csv")
    val_csv = os.path.join(args.data_dir, "splits", "test.csv")
    train_ds = CSVDataset(train_csv, transform=train_tfms)
    val_ds = CSVDataset(val_csv, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model / criterion
    model, criterion = get_model_and_loss(pretrained=True, freeze_encoder=True)
    model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = WarmupCosineSchedule(optimizer, warmup_steps=int(0.1 * args.epochs * len(train_loader)), t_total=args.epochs * len(train_loader))

    # Logs
    history = {
        "train_loss": [], "val_loss": [],
        "train_seg": [], "val_seg": [],
        "train_amb": [], "val_amb": [],
        "train_dice": [], "val_dice": []
    }
    best_val = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_seg, tr_amb, tr_dice = run_epoch(model, train_loader, criterion, device, train=True)
        with torch.no_grad():
            val_loss, val_seg, val_amb, val_dice = run_epoch(model, val_loader, criterion, device, train=False)
        sched.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_seg"].append(tr_seg)
        history["val_seg"].append(val_seg)
        history["train_amb"].append(tr_amb)
        history["val_amb"].append(val_amb)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(val_dice)
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train L:{tr_loss:.3f} S:{tr_seg:.3f} A:{tr_amb:.3f} D:{tr_dice:.3f} | "
            f"Val L:{val_loss:.3f} S:{val_seg:.3f} A:{val_amb:.3f} D:{val_dice:.3f}"
        )

    # --- plot curves ---
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(14, 8))
    # --- Total loss ---
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title("Total Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    # --- Segmentation loss ---
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_seg"], label="Train Seg")
    plt.plot(epochs, history["val_seg"], label="Val Seg")
    plt.title("Segmentation Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    # --- Ambiguity loss ---
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_amb"], label="Train Amb")
    plt.plot(epochs, history["val_amb"], label="Val Amb")
    plt.title("Ambiguity Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    # --- Dice ---
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.title("Dice Score"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))