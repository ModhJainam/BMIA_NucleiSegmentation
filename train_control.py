import os
import argparse
from typing import Tuple, Optional
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data import DataLoader, CSVDataset
from monai.metrics import DiceMetric
from monai.optimizers import WarmupCosineSchedule

from model_control import get_model_and_loss, get_transforms

# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

# -----------------------------------------------------------------------------
#  Epoch routine
# -----------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float]:
    """Return avg_total_loss, avg_seg_loss, mean Dice."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = seg_loss_sum = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for batch in loader:
        imgs = batch["image_path"].to(device)
        gt   = batch["nucleus_gt"].to(device)

        with torch.set_grad_enabled(train_mode):
            preds = model(imgs)
            loss_tensor, parts = criterion(preds, {"nucleus_gt": gt})
            if train_mode:
                loss_tensor.backward(); optimizer.step(); optimizer.zero_grad()

        total_loss += loss_tensor.item()
        seg_loss_sum += parts["seg"].item()

        with torch.no_grad():
            pred_bin = (torch.sigmoid(preds["nucleus_pred"]) > 0.5).float()
            dice_metric(y_pred=pred_bin, y=gt)

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_seg  = seg_loss_sum / n_batches
    dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return avg_loss, avg_seg, dice

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for nucleus segmentation")
    parser.add_argument("--data_dir", default="processed_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="outputs_control")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data -----------------------------------------------------------------
    train_tfms, val_tfms = get_transforms()
    train_ds = CSVDataset(os.path.join(args.data_dir, "splits", "train.csv"), transform=train_tfms)
    val_ds   = CSVDataset(os.path.join(args.data_dir, "splits", "test.csv"),  transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model / loss ---------------------------------------------------------
    model, criterion = get_model_and_loss(pretrained=True, freeze_encoder=True)
    model.to(device)

    # Optimizer & schedule --------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(0.1 * args.epochs * len(train_loader)),
        t_total=args.epochs * len(train_loader),
    )

    # Logs ------------------------------------------------------------------
    hist = {k: [] for k in ["train_loss", "val_loss", "train_seg", "val_seg", "train_dice", "val_dice"]}
    best_val = float("inf")

    # Training loop ---------------------------------------------------------
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_seg, tr_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss, val_seg, val_dice = run_epoch(model, val_loader, criterion, device)
        sched.step()

        for k, v in zip(hist.keys(), [tr_loss, val_loss, tr_seg, val_seg, tr_dice, val_dice]):
            hist[k].append(v)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        print(
            f"Epoch {ep}/{args.epochs} | "
            f"Train L:{tr_loss:.3f} S:{tr_seg:.3f} D:{tr_dice:.3f} | "
            f"Val L:{val_loss:.3f} S:{val_seg:.3f} D:{val_dice:.3f}")

    # Curves ---------------------------------------------------------------
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["train_loss"], label="Train")
    plt.plot(epochs, hist["val_loss"],   label="Val")
    plt.title("Total Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["train_dice"], label="Train Dice")
    plt.plot(epochs, hist["val_dice"],   label="Val Dice")
    plt.title("Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.grid(True)

    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
