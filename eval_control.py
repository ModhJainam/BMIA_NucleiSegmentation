import os
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk
from monai.data import CSVDataset, DataLoader
from monai.metrics import DiceMetric

from model_control import get_model_and_loss, get_transforms
from train_control import set_seed

# -----------------------------------------------------------------------------
#  Post‑processing: watershed (distance‑based only)
# -----------------------------------------------------------------------------

def postprocess_instance_segmentation(
    seg_prob: np.ndarray,
    th: float = 0.5,
    min_size: int = 10,
    h_min: float = 0.05,
) -> np.ndarray:
    """Convert nucleus probability map into an instance‑labelled map (SimpleITK).

    Steps
    -----
    1. Threshold → binary mask & morphological opening.
    2. Remove small objects < *min_size*.
    3. Distance transform (inside nucleus) → inverted surface.
    4. H‑minima suppression, regional minima as seeds.
    5. Watershed to get instances.
    """

    mask = (seg_prob > th).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32)

    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk = sitk.BinaryMorphologicalOpening(mask_sitk, [1, 1], sitk.sitkBall)

    # Remove tiny blobs
    cc = sitk.ConnectedComponent(mask_sitk)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    cleaned = sitk.Image(cc.GetSize(), sitk.sitkUInt8); cleaned.CopyInformation(cc)
    for lbl in stats.GetLabels():
        if stats.GetNumberOfPixels(lbl) >= min_size:
            cleaned = cleaned | sitk.Equal(cc, lbl)

    if sitk.GetArrayFromImage(cleaned).sum() == 0:
        return np.zeros_like(mask, dtype=np.int32)

    dist = sitk.SignedMaurerDistanceMap(
        cleaned, insideIsPositive=True, squaredDistance=False, useImageSpacing=False
    )
    dist = sitk.Clamp(dist, lowerBound=0.0, upperBound=float("inf"))

    inv = sitk.InvertIntensity(dist, maximum=sitk.GetArrayFromImage(dist).max() + 1e-6)
    inv = sitk.HMinima(inv, h_min)

    minima = sitk.RegionalMinima(inv, backgroundValue=0, foregroundValue=1, fullyConnected=True)
    markers = sitk.ConnectedComponent(minima)

    labels = sitk.MorphologicalWatershedFromMarkers(inv, markers, markWatershedLine=False, fullyConnected=True)
    labels = sitk.Mask(labels, cleaned)
    return sitk.GetArrayFromImage(labels).astype(np.int32)


# -----------------------------------------------------------------------------
#  Evaluate instances (IoU & Dice)
# -----------------------------------------------------------------------------

def evaluate_instances(pred: np.ndarray, gt: np.ndarray):
    pred_ids = [i for i in np.unique(pred) if i != 0]
    gt_ids = [i for i in np.unique(gt) if i != 0]
    if not pred_ids and not gt_ids:
        return 1.0, 1.0
    if not pred_ids or not gt_ids:
        return 0.0, 0.0

    overlaps = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int32)
    unions = np.zeros_like(overlaps)
    for pi, pid in enumerate(pred_ids):
        pmask = pred == pid
        for gi, gid in enumerate(gt_ids):
            gmask = gt == gid
            inter = np.logical_and(pmask, gmask).sum()
            if inter:
                overlaps[pi, gi] = inter
                unions[pi, gi] = pmask.sum() + gmask.sum() - inter
    matched = set(); inter_tot = 0; union_tot = 0
    for gi in range(len(gt_ids)):
        pi = np.argmax(overlaps[:, gi])
        if overlaps[pi, gi] and pi not in matched:
            matched.add(pi)
            inter_tot += overlaps[pi, gi]
            union_tot += unions[pi, gi]
    if union_tot == 0:
        return 0.0, 0.0
    iou = inter_tot / union_tot
    dice = 2 * inter_tot / (inter_tot + union_tot)
    return dice, iou

# -----------------------------------------------------------------------------
#  Visualisation helper (2 × 3 grid, no ambiguity)
# -----------------------------------------------------------------------------

def _label2rgb(lbl: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random colour mapping for integer labels (background=0)."""
    rng = np.random.RandomState(seed)
    rgb = np.zeros(lbl.shape + (3,), dtype=np.uint8)
    for uid in np.unique(lbl):
        if uid == 0:
            continue
        rgb[lbl == uid] = rng.randint(0, 255, 3, dtype=np.uint8)
    return rgb


def visualize(raw_img, gt_lbl, pre_img, pred_bin, inst_pred, path, title=""):
    """Save a 2 × 3 figure: raw | GT | pre  /  pred‑bin | instances | empty"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()

    ax[0].imshow(raw_img.astype(np.uint8));      ax[0].set_title("Raw Image"); ax[0].axis("off")
    ax[1].imshow(_label2rgb(gt_lbl));            ax[1].set_title("GT Labels"); ax[1].axis("off")
    ax[2].imshow(pre_img.astype(np.uint8));      ax[2].set_title("Pre‑processed"); ax[2].axis("off")

    ax[3].imshow(pred_bin, cmap="gray");        ax[3].set_title("Pred Binary"); ax[3].axis("off")
    ax[4].imshow(_label2rgb(inst_pred));         ax[4].set_title("Pred Instances"); ax[4].axis("off")
    ax[5].axis("off")

    if title:
        fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=200)
    plt.close(fig)

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet nucleus segmentation model")
    parser.add_argument("--data_dir", default="processed_data")
    parser.add_argument("--ckpt", default=None, help="Path to model weight .pth file (optional)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", default="outputs_control/eval")
    parser.add_argument("--vis", type=int, default=3, help="#samples to visualise")
    args = parser.parse_args()

    set_seed(42)

    out_dir = Path(args.out)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data ------------------------------------------------------------------
    _, val_tfms = get_transforms()
    csv_path = os.path.join(args.data_dir, "splits", "test.csv")
    val_ds = CSVDataset(csv_path, transform=val_tfms)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=4)

    # Model -----------------------------------------------------------------
    model, _ = get_model_and_loss(pretrained=True)
    if args.ckpt and os.path.isfile(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"Loaded weights from {args.ckpt}")
    else:
        print("[Info] Using pretrained UNet weights (no ckpt provided or file missing).")
    model.to(device); model.eval()

    # Metrics ---------------------------------------------------------------
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    inst_dice_tot = inst_iou_tot = 0.0
    n = 0; vis_done = 0

    with torch.no_grad():
        for bidx, batch in enumerate(val_loader):
            imgs = batch["image_path"].to(device)
            gt_inst = batch["label_path"].cpu().numpy().astype(np.int32)

            outputs = model(imgs)
            seg_prob = torch.sigmoid(outputs["nucleus_pred"]).cpu().numpy()

            # semantic Dice -------------------------------------------------
            preds_bin = (seg_prob > 0.5).astype(np.float32)
            dice_metric(y_pred=torch.from_numpy(preds_bin), y=torch.from_numpy((gt_inst > 0).astype(np.float32)))

            for i in range(seg_prob.shape[0]):
                inst_pred = postprocess_instance_segmentation(seg_prob[i, 0])
                d, j = evaluate_instances(inst_pred, gt_inst[i, 0])
                inst_dice_tot += d; inst_iou_tot += j; n += 1

                # ---------------- visualisation -----------------------
                if vis_done < args.vis:
                    gidx = bidx * args.batch + i
                    raw_path = val_ds.data[gidx]["image_path"]
                    raw_img = np.array(Image.open(raw_path))
                    pre_img = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    pred_bin = preds_bin[i, 0]

                    fig_title = Path(raw_path).stem
                    fig_path = out_dir / "figs" / f"sample_{gidx}.png"
                    visualize(raw_img, gt_inst[i, 0], pre_img, pred_bin, inst_pred, fig_path, fig_title)
                    vis_done += 1

    sem_dice = dice_metric.aggregate().item() if getattr(dice_metric, "count", 1) > 0 else 0.0
    dice_metric.reset()

    if n:
        print(
            f"Semantic Dice (pre‑postproc): {sem_dice:.4f}\nInstance Dice: {inst_dice_tot/n:.4f}\nInstance IoU: {inst_iou_tot/n:.4f}\nImages: {n}\nVisualisations saved: {vis_done}")
    else:
        print("No validation images found.")