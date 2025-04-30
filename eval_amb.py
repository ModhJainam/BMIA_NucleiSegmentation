import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from monai.data import CSVDataset, DataLoader
from model_amb import get_model_and_loss, get_transforms  # unified imports
from train_amb import set_seed
import SimpleITK as sitk

# -----------------------------------------------------------------------------
#  Post-processing: watershed-based instance separation
# -----------------------------------------------------------------------------

def postprocess_instance_segmentation(
    seg_prob: np.ndarray,
    conf_map: np.ndarray,
    th: float = 0.5,
    min_size: int = 40,
    h_min: float = 0.15,
    alpha: float = 0.3,
) -> np.ndarray:
    """Instance segmentation relying almost exclusively on SimpleITK.

    Pipeline
    --------
    1. Threshold & clean the nucleus probability map using morphological opening
       and size filtering.
    2. Build a fused surface that blends the Euclidean distance
       transform with the confidence map (inside the mask only).
    3. Apply H‑minima suppression to curb shallow minima, extract seeds via
       `RegionalMinima` and run `MorphologicalWatershedFromMarkers`.

    Parameters
    ----------
    seg_prob : np.ndarray
        Nucleus probability map (0‑1).
    conf_map : np.ndarray
        Ambiguity/confidence map (0‑1, higher = more confident).
    th : float, optional
        Threshold for the foreground mask. Default = 0.5.
    min_size : int, optional
        Minimum pixel area to keep a nucleus. Default = 40 px.
    h_min : float, optional
        H‑minima suppression height. Default = 0.15.
    alpha : float, optional
        Blend weight for confidence in fused surface (0 = dist only, 1 = conf
        only). Default = 0.3.

    Returns
    -------
    np.ndarray
        Integer‑labelled instance segmentation map.
    """

    # ---------------------------------------------------------------------
    # 1. Binary foreground mask & morphological opening (SITK)
    # ---------------------------------------------------------------------
    mask_arr = (seg_prob > th).astype(np.uint8)
    if mask_arr.sum() == 0:
        return np.zeros_like(mask_arr, dtype=np.int32)

    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk = sitk.BinaryMorphologicalOpening(mask_sitk, [1, 1], sitk.sitkBall)

    # ------------------------------------------------------------------
    # Remove small components (< min_size) using ConnectedComponent + stats
    # ------------------------------------------------------------------
    cc = sitk.ConnectedComponent(mask_sitk)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    keep = []
    for lbl in stats.GetLabels():
        if stats.GetNumberOfPixels(lbl) >= min_size:
            keep.append(lbl)

    cleaned = sitk.Image(cc.GetSize(), sitk.sitkUInt8)
    cleaned.CopyInformation(cc)
    for lbl in keep:
        cleaned = cleaned | sitk.Equal(cc, lbl)

    cleaned_arr = sitk.GetArrayFromImage(cleaned)
    if cleaned_arr.sum() == 0:
        return np.zeros_like(cleaned_arr, dtype=np.int32)

    # ------------------------------------------------------------------
    # 2. Distance transform (Maurer) + confidence fusion
    # ------------------------------------------------------------------
    dist_sitk = sitk.SignedMaurerDistanceMap(cleaned, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)
    dist_arr = sitk.GetArrayFromImage(dist_sitk)
    dist_arr = np.clip(dist_arr, 0, None)
    if dist_arr.max() > 0:
        dist_arr = dist_arr / dist_arr.max()

    conf_norm = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min() + 1e-8)
    fused = (1.0 - alpha) * dist_arr + alpha * conf_norm * cleaned_arr

    # ------------------------------------------------------------------
    # 3. H‑minima suppression, seed extraction, watershed (all SITK)
    # ------------------------------------------------------------------
    fused_sitk = sitk.GetImageFromArray(fused.astype(np.float32))
    fused_sitk = sitk.InvertIntensity(fused_sitk, maximum=1.0)
    fused_sitk = sitk.HMinima(fused_sitk, h_min)

    minima = sitk.RegionalMinima(fused_sitk, backgroundValue=0, foregroundValue=1, fullyConnected=True)
    markers = sitk.ConnectedComponent(minima)

    labels = sitk.MorphologicalWatershedFromMarkers(fused_sitk, markers, markWatershedLine=False, fullyConnected=True)
    labels = sitk.Mask(labels, cleaned)

    return sitk.GetArrayFromImage(labels).astype(np.int32)


# -----------------------------------------------------------------------------
#  Instance‑level metric (Dice + IoU)
# -----------------------------------------------------------------------------

def evaluate_instances(pred_label: np.ndarray, true_label: np.ndarray):
    pred_ids = [i for i in np.unique(pred_label) if i != 0]
    true_ids = [i for i in np.unique(true_label) if i != 0]
    if len(true_ids) == len(pred_ids) == 0:
        return 1.0, 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0, 0.0

    overlaps = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int32)
    unions = np.zeros_like(overlaps)
    for pi, p in enumerate(pred_ids):
        pm = pred_label == p
        for ti, t in enumerate(true_ids):
            tm = true_label == t
            inter = np.logical_and(pm, tm).sum()
            if inter:
                overlaps[pi, ti] = inter
                unions[pi, ti] = pm.sum() + tm.sum() - inter
    matched = set(); inter_tot = 0; union_tot = 0
    for ti in range(len(true_ids)):
        pi = np.argmax(overlaps[:, ti])
        if overlaps[pi, ti] and pi not in matched:
            matched.add(pi)
            inter_tot += overlaps[pi, ti]
            union_tot += unions[pi, ti]
    if union_tot == 0:
        return 0.0, 0.0
    iou = inter_tot / union_tot
    dice = 2 * inter_tot / (inter_tot + union_tot)
    return dice, iou


# -----------------------------------------------------------------------------
#  Visualisation helper (3 × 3 grid)
# -----------------------------------------------------------------------------

def _label2rgb(lbl: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random colour‑map for integer label images (0 treated as background)."""
    rng = np.random.RandomState(seed)
    rgb = np.zeros(lbl.shape + (3,), dtype=np.uint8)
    for uid in np.unique(lbl):
        if uid == 0:
            continue
        rgb[lbl == uid] = rng.randint(0, 255, 3, dtype=np.uint8)
    return rgb


def visualize(raw_img, gt_lbl, gt_amb, pre_img, pred_bin, pred_amb, inst_pred, path, title=""):
    """Create a 3 × 3 figure as requested in the prompt, with an overall title."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    ax = axes.ravel()

    # Row 1 -------------------------------------------------------------------
    ax[0].imshow(raw_img.astype(np.uint8));        ax[0].set_title("Raw Image");      ax[0].axis("off")
    ax[1].imshow(pre_img.astype(np.uint8));       ax[1].set_title("Pre-processed"); ax[3].axis("off")
    ax[2].axis("off")  # empty cell

    # Row 2 -------------------------------------------------------------------
    ax[3].imshow(_label2rgb(gt_lbl));             ax[3].set_title("GT Labels");     ax[1].axis("off")
    ax[4].imshow(pred_bin, cmap="gray");         ax[4].set_title("Pred Binary");   ax[4].axis("off")
    ax[5].imshow(_label2rgb(inst_pred));          ax[5].set_title("Pred Instances"); ax[7].axis("off")

    # Row 3 -------------------------------------------------------------------
    ax[6].imshow(gt_amb, cmap="inferno");        ax[6].set_title("GT Ambiguity");  ax[2].axis("off")
    ax[7].imshow(pred_amb, cmap="inferno");      ax[7].set_title("Pred Ambiguity"); ax[5].axis("off")
    ax[8].axis("off")  # empty cell

    # Figure title ------------------------------------------------------------
    if title:
        fig.suptitle(title, fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
    plt.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ambiguity-aware HoVerNet on test set and create 3×3 visualisations")
    parser.add_argument("--data_dir", default="processed_data")
    parser.add_argument("--ckpt", default="outputs/best_model.pth")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", default="outputs/eval")
    parser.add_argument("--vis", type=int, default=3, help="#samples to visualise")
    args = parser.parse_args()

    set_seed(42)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Data ----------------------------------------------------------------
    # ---------------------------------------------------------------------
    _, test_tfms = get_transforms()
    test_csv = os.path.join(args.data_dir, "splits", "test.csv")
    test_ds = CSVDataset(test_csv, transform=test_tfms)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=4)

    # ---------------------------------------------------------------------
    # Model ---------------------------------------------------------------
    # ---------------------------------------------------------------------
    model, _ = get_model_and_loss(pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device); model.eval()
    print(f"Loaded weights from {args.ckpt}")

    # ---------------------------------------------------------------------
    # Evaluation loop -----------------------------------------------------
    # ---------------------------------------------------------------------
    tot_dice = tot_iou = n = 0; vis_done = 0
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            imgs = batch["image_path"].to(device)              # pre-processed images (B × C × H × W)
            gt_inst = batch["label_path"].cpu().numpy().astype(np.int32)
            gt_amb  = batch["confidence_path"].cpu().numpy()
            outputs = model(imgs)
            seg_prob = torch.sigmoid(outputs["nucleus_pred"]).cpu().numpy()
            amb_prob = torch.sigmoid(outputs["amb_pred"]).cpu().numpy()

            for i in range(seg_prob.shape[0]):
                seg_map, amb_map = seg_prob[i, 0], amb_prob[i, 0]
                inst_pred = postprocess_instance_segmentation(seg_map, amb_map)
                d, j = evaluate_instances(inst_pred, gt_inst[i, 0])
                tot_dice += d; tot_iou += j; n += 1

                # ----------------------------------------------------
                # Visualisation ------------------------------------
                # ----------------------------------------------------
                if vis_done < args.vis:
                    gidx     = bidx * args.batch + i  # global index in dataset
                    raw_path = test_ds.data[gidx]["image_path"]  # original file path from CSV
                    raw_img  = np.array(Image.open(raw_path))

                    pre_img  = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    pred_bin = (seg_map > 0.5).astype(np.uint8)

                    # Figure title = file basename (organ + ID if encoded)
                    fname = os.path.splitext(os.path.basename(raw_path))[0]
                    fig_title = fname

                    fig_path = os.path.join(args.out, "figs", f"sample_{gidx}.png")
                    visualize(
                        raw_img=raw_img,
                        gt_lbl=gt_inst[i, 0],
                        gt_amb=gt_amb[i, 0],
                        pre_img=pre_img,
                        pred_bin=pred_bin,
                        pred_amb=amb_map,
                        inst_pred=inst_pred,
                        path=fig_path,
                        title=fig_title,
                    )
                    vis_done += 1

    # ---------------------------------------------------------------------
    # Summary -------------------------------------------------------------
    # ---------------------------------------------------------------------
    if n:
        print(f"Test images: {n}\nMean Instance Dice: {tot_dice/n:.4f}\nMean Instance IoU: {tot_iou/n:.4f}\nVisualisations saved: {vis_done}")
    else:
        print("No test images found.")
