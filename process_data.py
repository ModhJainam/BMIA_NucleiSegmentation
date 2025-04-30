import os
import shutil
import glob
import logging
import csv
from typing import Optional

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def _find_ambiguity_dir(organ_path: str) -> Optional[str]:
    """Return the directory that contains the ambiguity ("vague") masks.

    NuInSeg packages ambiguity maps in a folder whose name can vary between
    "mask binary" or "vague areas/mask binary" depending on extraction.
    This helper tries a small set of heuristics to locate it.
    """
    candidate_subdirs = [
        os.path.join("vague areas", "mask binary"),
        "vague areas/mask binary",  # in case OS kept the slash‑in‑name
    ]

    for sub in candidate_subdirs:
        d = os.path.join(organ_path, sub)
        if os.path.isdir(d):
            return d
    return None


def _relative(path: str, to_dir: str) -> str:
    """Return *POSIX*‑style relative path so downstream CSV is OS‑agnostic."""
    return os.path.relpath(path, start=to_dir).replace("\\", "/")


# ----------------------------------------------------------------------------
# Main processing routine
# ----------------------------------------------------------------------------

def process_dataset(input_path: str, output_path: str) -> None:
    """Process the NuInSeg dataset into a MONAI‑friendly folder layout.

    Expected NuInSeg folder structure per *organ* sub‑directory::

        ├── tissue images          (PNG RGB patches)
        ├── label masks modify    (TIFF – unique integer ID per nucleus)
        └── mask binary           (PNG binary maps of *ambiguous* regions)

    The script copies / converts patches into::

        processed_data/
            images/           *.png   (RGB input)
            labels/           *.tif   (instance label – **int16**, values kept)
            confidence_maps/  *.tif   (binary ambiguity mask, float32 0/1)
            metadata.csv      image_path,label_path,confidence_path

    Notes
    -----
    * The **label mask is kept *as‑is*** (unique IDs preserved). No binarisation
      so instance information survives for later post‑processing.
    * Ambiguity PNGs are normalised to {0,1} and saved as float32 TIFF so they
      load cleanly via MONAI's ITKReader.
    * Any organ lacking one of the required sub‑folders is skipped with a warn.
    """
    logger.info(f"Processing dataset from {input_path} → {output_path}")

    # ---------------------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------------------
    dirs = ["images", "labels", "confidence_maps"]
    for d in dirs:
        os.makedirs(os.path.join(output_path, d), exist_ok=True)

    # ---------------------------------------------------------------------
    # Prepare metadata CSV
    # ---------------------------------------------------------------------
    metadata_path = os.path.join(output_path, "metadata.csv")
    with open(metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "label_path", "confidence_path"])

    # ---------------------------------------------------------------------
    # Iterate over each organ folder in the original dataset
    # ---------------------------------------------------------------------
    for organ in tqdm(os.listdir(input_path), desc="Organs"):
        organ_path = os.path.join(input_path, organ)
        if not os.path.isdir(organ_path):
            continue

        tissue_dir = os.path.join(organ_path, "tissue images")
        label_dir = os.path.join(organ_path, "label masks modify")
        ambiguity_dir = _find_ambiguity_dir(organ_path)

        if not all(
            os.path.isdir(p) for p in [tissue_dir, label_dir] if p is not None
        ) or ambiguity_dir is None:
            logger.warning(
                f"Skipping '{organ}' – required sub‑folders missing (tissue, label, ambiguity)."
            )
            continue

        image_files = glob.glob(os.path.join(tissue_dir, "*.png"))
        base_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

        for base in tqdm(base_names, desc=f"  ↳ {organ}"):
            src_paths = {
                "image": os.path.join(tissue_dir, f"{base}.png"),
                "label": os.path.join(label_dir, f"{base}.tif"),
                "amb": os.path.join(ambiguity_dir, f"{base}.png"),
            }

            if not all(os.path.exists(p) for p in src_paths.values()):
                logger.error(f"    Missing one or more files for sample '{base}', skipping.")
                continue

            # ----------------------------------------------------------------
            # 1. Copy RGB image as‑is (.png)
            # ----------------------------------------------------------------
            dst_img_path = os.path.join(output_path, "images", f"{base}.png")
            shutil.copy2(src_paths["image"], dst_img_path)

            # ----------------------------------------------------------------
            # 2. Convert & save label mask – preserve instance IDs (uint16)
            # ----------------------------------------------------------------
            label_itk = sitk.ReadImage(src_paths["label"])  # keeps metadata
            label_arr = sitk.GetArrayFromImage(label_itk)    # shape (H,W)
            label_itk_out = sitk.GetImageFromArray(label_arr.astype(np.uint16))
            label_itk_out.CopyInformation(label_itk)  # copy spacing/origin if present
            dst_lbl_path = os.path.join(output_path, "labels", f"{base}.tif")
            sitk.WriteImage(label_itk_out, dst_lbl_path)

            # ----------------------------------------------------------------
            # 3. Convert ambiguity map –> binary float32 TIFF
            # ----------------------------------------------------------------
            amb_arr = sitk.GetArrayFromImage(sitk.ReadImage(src_paths["amb"]))
            # Some ambiguity PNGs are 0/255, others 0/1 – normalise:
            if amb_arr.max() > 1:
                amb_arr = amb_arr / 255.0
            amb_arr = (amb_arr > 0.5).astype(np.float32)  # ensure clean 0/1
            amb_itk_out = sitk.GetImageFromArray(amb_arr)
            amb_itk_out.CopyInformation(label_itk)  # align spacing/origin
            dst_amb_path = os.path.join(output_path, "confidence_maps", f"{base}.tif")
            sitk.WriteImage(amb_itk_out, dst_amb_path)

            # ----------------------------------------------------------------
            # 4. Append entry to metadata CSV (paths *relative* to processed_data)
            # ----------------------------------------------------------------
            with open(metadata_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        _relative(dst_img_path, output_path),
                        _relative(dst_lbl_path, output_path),
                        _relative(dst_amb_path, output_path),
                    ]
                )

    logger.info("✔ Dataset processing completed successfully!")


# ----------------------------------------------------------------------------
# CLI entry‑point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process NuInSeg dataset into MONAI‑ready format.")
    parser.add_argument("--input", required=True, help="Path to raw NuInSeg root directory")
    parser.add_argument(
        "--output",
        default="./processed_data",
        help="Destination directory for processed dataset (default: ./processed_data)",
    )
    args = parser.parse_args()

    process_dataset(args.input, args.output)
