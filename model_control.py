import torch
import torch.nn as nn
from typing import Dict

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.bundle import download, load

# -----------------------------------------------------------------------------
#  Simple UNet for Semantic Segmentation (2‑D) – fixed channel mismatch
# -----------------------------------------------------------------------------
class SimpleSegUNet(nn.Module):
    """UNet backbone that outputs a *feature map* (C=64), followed by a small
    classifier head that collapses it to a single‑channel logits map.
    """

    def __init__(
        self,
        pretrained: bool = True,
        bundle_name: str = "spleen_ct_segmentation",
        spatial_dims: int = 2,
        in_channels: int = 3,
        feat_channels: int = 64,
    ) -> None:
        super().__init__()

        # UNet now outputs *feat_channels* feature maps, not 1
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feat_channels,
            channels=(feat_channels, feat_channels * 2, feat_channels * 4, feat_channels * 8, feat_channels * 16),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        if pretrained:
            self._load_pretrained_weights(bundle_name, in_channels)

        # 64‑ch → 1‑ch classifier head
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _replicate_first_weight(w: torch.Tensor, new_in: int) -> torch.Tensor:
        return w.repeat(1, new_in, 1, 1) / new_in

    def _load_pretrained_weights(self, bundle_name: str, in_channels: int):
        """Load what matches from a MONAI bundle (encoder weights)."""
        try:
            download(name=bundle_name, bundle_dir="./bundles", source="github")
            zoo_sd = load(name=bundle_name, bundle_dir="./bundles", return_state_dict=True)
            own_sd = self.unet.state_dict()
            adapted = {}
            for k, v in zoo_sd.items():
                if k in own_sd and v.shape == own_sd[k].shape:
                    adapted[k] = v
                elif k.endswith("conv.conv.weight") and v.shape[1] == 1 and in_channels == 3:
                    adapted[k] = self._replicate_first_weight(v, 3)
            self.unet.load_state_dict({**own_sd, **adapted})
            print(f"[Info] Loaded pretrained weights from bundle '{bundle_name}'.")
        except Exception as e:
            print(f"[Warn] Could not load pretrained weights from bundle '{bundle_name}': {e}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.unet(x)            # B × 64 × H × W
        logits = self.head(feat)       # B × 1 × H × W
        return {"nucleus_pred": logits}


# -----------------------------------------------------------------------------
#  Loss
# -----------------------------------------------------------------------------
class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)

    def forward(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]):
        ls = self.dice(out["nucleus_pred"], tgt["nucleus_gt"])
        return ls, {"seg": ls}


# -----------------------------------------------------------------------------
#  Factory
# -----------------------------------------------------------------------------

def get_model_and_loss(pretrained: bool = True, freeze_encoder: bool = False):
    model = SimpleSegUNet(pretrained=pretrained)
    if freeze_encoder:
        for n, p in model.unet.named_parameters():
            if n.startswith("encoder") or n.startswith("down"):
                p.requires_grad_(False)
    crit = SegLoss()
    return model, crit


# -----------------------------------------------------------------------------
#  Transforms (unchanged)
# -----------------------------------------------------------------------------
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Lambdad,
    CastToTyped, ToTensord, Resized, ToNumpyd, MapTransform,
)
from monai.apps.pathology.transforms import NormalizeHEStainsd
from monai.data import PILReader, ITKReader
import numpy as np
import SimpleITK as sitk


def rgb_slicer(arr):
    return arr[..., :3]


class _Denoise(MapTransform):
    def __init__(self, keys, iters=3, ts=0.05):
        super().__init__(keys); self.it = iters; self.ts = ts

    def __call__(self, data):
        for k in self.keys:
            img = data[k]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            ch_out = []
            for c in range(img.shape[0]):
                itk = sitk.GetImageFromArray(img[c]); itk = sitk.Cast(itk, sitk.sitkFloat32)
                f = sitk.CurvatureFlowImageFilter(); f.SetNumberOfIterations(self.it); f.SetTimeStep(self.ts)
                ch_out.append(sitk.GetArrayFromImage(f.Execute(itk)))
            data[k] = np.stack(ch_out)
        return data


class _Prep(MapTransform):
    def __call__(self, data):
        inst = data["label_path"]
        inst_arr = inst.as_tensor() if hasattr(inst, "as_tensor") else torch.as_tensor(inst)
        data["nucleus_gt"] = (inst_arr > 0).float()
        return data


def get_transforms(size=(256, 256)):
    stain = {
        "tli": 240,
        "alpha": 1,
        "beta": 0.15,
        "target_he": ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)),
        "max_cref": (1.9705, 1.0308),
    }
    resize = Resized(keys=["image_path", "label_path"], spatial_size=size, mode=("bilinear", "nearest"))

    common = [
        Lambdad(keys="image_path", func=rgb_slicer),
        ToNumpyd(keys="image_path"),
        NormalizeHEStainsd(keys="image_path", **stain),
        _Denoise(keys=["image_path"]),
        EnsureChannelFirstd(keys="image_path", channel_dim=-1),
        EnsureChannelFirstd(keys=["label_path"], channel_dim="no_channel"),
        CastToTyped(keys=["image_path"], dtype=np.float32),
        resize,
        ScaleIntensityd(keys="image_path", minv=0.0, maxv=1.0),
        _Prep(keys=None),
        ToTensord(keys=["image_path", "label_path", "nucleus_gt"]),
    ]

    loaders = [PILReader(), ITKReader()]
    train = Compose([LoadImaged(keys=["image_path", "label_path"], reader=loaders), *common])
    val   = Compose([LoadImaged(keys=["image_path", "label_path"], reader=loaders), *common])
    return train, val
