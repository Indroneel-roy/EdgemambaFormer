"""
Dataset discovery, image/mask pairing, and PyTorch Dataset/DataLoader
utilities for the four polyp-segmentation datasets used in the shared-
training protocol (Section 4.1):

    Kvasir-SEG          -- root/Kvasir-SEG/{images,masks}/*  (or root/{images,masks}/*)
    CVC-ClinicDB        -- root/PNG/Original/*.png, root/PNG/'Ground Truth'/*.png
    CVC-ColonDB         -- root/{images,masks}/*  (100% zero-shot test)
    ETIS-LaribPolypDB   -- root/{images,masks}/*  (100% zero-shot test)
"""

import os
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")


def _glob_multi(directory, patterns=IMG_EXTS):
    paths = []
    for p in patterns:
        paths.extend(Path(directory).glob(p))
    return sorted(paths)


def autodetect_roots(base="/kaggle/input"):
    """
    Scans `base` for the four datasets based on their known folder
    signatures. Returns a dict {name: path or None}. A None value means
    autodetection failed for that dataset -- set it manually in
    CFG['data_roots'][...] after checking your data directory.
    """
    found = {"kvasir": None, "clinicdb": None, "colondb": None, "etis": None}
    if not os.path.isdir(base):
        return found

    for dirpath, dirnames, filenames in os.walk(base):
        low = dirpath.lower()
        dirset = set(d.lower() for d in dirnames)

        # Kvasir-SEG: has images/ + masks/ subdirs, and 'kvasir' in the path
        if {"images", "masks"} <= dirset and "kvasir" in low and found["kvasir"] is None:
            found["kvasir"] = dirpath

        # CVC-ClinicDB: has a PNG/ subdir containing Original/ and Ground Truth/
        if "png" in dirset and found["clinicdb"] is None:
            png_dir = os.path.join(dirpath, [d for d in dirnames if d.lower() == "png"][0])
            try:
                sub = set(d.lower() for d in os.listdir(png_dir))
                if "original" in sub and any("ground" in s for s in sub):
                    found["clinicdb"] = dirpath
            except OSError:
                pass

        # ETIS / ColonDB: flat images/ + masks/, disambiguated by keyword in path
        if {"images", "masks"} <= dirset:
            if ("etis" in low or "larib" in low) and found["etis"] is None:
                found["etis"] = dirpath
            elif "colon" in low and found["colondb"] is None:
                found["colondb"] = dirpath

    return found


def build_pairs_flat(root, img_subdir="images", mask_subdir="masks"):
    """Layout: root/images/*, root/masks/*  (ETIS-LaribPolypDB, CVC-ColonDB)."""
    root = Path(root)
    img_dir, mask_dir = root / img_subdir, root / mask_subdir
    pairs = []
    for ip in _glob_multi(img_dir):
        mp = None
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            cand = mask_dir / (ip.stem + ext)
            if cand.exists():
                mp = cand
                break
        if mp is None and (mask_dir / ip.name).exists():
            mp = mask_dir / ip.name
        if mp is not None:
            pairs.append((str(ip), str(mp)))
    return pairs


def build_pairs_kvasir(root):
    """Layout: root/Kvasir-SEG/{images,masks}/*, falling back to root/{images,masks}/*."""
    root = Path(root)
    inner = root / "Kvasir-SEG"
    target = inner if (inner / "images").exists() else root
    return build_pairs_flat(target, "images", "masks")


def build_pairs_clinicdb(root):
    """Layout: root/PNG/Original/*.png, root/PNG/'Ground Truth'/*.png."""
    root = Path(root)
    img_dir = root / "PNG" / "Original"
    mask_dir = root / "PNG" / "Ground Truth"
    pairs = []
    for ip in _glob_multi(img_dir):
        mp = mask_dir / ip.name
        if mp.exists():
            pairs.append((str(ip), str(mp)))
    return pairs


def stratified_split(pairs, train_ratio, seed):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    split = int(len(pairs) * train_ratio)
    return pairs[:split], pairs[split:]


class PolypDataset(Dataset):
    """
    Generic image/mask dataset -- works for any of the four sources above,
    since build_pairs_* already normalises everything to (image_path, mask_path).
    """

    def __init__(self, pairs, transforms=None):
        self.pairs = pairs
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        mask = (mask > 127).float().unsqueeze(0)
        return image, mask


def make_loader(pairs, transforms, cfg, shuffle):
    ds = PolypDataset(pairs, transforms)
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=shuffle,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
