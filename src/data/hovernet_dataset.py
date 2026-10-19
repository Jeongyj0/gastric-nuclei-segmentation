from pathlib import Path
import hashlib
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augment import augment_train
from src.study import project_path


ROOT = Path(__file__).resolve().parents[2]
HOVERNET = ROOT / "source_core" / "hovernet"

sys.path.insert(0, str(HOVERNET))

from models.hovernet.targets import gen_targets


INPUT_SIZE = 256
OUTPUT_SIZE = 164
CONTEXT_SIZE = 540
PATCHES_PER_IMAGE = 4


def sample_rng(seed, epoch, sample_id, slot):
    value = f"{seed}:{epoch}:{sample_id}:{slot}".encode()

    digest = hashlib.blake2b(
        value,
        digest_size=8,
    ).digest()

    return np.random.default_rng(
        int.from_bytes(digest, "little")
    )


def extract_context(array, center_x, center_y):
    pad = CONTEXT_SIZE // 2

    padded = cv2.copyMakeBorder(
        array,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_REFLECT_101,
    )

    x = center_x
    y = center_y

    return padded[
        y:y + CONTEXT_SIZE,
        x:x + CONTEXT_SIZE,
    ]


def sample_center(height, width, slot, rng):
    row = slot // 2
    col = slot % 2

    x0 = 0 if col == 0 else width // 2
    x1 = width // 2 if col == 0 else width

    y0 = 0 if row == 0 else height // 2
    y1 = height // 2 if row == 0 else height

    x = int(rng.integers(x0, x1))
    y = int(rng.integers(y0, y1))

    return x, y


class HoverNetTrainDataset(Dataset):
    def __init__(
        self,
        dataframe,
        root=ROOT,
        seed=42,
        patches_per_image=PATCHES_PER_IMAGE,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.root = Path(root)
        self.seed = seed
        self.patches_per_image = patches_per_image
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.df) * self.patches_per_image

    def __getitem__(self, index):
        image_index = index // self.patches_per_image
        slot = index % self.patches_per_image

        row = self.df.iloc[image_index]

        image_path = project_path(self.root / row["image_path"], must_exist=True)
        inst_path = project_path(self.root / row["inst_mask_path"], must_exist=True)

        with Image.open(image_path) as image:
            image = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

        inst = np.load(inst_path, allow_pickle=False).astype(np.int32)

        if image.shape[:2] != inst.shape:
            raise ValueError(
                "Image/mask shape mismatch in training sample"
            )

        rng = sample_rng(
            self.seed,
            self.epoch,
            row["id"],
            slot,
        )

        center_x, center_y = sample_center(
            inst.shape[0],
            inst.shape[1],
            slot,
            rng,
        )

        image = extract_context(
            image,
            center_x,
            center_y,
        )

        inst = extract_context(
            inst,
            center_x,
            center_y,
        )

        image, inst = augment_train(
            image,
            inst,
            rng,
            INPUT_SIZE,
        )

        targets = gen_targets(
            inst,
            (OUTPUT_SIZE, OUTPUT_SIZE),
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float()

        np_map = torch.from_numpy(
            targets["np_map"].astype(np.int64)
        )

        hv_map = torch.from_numpy(
            targets["hv_map"].astype(np.float32)
        )

        return {
            "img": image,
            "np_map": np_map,
            "hv_map": hv_map,
            "source": row["source"],
        }
