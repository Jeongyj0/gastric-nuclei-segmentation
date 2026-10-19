from pathlib import Path
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
HOVERNET = ROOT / "source_core" / "hovernet"

sys.path.insert(0, str(HOVERNET))

from models.hovernet.post_proc import process


INPUT_SIZE = 256
OUTPUT_SIZE = 164


def prepare_patches(image):
    h, w = image.shape[:2]

    rows = math.ceil((h - OUTPUT_SIZE) / OUTPUT_SIZE) + 1
    cols = math.ceil((w - OUTPUT_SIZE) / OUTPUT_SIZE) + 1

    last_h = rows * OUTPUT_SIZE
    last_w = cols * OUTPUT_SIZE

    margin = (INPUT_SIZE - OUTPUT_SIZE) // 2

    pad_top = margin
    pad_left = margin
    pad_bottom = last_h + INPUT_SIZE - h
    pad_right = last_w + INPUT_SIZE - w

    padded = np.pad(
        image,
        (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        ),
        mode="reflect",
    )

    patches = []
    for row in range(rows):
        y = row * OUTPUT_SIZE
        for col in range(cols):
            x = col * OUTPUT_SIZE
            patch = padded[y : y + INPUT_SIZE, x : x + INPUT_SIZE]
            patches.append((row, col, patch))

    return patches, rows, cols


@torch.inference_mode()
def predict_raw_map(model, image, device, batch_size=1):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    patches, rows, cols = prepare_patches(image)

    prediction = np.zeros(
        (rows * OUTPUT_SIZE, cols * OUTPUT_SIZE, 3),
        dtype=np.float32,
    )

    model.eval()

    for start in range(0, len(patches), batch_size):
        chunk = patches[start : start + batch_size]
        batch_np = np.stack(
            [item[2].transpose(2, 0, 1) for item in chunk],
            axis=0,
        ).copy()
        tensor = torch.from_numpy(batch_np).float().to(device)

        outputs = model(tensor)
        np_prob = F.softmax(outputs["np"], dim=1)[:, 1:2]
        raw_batch = torch.cat([np_prob, outputs["hv"]], dim=1)
        raw_batch = raw_batch.permute(0, 2, 3, 1).cpu().numpy()

        for (row, col, _), raw in zip(chunk, raw_batch):
            y = row * OUTPUT_SIZE
            x = col * OUTPUT_SIZE
            prediction[y : y + OUTPUT_SIZE, x : x + OUTPUT_SIZE] = raw

    h, w = image.shape[:2]
    return prediction[:h, :w], len(patches)


def postprocess(raw_map, return_centroids=False):
    return process(
        raw_map,
        nr_types=None,
        return_centroids=return_centroids,
    )
