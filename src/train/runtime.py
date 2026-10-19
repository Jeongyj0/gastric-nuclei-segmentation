import hashlib
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from src.study import ROOT, project_path

sys.path.insert(0, str(ROOT / "source_core" / "hovernet"))
from models.hovernet.net_desc import create_model
from models.hovernet.utils import dice_loss, mse_loss, msge_loss, xentropy_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_pretrained(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("desc", checkpoint)
    current = model.state_dict()
    matched = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key in current and current[key].shape == value.shape:
            matched[key] = value
    required = {key for key in current if key.split(".")[0] in ENCODER
                and not key.endswith("num_batches_tracked")}
    missing_encoder = sorted(required - matched.keys())
    if missing_encoder:
        raise RuntimeError(f"Unmatched pretrained encoder state entries: {len(missing_encoder)}.")
    result = model.load_state_dict(matched, strict=False)
    return {"matched": len(matched), "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys), "encoder_complete": True}


def model_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


ENCODER = ("conv0", "d0", "d1", "d2", "d3")


def training_mode(model, frozen):
    model.train()
    model.freeze = False
    for name in ENCODER:
        module = getattr(model, name)
        module.train(not frozen)
        for parameter in module.parameters():
            parameter.requires_grad_(not frozen)


def compute_loss(outputs, true_np, true_hv):
    np_prob = F.softmax(outputs["np"].permute(0, 2, 3, 1).contiguous(), dim=-1)
    hv = outputs["hv"].permute(0, 2, 3, 1).contiguous()
    onehot = F.one_hot(true_np, num_classes=2).float()
    terms = {"np_ce": xentropy_loss(onehot, np_prob), "np_dice": dice_loss(onehot, np_prob),
             "hv_mse": mse_loss(true_hv, hv), "hv_msge": msge_loss(true_hv, hv, onehot[..., 1])}
    total = sum(terms.values())
    return total, terms


def save_checkpoint(path, model, optimizer, scheduler, metadata):
    path = project_path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": {
                    "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}, temporary)
    os.replace(temporary, path)
