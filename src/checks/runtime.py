import sys
import numpy as np
from src.study import ROOT, project_path, sha256


def check_runtime(device_name="cuda"):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT / "source_core" / "hovernet"))
    from models.hovernet.utils import msge_loss
    from models.hovernet.targets import gen_targets
    from src.train.runtime import create_model, load_pretrained, model_hash, set_seed, compute_loss, training_mode
    from src.eval.inference import predict_raw_map, postprocess
    from src.eval.validation_inference import predict_raw_map as validation_predict
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    y, x = torch.meshgrid(torch.arange(19, device=device, dtype=torch.float32),
                          torch.arange(21, device=device, dtype=torch.float32), indexing="ij")
    true = torch.zeros((1, 19, 21, 2), device=device)
    focus = torch.zeros((1, 19, 21), device=device)
    focus[:, 4:-4, 4:-4] = 1
    ramps = []
    for channel in (0, 1):
        for axis, ramp in (("X", x), ("Y", y)):
            prediction = torch.zeros_like(true)
            prediction[0, :, :, channel] = ramp
            prediction.requires_grad_(True)
            value = msge_loss(true, prediction, focus)
            value.backward()
            expected = 72.0 if (channel, axis) in ((0, "X"), (1, "Y")) else 0.0
            assert abs(float(value.detach()) - expected) < 1e-4
            assert torch.isfinite(prediction.grad).all().item()
            ramps.append({"channel": channel, "axis": axis, "loss": float(value.detach())})
    ann = np.zeros((256, 256), dtype=np.int32)
    ann[99:145, 100:150] = 1
    ann[75:78, 75:78] = 2
    target = gen_targets(ann, (164, 164))
    hv = target["hv_map"]
    assert hv.shape == (164, 164, 2) and target["np_map"].shape == (164, 164)
    assert hv[74, 60, 0] < hv[74, 90, 0] and hv[60, 76, 1] < hv[90, 76, 1]
    assert np.count_nonzero(target["np_map"][29:32, 29:32]) == 9
    assert np.count_nonzero(hv[29:32, 29:32]) == 0
    logits = torch.zeros((1, 2, 164, 164), device=device, requires_grad=True)
    prediction_hv = torch.zeros_like(logits, requires_grad=True)
    loss, terms = compute_loss({"np": logits, "hv": prediction_hv},
                              torch.from_numpy(target["np_map"]).long().unsqueeze(0).to(device),
                              torch.from_numpy(hv).unsqueeze(0).to(device))
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(prediction_hv.grad).all()
    assert abs(float(loss.detach()) - sum(float(v.detach()) for v in terms.values())) < 1e-5

    class CoordinateModel(torch.nn.Module):
        def forward(self, image):
            center = image[:, :2, 46:210, 46:210]
            return {"np": torch.zeros_like(center), "hv": center}

    assert validation_predict is predict_raw_map
    yy, xx = np.mgrid[:257, :331].astype(np.float32)
    image = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    first, tiles = predict_raw_map(CoordinateModel(), image, device, 1)
    second, _ = predict_raw_map(CoordinateModel(), image, device, 3)
    assert tiles == 6 and np.array_equal(first, second)
    assert np.array_equal(first[..., 1], xx) and np.array_equal(first[..., 2], yy)
    empty, _ = postprocess(np.zeros((164, 164, 3), dtype=np.float32))
    assert np.count_nonzero(empty) == 0
    set_seed(123)
    model = create_model(input_ch=3, nr_types=None, freeze=False, mode="fast")
    pretrained = project_path("pretrained/ImageNet-ResNet50-Preact_pytorch.tar", must_exist=True)
    matched = load_pretrained(model, pretrained)
    initial = model_hash(model)
    training_mode(model, True)
    assert not model.conv0.training and not any(p.requires_grad for p in model.d3.parameters())
    training_mode(model, False)
    assert model.conv0.training and all(p.requires_grad for p in model.d3.parameters())
    model = model.to(device).eval()
    with torch.inference_mode():
        result = model(torch.zeros((1, 3, 256, 256), device=device))
    assert all(tuple(v.shape) == (1, 2, 164, 164) and torch.isfinite(v).all().item()
               for v in result.values())
    del model, result
    set_seed(123)
    repeated = create_model(input_ch=3, nr_types=None, freeze=False, mode="fast")
    load_pretrained(repeated, pretrained)
    assert model_hash(repeated) == initial
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "ramps": ramps, "target_small_nucleus_rule_preserved": True,
            "four_losses_finite_and_differentiable": True, "shared_inference": True,
            "tile_reconstruction": True, "empty_postprocessing": True,
            "pretrained_match_count": matched["matched"], "pretrained_sha256": sha256(pretrained),
            "same_seed_initialization_equal": True, "full_model_forward": True,
            "training_executed": False, "optimizer_steps": 0}
