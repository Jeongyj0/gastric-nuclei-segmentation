import sys
import numpy as np
from src.study import ROOT

sys.path.insert(0, str(ROOT / "source_core" / "hovernet"))
from metrics.stats_utils import get_dice_1, get_fast_aji, get_fast_aji_plus, get_fast_pq

SCORES = ("dice", "aji", "aji_plus", "dq", "sq", "pq", "precision", "recall", "np_dice")


def normalize_labels(array):
    array = np.asarray(array)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError("Instance mask must be a nonnegative 2D integer array.")
    labels = np.union1d(np.array([0], dtype=array.dtype), np.unique(array))
    return np.searchsorted(labels, array).astype(np.int32)


def score_instances(true, pred):
    true, pred = normalize_labels(true), normalize_labels(pred)
    if true.shape != pred.shape:
        raise ValueError("Ground-truth and prediction shapes differ.")
    ntrue, npred = int(true.max(initial=0)), int(pred.max(initial=0))
    scores = {key: 0.0 for key in SCORES if key != "np_dice"}
    scores.update(tp=0, fp=npred, fn=ntrue, paired_iou_sum=0.0,
                  gt_instances=ntrue, pred_instances=npred,
                  empty_gt=int(ntrue == 0), empty_pred=int(npred == 0),
                  empty_both=int(ntrue == 0 and npred == 0))
    if ntrue and npred:
        true, pred = np.pad(true, 1), np.pad(pred, 1)
        (dq, sq, pq), pairing = get_fast_pq(true, pred, match_iou=0.5)
        tp, fp, fn = len(pairing[0]), len(pairing[3]), len(pairing[2])
        scores.update(dq=float(dq), sq=float(sq), pq=float(pq),
                      dice=float(get_dice_1(true, pred)), aji=float(get_fast_aji(true, pred)),
                      aji_plus=float(get_fast_aji_plus(true, pred)),
                      tp=tp, fp=fp, fn=fn, precision=tp / npred, recall=tp / ntrue,
                      paired_iou_sum=float(sq * (tp + 1e-6)) if tp else 0.0)
    return scores


def score_raw_prediction(true_inst, pred_inst, np_probability):
    probability = np.asarray(np_probability)
    if probability.shape != true_inst.shape or not np.isfinite(probability).all():
        raise ValueError("NP probability shape or finiteness check failed.")
    scores = score_instances(true_inst, pred_inst)
    true_np, pred_np = true_inst > 0, probability >= 0.5
    intersection = int(np.count_nonzero(true_np & pred_np))
    denominator = int(np.count_nonzero(true_np) + np.count_nonzero(pred_np))
    scores.update(np_dice=2 * intersection / denominator if denominator else 0.0,
                  np_intersection=intersection, np_denominator=denominator)
    return scores


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation images.")
    result = {"images": len(rows), "primary_metric": "image_mean_pq",
              "empty_policy": "Undefined scores for empty ground truth or predictions are zero; all images are included"}
    for name in SCORES:
        values = np.asarray([r[name] for r in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite metric found: " + name)
        result["mean_" + name] = float(values.mean())
        result["std_" + name] = float(values.std(ddof=1)) if len(values) > 1 else None
    for name in ("tp", "fp", "fn", "gt_instances", "pred_instances", "empty_gt", "empty_pred", "empty_both"):
        result[name] = sum(int(row[name]) for row in rows)
    tp, fp, fn = result["tp"], result["fp"], result["fn"]
    iou = sum(row["paired_iou_sum"] for row in rows)
    denominator = tp + 0.5 * fp + 0.5 * fn
    np_denominator = sum(row["np_denominator"] for row in rows)
    result.update(pooled_dq=tp / denominator if denominator else 0.0,
                  pooled_sq=iou / tp if tp else 0.0,
                  pooled_pq=iou / denominator if denominator else 0.0,
                  instance_precision=tp / (tp + fp) if tp + fp else 0.0,
                  instance_recall=tp / (tp + fn) if tp + fn else 0.0,
                  pooled_np_dice=2 * sum(r["np_intersection"] for r in rows) / np_denominator
                  if np_denominator else 0.0)
    return result
