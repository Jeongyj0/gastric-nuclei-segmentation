from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PQ_ATOL = 1e-12
COUNT_FIELDS = ("tp", "fp", "fn", "gt_instances", "pred_instances")
CURVE_FIELDS = ("tissue", "condition_id", "score", "x", "y", "TP", "FP", "GT", "N", "all_predictions_endpoint")


def _labels(mask, name):
    a = np.asarray(mask)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a two-dimensional integer mask")
    if np.any(a < 0):
        raise ValueError(f"{name} has negative labels")
    labels = np.union1d(np.array([0], dtype=a.dtype), np.unique(a))
    dense = np.searchsorted(labels, a).ravel()
    return labels, dense, a.shape


def score_and_match(gt, pred, np_map):
    gt_ids, g, shape = _labels(gt, "GT")
    pred_ids, p, pred_shape = _labels(pred, "Prediction")
    probability = np.asarray(np_map)
    if shape != pred_shape or probability.shape != shape:
        raise ValueError("GT, prediction and NP shapes must agree")
    if probability.dtype != np.float32:
        raise ValueError("NP map must contain saved float32 foreground values")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("NP values must be finite and in [0,1]")
    ng, npred = len(gt_ids) - 1, len(pred_ids) - 1
    ga = np.bincount(g, minlength=ng + 1)
    pa = np.bincount(p, minlength=npred + 1)
    sums = np.bincount(p, weights=probability.ravel().astype(np.float64), minlength=npred + 1)
    scores = sums[1:] / pa[1:]
    is_tp = np.zeros(npred, dtype=np.bool_)
    matched_gt = np.zeros(npred, dtype=gt_ids.dtype)
    matched_iou = np.zeros(npred, dtype=np.float64)
    paired_iou = np.empty(0, dtype=np.float64)
    if ng and npred:
        both = (g > 0) & (p > 0)
        if ng > np.iinfo(np.int64).max // (npred + 1) - 1:
            raise OverflowError("Too many instance labels for overlap encoding")
        encoded, intersection = np.unique(g[both] * (npred + 1) + p[both], return_counts=True)
        gi, pi = encoded // (npred + 1), encoded % (npred + 1)
        union = ga[gi] + pa[pi] - intersection
        overlap_iou = intersection.astype(np.float64) / union
        valid = overlap_iou > 0.5
        gi, pi, paired_iou = gi[valid], pi[valid], overlap_iou[valid]
        if len(np.unique(gi)) != len(gi) or len(np.unique(pi)) != len(pi):
            raise AssertionError("Strict IoU > 0.5 pairing was not unique")
        is_tp[pi - 1] = True
        matched_gt[pi - 1] = gt_ids[gi]
        matched_iou[pi - 1] = paired_iou
    tp = int(is_tp.sum())
    fp, fn = npred - tp, ng - tp
    iou_sum = float(paired_iou.sum(dtype=np.float64))
    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = tp / denominator if denominator else 0.0
    sq = iou_sum / (tp + 1e-6)
    return {
        "instance_ids": pred_ids[1:], "scores": scores, "is_tp": is_tp,
        "matched_gt": matched_gt, "iou": matched_iou,
        "totals": {"tp": tp, "fp": fp, "fn": fn, "gt_instances": ng,
                   "pred_instances": npred, "paired_iou_sum": iou_sum,
                   "dq": float(dq), "sq": float(sq), "pq": float(dq * sq),
                   "empty_gt": int(ng == 0), "empty_pred": int(npred == 0),
                   "empty_both": int(ng == 0 and npred == 0)},
    }


def audit_operating_point(result, expected, pq_atol=PQ_ATOL):
    totals = result["totals"]
    mismatches = {}
    for field in COUNT_FIELDS:
        saved = float(expected[field])
        if not np.isfinite(saved) or saved != int(saved) or totals[field] != int(saved):
            mismatches[field] = {"observed": totals[field], "expected": expected[field]}
    saved_pq = float(expected["pq"])
    error = abs(totals["pq"] - saved_pq)
    if not np.isfinite(saved_pq) or error > pq_atol:
        mismatches["pq"] = {"observed": totals["pq"], "expected": saved_pq, "absolute_error": error}
    if mismatches:
        raise ValueError("Frozen operating-point mismatch: " + json.dumps(mismatches, sort_keys=True))
    return {"passed": True, "count_fields": list(COUNT_FIELDS), "pq_absolute_error": error, "pq_atol": pq_atol}


def curve_points(scores, is_tp, gt_total, n_images=75):
    scores = np.asarray(scores, dtype=np.float64)
    is_tp = np.asarray(is_tp)
    if scores.ndim != 1 or is_tp.shape != scores.shape or is_tp.dtype != np.bool_:
        raise ValueError("scores and boolean is_tp must be one-dimensional and aligned")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Instance scores must be finite and in [0,1]")
    if n_images <= 0 or int(n_images) != n_images or gt_total < 0 or int(gt_total) != gt_total:
        raise ValueError("Invalid image or GT count")
    if int(is_tp.sum()) > gt_total:
        raise ValueError("TP exceeds the number of GT instances")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    group_end = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True]) if len(ordered) else np.empty(0, dtype=np.int64)
    tp = np.r_[0, np.cumsum(is_tp[order], dtype=np.int64)[group_end]]
    kept = np.r_[0, group_end + 1]
    fp = kept - tp
    thresholds = np.r_[np.inf, ordered[group_end]]
    endpoint = np.zeros(len(tp), dtype=np.bool_)
    endpoint[-1] = True
    return {"score": thresholds, "x": fp / float(n_images),
            "y": tp / float(gt_total) if gt_total else np.zeros(len(tp)),
            "TP": tp, "FP": fp, "GT": np.full(len(tp), gt_total, dtype=np.int64),
            "N": np.full(len(tp), n_images, dtype=np.int64),
            "all_predictions_endpoint": endpoint}


def write_curve_csv(path, tissue, condition_id, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CURVE_FIELDS)
        for i in range(len(points["score"])):
            writer.writerow([tissue, condition_id, repr(float(points["score"][i])),
                             repr(float(points["x"][i])), repr(float(points["y"][i])),
                             int(points["TP"][i]), int(points["FP"][i]),
                             int(points["GT"][i]), int(points["N"][i]),
                             int(points["all_predictions_endpoint"][i])])


def save_private_result(directory, result, audit=None, provenance=None):
    directory = Path(directory)
    if "private" not in directory.parts:
        raise ValueError("Per-instance records must be stored under private/")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "instance_scores_matches.npz",
                        **{k: result[k] for k in ("instance_ids", "scores", "is_tp", "matched_gt", "iou")})
    metadata = {"totals": result["totals"], "audit": audit, "provenance": provenance}
    (directory / "froc_image_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_condition(result_directories, tissue, condition_id, output_root, expected_n=75):
    directories = [Path(p).resolve() for p in result_directories]
    if len(directories) != expected_n or len(set(directories)) != expected_n:
        raise ValueError("Incomplete or duplicate image records for condition")
    if tissue not in ("STNT", "STIN", "STDI", "STMX") or condition_id not in ("base", "real300", "add", "replace50", "replace100"):
        raise ValueError("Unknown tissue or condition")
    scores, flags, totals, pq_errors = [], [], [], []
    for directory in directories:
        if "private" not in directory.parts:
            raise ValueError("Per-instance records must come from private/")
        meta = json.loads((directory / "froc_image_summary.json").read_text(encoding="utf-8"))
        audit = meta.get("audit") or {}
        if audit.get("passed") is not True or tuple(audit.get("count_fields", [])) != COUNT_FIELDS:
            raise ValueError("Missing successful original-operating-point audit")
        pq_error = float(audit["pq_absolute_error"])
        if not np.isfinite(pq_error) or pq_error > PQ_ATOL:
            raise ValueError("Invalid original-operating-point PQ audit")
        with np.load(directory / "instance_scores_matches.npz", allow_pickle=False) as data:
            ids = data["instance_ids"]
            s, tp = data["scores"].copy(), data["is_tp"].copy()
            matched, iou = data["matched_gt"], data["iou"]
            if not (ids.shape == s.shape == tp.shape == matched.shape == iou.shape) or s.ndim != 1:
                raise ValueError("Inconsistent private instance arrays")
            if s.dtype != np.float64 or tp.dtype != np.bool_ or np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
                raise ValueError("Invalid private instance IDs/scores/match flags")
            if not np.array_equal(matched > 0, tp) or not np.array_equal(iou > .5, tp) or np.any(iou[~tp] != 0):
                raise ValueError("Inconsistent private TP/match/IoU records")
            if len(np.unique(matched[tp])) != int(tp.sum()):
                raise ValueError("Private match records are not one-to-one")
        t = meta["totals"]
        if t["tp"] != int(tp.sum()) or t["pred_instances"] != len(ids) or t["fp"] != len(ids) - int(tp.sum()) or t["fn"] != t["gt_instances"] - int(tp.sum()):
            raise ValueError("Private totals disagree with instance records")
        scores.append(s)
        flags.append(tp)
        totals.append(t)
        pq_errors.append(pq_error)
    gt_total = sum(t["gt_instances"] for t in totals)
    points = curve_points(np.concatenate(scores), np.concatenate(flags), gt_total, expected_n)
    summed_tp, summed_fp = sum(t["tp"] for t in totals), sum(t["fp"] for t in totals)
    if points["TP"][-1] != summed_tp or points["FP"][-1] != summed_fp:
        raise AssertionError("Curve endpoint does not reproduce the original operating point")
    path = Path(output_root) / "data/froc_curves" / f"{tissue}_{condition_id}.csv"
    write_curve_csv(path, tissue, condition_id, points)
    return {"tissue": tissue, "condition_id": condition_id,
            **{k: float(points[k][-1]) for k in ("score", "x", "y")},
            **{k: int(points[k][-1]) for k in ("TP", "FP", "GT", "N")},
            "all_predictions_endpoint": 1, "FN": sum(t["fn"] for t in totals),
            "pred_instances": sum(t["pred_instances"] for t in totals),
            "pq_audit_pass_count": expected_n, "pq_max_absolute_error": max(pq_errors),
            "curve_point_count": len(points["score"])}


def self_test(project_root=None):
    import importlib.util
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    canonical_path = root / "source_core/hovernet/metrics/stats_utils.py"
    spec = importlib.util.spec_from_file_location("froc_canonical_stats", canonical_path)
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    checks = []

    def reference(gt, pred):
        gids, g, shape = _labels(gt, "GT")
        pids, p, _ = _labels(pred, "Prediction")
        ng, npr = len(gids) - 1, len(pids) - 1
        if not ng or not npr:
            return {"tp": 0, "fp": npr, "fn": ng, "pq": 0.0, "pairs": {}}
        metrics, pairs = canonical.get_fast_pq(np.pad(g.reshape(shape), 1), np.pad(p.reshape(shape), 1), match_iou=0.5)
        return {"tp": len(pairs[0]), "fp": len(pairs[3]), "fn": len(pairs[2]), "pq": float(metrics[2]),
                "pairs": {int(pids[pp]): int(gids[gg]) for gg, pp in zip(pairs[0], pairs[1])}}

    def check(name, gt, pred, probability=None, test_filters=True):
        gt, pred = np.asarray(gt, dtype=np.uint32), np.asarray(pred, dtype=np.uint32)
        probability = np.full(gt.shape, .75, dtype=np.float32) if probability is None else np.asarray(probability, dtype=np.float32)
        result = score_and_match(gt, pred, probability)
        ref = reference(gt, pred)
        for key in ("tp", "fp", "fn"):
            assert result["totals"][key] == ref[key], (name, key)
        assert abs(result["totals"]["pq"] - ref["pq"]) <= PQ_ATOL, name
        actual_pairs = {int(p): int(g) for p, g, yes in zip(result["instance_ids"], result["matched_gt"], result["is_tp"]) if yes}
        assert actual_pairs == ref["pairs"], name
        for ident, score in zip(result["instance_ids"], result["scores"]):
            assert score == probability[pred == ident].astype(np.float64).mean(), name
        threshold_tests = 0
        if test_filters:
            for threshold in np.r_[np.inf, np.unique(result["scores"])]:
                selected = result["scores"] >= threshold
                filtered = np.where(np.isin(pred, result["instance_ids"][selected]), pred, 0)
                filtered_ref = reference(gt, filtered)
                assert int(result["is_tp"][selected].sum()) == filtered_ref["tp"], (name, threshold)
                assert int(selected.sum() - result["is_tp"][selected].sum()) == filtered_ref["fp"], (name, threshold)
                threshold_tests += 1
        checks.append({"name": name, "canonical_pq_difference": abs(result["totals"]["pq"] - ref["pq"]), "threshold_filter_checks": threshold_tests})
        return result

    exact = check("strict_half_is_not_match", [[7, 7, 0, 0]], [[9, 0, 0, 0]])
    assert exact["totals"]["tp"] == 0
    above = check("strict_above_half_matches", [[7, 7, 7, 0]], [[9, 9, 0, 0]])
    assert above["totals"]["tp"] == 1
    check("both_empty", np.zeros((3, 4)), np.zeros((3, 4)))
    check("empty_gt", [[0, 0, 0]], [[4, 0, 9]])
    check("empty_pred", [[4, 0, 9]], [[0, 0, 0]])
    check("no_background_sparse_ids", [[4000000000, 4000000000], [17, 17]], [[8, 8], [99, 99]])
    check("split_at_half", [[2, 2, 2, 2]], [[4, 4, 8, 8]])
    check("merge_at_half", [[2, 2, 3, 3]], [[4, 4, 4, 4]])
    rng = np.random.default_rng(20261013)
    for i in range(24):
        gt = np.kron(rng.integers(0, 6, (4, 5), dtype=np.uint32), np.ones((3, 3), dtype=np.uint32))
        pred = gt * 17
        change = rng.random(gt.shape) < (.04 + .008 * i)
        pred[change] = rng.integers(0, 7, int(change.sum()), dtype=np.uint32) * 17
        check(f"random_partition_{i:02d}", gt, pred, rng.random(gt.shape, dtype=np.float32))
    points = curve_points(np.array([.8, .5, .8, .5, .2]), np.array([True, True, False, False, True]), 4, 75)
    assert np.array_equal(points["score"], [np.inf, .8, .5, .2])
    assert np.array_equal(points["TP"], [0, 1, 2, 3])
    assert np.array_equal(points["FP"], [0, 1, 2, 2])
    assert points["x"][-1] == 2 / 75 and points["y"][-1] == 3 / 4
    assert np.array_equal(points["all_predictions_endpoint"], [False, False, False, True])
    empty = curve_points([], np.array([], dtype=bool), 0, 75)
    assert len(empty["score"]) == 1 and empty["all_predictions_endpoint"][0] and empty["y"][0] == 0
    audit_operating_point(above, above["totals"])
    for field, increment in (("tp", 1), ("pq", 1e-5)):
        bad = dict(above["totals"])
        bad[field] += increment
        try:
            audit_operating_point(above, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatch did not fail: " + field)
    return {"status": "passed", "scope": "CPU synthetic arrays only; no real-image inference/evaluation/bootstrap", "canonical_source": str(canonical_path),
            "canonical_sha256": hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
            "test_cases": checks, "tie_grouping_origin_endpoint": "passed", "mismatch_fail_closed": "passed",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FROC scores and instance matches.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test-report", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test, or import score_and_match/curve_points from analysis.froc")
    report = self_test()
    if args.test_report:
        args.test_report.parent.mkdir(parents=True, exist_ok=True)
        args.test_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "synthetic_cases": len(report["test_cases"]),
                      "threshold_filter_checks": sum(x["threshold_filter_checks"] for x in report["test_cases"])}, ensure_ascii=False))
