import time
import numpy as np
from PIL import Image
from src.study import project_path
from src.eval.metrics import score_raw_prediction, summarize


def evaluate_images(model, rows, device, batch_size=1, progress=True, *, on_prediction=None):
    from src.eval.inference import predict_raw_map, postprocess
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        with Image.open(project_path(row["image_path"], must_exist=True)) as image:
            image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        true_inst = np.load(project_path(row["inst_mask_path"], must_exist=True), allow_pickle=False)
        if image.shape[:2] != true_inst.shape:
            raise ValueError(f"Evaluation sample_{index:03d}: image/mask shape mismatch")
        raw, patches = predict_raw_map(model, image, device, batch_size=batch_size)
        if not np.isfinite(raw).all():
            raise FloatingPointError(f"Evaluation sample_{index:03d}: nonfinite prediction")
        predicted, _ = postprocess(raw, return_centroids=False)
        scores = score_raw_prediction(true_inst, predicted, raw[..., 0])
        scores.update(id=row["id"], pid=row["pid"], patches=patches)
        if on_prediction is not None:
            on_prediction(index, row, true_inst, predicted, raw, scores)
        results.append(scores)
        if progress and (index % 15 == 0 or index == len(rows)):
            print(f"  Evaluating {index}/{len(rows)}", flush=True)
    summary = summarize(results)
    summary["seconds"] = time.perf_counter() - started
    return results, summary
