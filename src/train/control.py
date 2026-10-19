import math


class ValidationPlateau:
    def __init__(self, optimizer, lr_schedule, early_stopping):
        self.optimizer = optimizer
        self.lr_schedule = dict(lr_schedule)
        self.early_stopping = dict(early_stopping)
        self.reference_pq = None
        self.bad_checks = 0
        self.lr_reduced = False
        self.reduced_at_update = None
        self.should_stop = False
        self.last_update = 0
        self.checks = 0

    def step(self, pq, update):
        if not math.isfinite(pq) or not 0 <= pq <= 1:
            raise ValueError("Validation PQ must be finite and within [0, 1].")
        if type(update) is not int or update <= self.last_update or self.should_stop:
            raise ValueError("Validation updates must be increasing integers before stopping.")
        self.last_update = update
        self.checks += 1
        if update <= self.early_stopping["start_after_updates"]:
            self.reference_pq = pq if self.reference_pq is None else max(self.reference_pq, pq)
            self.bad_checks = 0
        elif self.reference_pq is None or pq > self.reference_pq + self.early_stopping["min_delta"]:
            self.reference_pq = pq
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        if not self.lr_reduced and self.bad_checks >= self.lr_schedule["patience_checks"]:
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr_schedule["reduced_lr"]
            self.lr_reduced = True
            self.reduced_at_update = update
        self.should_stop = self.bad_checks >= self.early_stopping["patience_checks"]
        return self.should_stop

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "lr_schedule", "early_stopping", "reference_pq", "bad_checks", "lr_reduced",
            "reduced_at_update", "should_stop", "last_update", "checks")}


def validate_completed_run(status, settings):
    update = status.get("update")
    maximum = settings["max_optimizer_updates"]
    if status.get("status") != "completed" or type(update) is not int or not 0 < update <= maximum:
        raise ValueError("Main evaluation requires a successfully completed run.")
    if update == maximum:
        return
    early = settings.get("early_stopping")
    state = status.get("validation_control", {})
    if (early is None or status.get("stop_reason") != "early_stopping" or
            update <= early["start_after_updates"] or update % settings["validation_interval_updates"] or
            state.get("early_stopping") != early or state.get("lr_schedule") != settings["lr_schedule"] or
            state.get("last_update") != update or state.get("should_stop") is not True or
            state.get("lr_reduced") is not True or state.get("bad_checks") != early["patience_checks"]):
        raise ValueError("Run did not satisfy the configured early stopping rule.")


def validation_due(update, maximum, interval):
    return update == maximum or update % interval == 0


def learning_rate_after(completed_updates, schedule):
    return schedule["initial_lr"] * schedule["gamma"] ** (
        completed_updates // schedule["step_size_updates"])


def select_checkpoints(best, pq, np_dice, update):
    selected = []
    for name, score in (("pq", pq), ("np_dice", np_dice)):
        if best[name] is None or score > best[name]["score"]:
            best[name] = {"score": float(score), "update": int(update)}
            selected.append("best_" + name + ".pth")
    return selected
