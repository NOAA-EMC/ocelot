"""AR rollout curriculum callback."""

import lightning.pytorch as pl


class RolloutCurriculumCallback(pl.Callback):
    """Increment the number of AR windows during training following a curriculum.

    Monitors ``pl_module.get_current_rollout_steps()`` after every gradient
    update.  When that value changes (meaning the model's internal schedule
    decided to increase AR depth), the callback tells the datamodule to
    produce longer sequences via ``trainer.datamodule.set_ar_steps()``.

    This mirrors GraphCast's approach: start with 1 AR step, then gradually
    increase the rollout depth as training progresses.

    Args:
        max_ar_steps: Hard cap on the number of AR windows (inclusive).
                      The model's schedule can return any value up to this.
        log_on_change: If True, print a banner when ar_steps changes.
    """

    def __init__(self, max_ar_steps: int = 4, log_on_change: bool = True):
        self.max_ar_steps = int(max_ar_steps)
        self.log_on_change = bool(log_on_change)
        self._current_ar_steps: int = 1

    def on_train_start(self, trainer, pl_module):
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "set_ar_steps"):
            dm.set_ar_steps(self._current_ar_steps)
            if getattr(trainer, "is_global_zero", True):
                print(f"[RolloutCurriculum] on_train_start: forcing ar_steps={self._current_ar_steps}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "set_ar_steps"):
            return

        try:
            target_steps = int(pl_module.get_current_rollout_steps())
        except Exception:
            return

        target_steps = min(target_steps, self.max_ar_steps)

        if target_steps != self._current_ar_steps:
            dm.set_ar_steps(target_steps)
            if self.log_on_change and getattr(trainer, "is_global_zero", True):
                print(
                    f"[RolloutCurriculum] global_step={pl_module.global_step} "
                    f"ar_steps: {self._current_ar_steps} -> {target_steps}"
                )
            self._current_ar_steps = target_steps

    # --- checkpointing ---
    def state_dict(self):
        return {"current_ar_steps": self._current_ar_steps}

    def load_state_dict(self, state_dict):
        try:
            self._current_ar_steps = int(state_dict.get("current_ar_steps", 1))
        except Exception:
            self._current_ar_steps = 1
