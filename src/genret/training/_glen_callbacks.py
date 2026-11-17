"""Shared HF Trainer callbacks for the GLEN training commands.

Kept in a sibling module so the heavy `transformers` import happens lazily
the first time a callback is referenced from a command function, matching
the function-level-import pattern that `cli_glen` uses for the rest of
the transformers API.
"""

from __future__ import annotations

from typing import override

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class FinalStepCallback(TrainerCallback):
    """Force a final save (and eval, when configured) at `state.max_steps`.

    HF Trainer's `save_strategy=steps` does not guarantee a save at the
    exact final step when `max_steps` is not a multiple of `save_steps`.
    This callback forces it. The eval flag is gated on `args.do_eval`
    because `Trainer.get_eval_dataloader` raises `ValueError` when there
    is no eval dataset, which is the path when `eval_strategy="no"`.
    """

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if state.global_step >= state.max_steps:
            control.should_save = True
            if args.do_eval:
                control.should_evaluate = True
