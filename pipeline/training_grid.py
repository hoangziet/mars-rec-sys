"""
pipeline/training_grid.py
=========================
Grid-construction helpers shared by RQ2/RQ3/RQ4.

Routines
--------
- enforce_final_grid()  — preserve the configured training grid while
                            copying train_kwargs for final-stage runs.
"""


def enforce_final_grid(train_kwargs: dict) -> dict:
    """Return a copy of train_kwargs for final-stage runs.

    The final studies must follow the explicit model config. In particular,
    ``batch_size`` must come from the config file rather than a hidden runtime
    override, so this helper intentionally preserves all values verbatim.
    """
    return dict(train_kwargs)
