"""
pipeline/training_grid.py
=========================
Grid-construction helpers shared by RQ2/RQ3/RQ4.

Routines
--------
- enforce_final_grid()  — force batch_size=128 and other grid constraints
                            shared by all final-stage runs.
"""


def enforce_final_grid(train_kwargs: dict) -> dict:
    """Force grid-constraint values into a copy of train_kwargs.

    RQ2/RQ3/RQ4 must all use the same batch_size=128 (matching the
    final ablation grid) regardless of any other value in train_kwargs.
    """
    out = dict(train_kwargs)
    out["batch_size"] = 128
    return out
