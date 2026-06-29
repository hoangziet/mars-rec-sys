from __future__ import annotations

import math

WATCH_PAD_ID = 0
WATCH_MASK_ID = 1


def engagement_to_watch_bin(value: float, num_bins: int) -> int:
    """Map engagement score in [0, 1] to a watch-bin ID.

    Bin IDs are offset by 2: 0 = WATCH_PAD, 1 = WATCH_MASK,
    2..(2+num_bins-1) = engagement bins.
    """
    clipped = min(max(float(value), 0.0), 1.0)
    if clipped == 1.0:
        bucket = num_bins - 1
    else:
        bucket = int(math.floor(clipped * num_bins))
    return 2 + bucket
