"""Entity association between consecutive observations (greedy nearest)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from open_player.core.types import EntityState


def associate_entities(
    prev: Dict[str, EntityState],
    curr: List[EntityState],
    max_dist: float = 2.0,
) -> Dict[str, Optional[EntityState]]:
    """Greedy nearest-neighbour association by position.

    Returns a mapping from previous entity id to the matched current entity
    (or None when the previous entity was not re-observed).
    """
    matches: Dict[str, Optional[EntityState]] = {pid: None for pid in prev}
    remaining = list(curr)
    # Greedy: repeatedly take the closest (prev, curr) pair within max_dist.
    while remaining:
        best: Optional[tuple[str, int, float]] = None
        for pid, pent in prev.items():
            if matches[pid] is not None:
                continue
            for ci, cent in enumerate(remaining):
                d = float(np.linalg.norm(np.asarray(cent.position, dtype=np.float32) - np.asarray(pent.position, dtype=np.float32)))
                if d <= max_dist and (best is None or d < best[2]):
                    best = (pid, ci, d)
        if best is None:
            break
        pid, ci, _ = best
        matches[pid] = remaining.pop(ci)
    return matches
