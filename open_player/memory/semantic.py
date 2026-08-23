"""SemanticMemory: symbolic facts (subject, predicate, object) counters."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SemanticMemory:
    """Simple fact store; a learned semantic memory can replace it later."""

    def __init__(self) -> None:
        self._facts: Dict[Tuple[str, str, str], float] = {}

    def observe(self, subject: str, predicate: str, obj: str, weight: float = 1.0) -> None:
        key = (str(subject), str(predicate), str(obj))
        self._facts[key] = self._facts.get(key, 0.0) + float(weight)

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple[Tuple[str, str, str], float]]:
        out = []
        for (s, p, o), w in self._facts.items():
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if obj is not None and o != obj:
                continue
            out.append(((s, p, o), w))
        return sorted(out, key=lambda item: item[1], reverse=True)

    def stats(self) -> Dict[str, Any]:
        return {"num_facts": len(self._facts)}
