"""Event detection: continuous change -> boundary detection -> events.

Phase 0 ships a heuristic detector (no trained event model).  The interface
is kept so a learned ChangeEncoder / BoundaryDetector can replace it later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from open_player.core.types import Event, EventType, WorldState


class ChangeDetector(ABC):
    """Interface: detect events between two consecutive WorldStates."""

    @abstractmethod
    def detect(self, prev: Optional[WorldState], curr: WorldState, env_info: Dict[str, Any], t: int) -> List[Event]:
        ...


class HeuristicEventDetector(ChangeDetector):
    """Rule-based primitive event detector for Phase 0."""

    def __init__(self, move_threshold: float = 0.5, approach_threshold: float = 0.5, threat_delta: float = 0.1) -> None:
        self.move_threshold = float(move_threshold)
        self.approach_threshold = float(approach_threshold)
        self.threat_delta = float(threat_delta)
        self._counter = 0

    def detect(self, prev: Optional[WorldState], curr: WorldState, env_info: Dict[str, Any], t: int) -> List[Event]:
        events: List[Event] = []
        if prev is None:
            for e in curr.entity_states(0):
                if e.semantic_type != "empty":
                    events.append(self._event(t, EventType.APPEAR.value, [e.entity_id], e.position, 1.0))
            return events

        prev_ents = {e.entity_id: e for e in prev.entity_states(0) if e.semantic_type != "empty"}
        curr_ents = {e.entity_id: e for e in curr.entity_states(0) if e.semantic_type != "empty"}
        curr_beliefs = {b.entity_id: b for b in curr.belief_states(0)}

        def _add(etype: str, ids: List[str], loc: Optional[np.ndarray], conf: float, **meta: Any) -> None:
            events.append(self._event(t, etype, ids, loc, conf, **meta))

        # appear / disappear
        for eid, ent in curr_ents.items():
            if eid not in prev_ents:
                _add(EventType.APPEAR.value, [eid], ent.position, 0.9)
        for eid, ent in prev_ents.items():
            b = curr_beliefs.get(eid)
            if eid not in curr_ents or (b is not None and b.existence_probability < 0.5):
                _add(EventType.DISAPPEAR.value, [eid], ent.position, 0.9)

        # move / approach
        player_prev = prev_ents.get("player")
        player_curr = curr_ents.get("player")
        for eid, ent in curr_ents.items():
            pe = prev_ents.get(eid)
            if pe is None or player_curr is None:
                continue
            delta = float(np.linalg.norm(np.asarray(ent.position) - np.asarray(pe.position)))
            if delta >= self.move_threshold:
                if ent.semantic_type == "enemy" and player_prev is not None:
                    d_prev = float(np.linalg.norm(np.asarray(pe.position) - np.asarray(player_prev.position)))
                    d_curr = float(np.linalg.norm(np.asarray(ent.position) - np.asarray(player_curr.position)))
                    if d_prev - d_curr >= self.approach_threshold:
                        _add(EventType.APPROACH.value, [eid], ent.position, 0.8, distance_delta=d_prev - d_curr)
                    else:
                        _add(EventType.MOVE.value, [eid], ent.position, 0.8, distance=delta)
                else:
                    _add(EventType.MOVE.value, [eid], ent.position, 0.8, distance=delta)

        # collision / damage / death / collect (env hints)
        if env_info.get("damage_from"):
            _add(EventType.COLLISION.value, ["player", str(env_info["damage_from"])], player_curr.position if player_curr else None, 0.9)
        if env_info.get("hp_delta", 0) < 0:
            _add(EventType.DAMAGE.value, ["player"], player_curr.position if player_curr else None, 0.95, hp=env_info.get("hp"))
        if env_info.get("death") or env_info.get("hp", 1) <= 0:
            _add(EventType.DEATH.value, ["player"], player_curr.position if player_curr else None, 1.0)
        if env_info.get("collected_this_step"):
            _add(EventType.COLLECT.value, ["player", str(env_info.get("collected_entity", "resource"))], player_curr.position if player_curr else None, 1.0)

        # enter / exit region (quadrants of the grid)
        if player_prev is not None and player_curr is not None:
            q_prev = self._quadrant(player_prev.position)
            q_curr = self._quadrant(player_curr.position)
            if q_prev != q_curr:
                _add(EventType.EXIT.value, ["player"], player_prev.position, 0.7, region=q_prev)
                _add(EventType.ENTER.value, ["player"], player_curr.position, 0.7, region=q_curr)

        # threat increase / decrease
        t_prev = float(env_info.get("prev_threat", env_info.get("threat_level", 0.0)))
        t_curr = float(env_info.get("threat_level", 0.0))
        if t_curr - t_prev >= self.threat_delta:
            _add(EventType.THREAT_INCREASE.value, ["player"], player_curr.position if player_curr else None, 0.7, delta=t_curr - t_prev)
        elif t_prev - t_curr >= self.threat_delta:
            _add(EventType.THREAT_DECREASE.value, ["player"], player_curr.position if player_curr else None, 0.7, delta=t_prev - t_curr)

        # fallback state change
        if not events:
            gp = prev.global_t[0].detach().cpu().numpy()
            gc = curr.global_t[0].detach().cpu().numpy()
            if float(np.abs(gp - gc).mean()) > 0.2:
                _add(EventType.STATE_CHANGE.value, ["world"], player_curr.position if player_curr else None, 0.5)

        return events

    @staticmethod
    def _quadrant(pos: np.ndarray) -> Tuple[int, int]:
        p = np.asarray(pos, dtype=np.float32)
        return (int(p[0] >= 4.0), int(p[1] >= 4.0))

    def _event(self, t: int, etype: str, ids: List[str], loc: Optional[np.ndarray], conf: float, **meta: Any) -> Event:
        self._counter += 1
        return Event(
            event_id=f"e{t}-{self._counter}",
            type=etype,
            timestamp=int(t),
            entities=list(ids),
            location=None if loc is None else np.asarray(loc, dtype=np.float32).copy(),
            confidence=float(conf),
            metadata=dict(meta),
        )


class HybridEventDetector(ChangeDetector):
    """Heuristic events + learned change/boundary signal (Phase 1).

    The heuristic detector is untouched; the learned predictor (z_t, action,
    z_t1 -> change prob / boundary prob) is blended into every event's
    confidence and stored in the event metadata.
    """

    def __init__(
        self,
        heuristic: HeuristicEventDetector,
        world_model: Optional[Any] = None,
        conf_blend: float = 0.5,
        device: Any = "cpu",
    ) -> None:
        self.heuristic = heuristic
        self.world_model = world_model
        self.conf_blend = float(conf_blend)
        self.device = device

    @property
    def predictor_available(self) -> bool:
        return self.world_model is not None and getattr(self.world_model, "change_predictor", None) is not None

    def detect(self, prev: Optional[WorldState], curr: WorldState, env_info: Dict[str, Any], t: int) -> List[Event]:
        events = self.heuristic.detect(prev, curr, env_info, t)
        if not events or not self.predictor_available or prev is None:
            return events
        try:
            import torch
            action = int(env_info.get("action", 0))
            with torch.no_grad():
                z_t = self.world_model.representation(prev).z
                z_t1 = self.world_model.representation(curr).z
                a = torch.full((z_t.shape[0],), action, dtype=torch.long, device=self.device)
                logits, boundary = self.world_model.change_predictor(z_t, a, z_t1)
                prob = float(torch.sigmoid(logits[:, 1]).mean())
                bprob = float(boundary.mean())
        except Exception:  # pragma: no cover - defensive
            return events
        for e in events:
            e.confidence = float(self.conf_blend * prob + (1.0 - self.conf_blend) * e.confidence)
            e.metadata["learned_change_prob"] = prob
            e.metadata["boundary_prob"] = bprob
        return events

    def boundary_score(self, event: Event) -> float:
        return float(event.metadata.get("boundary_prob", 0.0))


class LearnedEventEmitter(ChangeDetector):
    """Strict-RGB event source: learned change signal only.

    No heuristic evidence, no GT positions, no env info.  Emits a single
    state-change event per step when the learned change probability exceeds
    the threshold (the Phase 1.5 strict mode's event stream).
    """

    def __init__(self, world_model: Optional[Any] = None, threshold: float = 0.5, device: Any = "cpu") -> None:
        self.world_model = world_model
        self.threshold = float(threshold)
        self.device = device
        self._counter = 0

    @property
    def available(self) -> bool:
        return self.world_model is not None and getattr(self.world_model, "change_predictor", None) is not None

    def detect(self, prev: Optional[WorldState], curr: WorldState, env_info: Dict[str, Any], t: int) -> List[Event]:
        if prev is None or not self.available:
            return []
        try:
            import torch
            action = int(env_info.get("action", 0))
            with torch.no_grad():
                z_t = self.world_model.representation(prev).z
                z_t1 = self.world_model.representation(curr).z
                a = torch.full((z_t.shape[0],), action, dtype=torch.long, device=self.device)
                logits, boundary = self.world_model.change_predictor(z_t, a, z_t1)
                prob = float(torch.sigmoid(logits[:, 1]).mean())
                bprob = float(boundary.mean())
        except Exception:  # pragma: no cover - defensive
            return []
        if prob < self.threshold:
            return []
        self._counter += 1
        return [Event(
            event_id=f"learned-e{t}-{self._counter}",
            type=EventType.STATE_CHANGE.value,
            timestamp=int(t),
            entities=["world"],
            confidence=prob,
            metadata={"learned_change_prob": prob, "boundary_prob": bprob, "source": "learned"},
        )]
