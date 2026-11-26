import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentEvent:
    id: int
    timestamp: float
    dp_index: int
    kind: str  # goal | plan | dp_start | action | observation | dp_verified | rollback | note
    content: str
    model: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentEvent":
        return cls(
            id=int(data["id"]),
            timestamp=float(data.get("timestamp", 0.0)),
            dp_index=int(data.get("dp_index", 0)),
            kind=str(data.get("kind", "note")),
            content=str(data.get("content", "")),
            model=data.get("model"),
            meta=dict(data.get("meta") or {}),
        )


class EventStream:
    """
    Append-only JSONL event log stored inside <project_dir>/.broutegpt/events.jsonl.
    Preserves full auditability across model switches and supports atomic rewind
    when rolling back hallucinated edits.
    """

    def __init__(self, project_dir: str | Path) -> None:
        self.project_root = Path(project_dir).expanduser().resolve()
        self.state_dir = self.project_root / ".broutegpt"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.state_dir / "events.jsonl"
        self._events: List[AgentEvent] = []
        self._load()

    def _load(self) -> None:
        self._events.clear()
        if not self.log_path.exists():
            return
        for raw_line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                self._events.append(AgentEvent.from_dict(json.loads(line)))
            except Exception:
                continue

    @property
    def events(self) -> List[AgentEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def append(
        self,
        kind: str,
        content: str,
        dp_index: int = 0,
        model: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        next_id = (self._events[-1].id + 1) if self._events else 1
        ev = AgentEvent(
            id=next_id,
            timestamp=time.time(),
            dp_index=dp_index,
            kind=kind,
            content=content,
            model=model,
            meta=dict(meta or {}),
        )
        self._events.append(ev)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        return ev

    def get(self, event_id: int) -> Optional[AgentEvent]:
        for ev in self._events:
            if ev.id == event_id:
                return ev
        return None

    def latest_by_kind(self, kind: str, dp_index: Optional[int] = None) -> Optional[AgentEvent]:
        for ev in reversed(self._events):
            if ev.kind == kind and (dp_index is None or ev.dp_index == dp_index):
                return ev
        return None

    def rewind_to(self, max_event_id: int, reason: str = "Rollback executed") -> int:
        """
        Truncates events after `max_event_id` and records a rollback marker event.
        Returns the number of discarded events.
        """
        kept = [ev for ev in self._events if ev.id <= max_event_id]
        discarded = len(self._events) - len(kept)
        self._events = kept
        with self.log_path.open("w", encoding="utf-8") as fh:
            for ev in self._events:
                fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        last_dp = kept[-1].dp_index if kept else 0
        self.append(
            kind="rollback",
            content=f"Rewound event stream to event #{max_event_id} (discarded {discarded} event(s)). Reason: {reason}",
            dp_index=last_dp,
            meta={"rewound_to_id": max_event_id, "discarded_count": discarded},
        )
        return discarded

