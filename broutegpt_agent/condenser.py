import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from broutegpt import _estimate_text_tokens
from .event_stream import AgentEvent, EventStream
from .router_adapter import CloudFirstRouter


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 80:
        return text[:max_chars]
    half = max_chars // 2 - 20
    return f"{text[:half]}\n...[clipped {len(text) - max_chars} chars]...\n{text[-half:]}"


class RollingCondenser:
    """
    OpenHands-inspired model-agnostic context condenser:
      - Pinned Head: Always keeps the master 'goal' and latest 'plan' verbatim.
      - Active DP Anchor: Keeps current 'dp_start' blueprint + latest 'dp_verified' summary.
      - Condensed Middle: Summarizes older intermediate events into a cached factual
        memory block when token/event thresholds are exceeded.
      - Verbatim Tail: Preserves the most recent action/observation turns intact so
        any active model sees exact code diffs, tracebacks, and verification output.
    """

    def __init__(
        self,
        state_dir: str | Path,
        keep_tail_events: int = 8,
        condense_batch_min: int = 4,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.state_dir / "condenser_state.json"
        self.keep_tail_events = keep_tail_events
        self.condense_batch_min = condense_batch_min
        self._summaries: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self._summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def invalidate_after(self, max_event_id: int) -> None:
        """Drops any cached summary spanning events beyond `max_event_id` after rollback."""
        stale_keys = [
            key for key, entry in self._summaries.items()
            if int(entry.get("end_id", 0)) > max_event_id
        ]
        for k in stale_keys:
            self._summaries.pop(k, None)
        if stale_keys:
            self._save_cache()

    def _deterministic_digest(self, events: List[AgentEvent]) -> str:
        lines: List[str] = []
        for ev in events:
            model_tag = f" [{ev.model}]" if ev.model else ""
            snippet = _clip_text(ev.content.replace("\n", " "), 220)
            lines.append(f"- #{ev.id} (DP {ev.dp_index}, {ev.kind}{model_tag}): {snippet}")
        return "Condensed Execution History:\n" + "\n".join(lines)

    def _summarize_span(
        self,
        events: List[AgentEvent],
        router: Optional[CloudFirstRouter] = None,
    ) -> str:
        if not events:
            return ""
        start_id = events[0].id
        end_id = events[-1].id
        span_key = f"{start_id}:{end_id}"
        if span_key in self._summaries:
            return str(self._summaries[span_key]["summary"])

        raw_transcript_lines: List[str] = []
        for ev in events:
            model_tag = f" via {ev.model}" if ev.model else ""
            raw_transcript_lines.append(
                f"[Event #{ev.id} | DP {ev.dp_index} | {ev.kind}{model_tag}]\n{_clip_text(ev.content, 1200)}"
            )
        transcript = "\n\n".join(raw_transcript_lines)

        summary_text = ""
        if router is not None:
            condense_prompt = (
                "Condense the following autonomous software-engineering event log into a compact, high-signal state summary "
                "so the next coding model can continue seamlessly without losing any technical facts.\n"
                "Include strictly:\n"
                "1. Files created/modified and their roles.\n"
                "2. Commands/tests executed and their outcomes.\n"
                "3. Bugs/errors hit and exact fixes applied.\n"
                "4. Any unresolved blockers or constraints.\n\n"
                f"EVENTS #{start_id} TO #{end_id}:\n{transcript}"
            )
            try:
                summary_text, _ = router.query_autonomous(
                    prompt=condense_prompt,
                    system_prompt="You are a concise technical state condenser. Output factual bullet points only.",
                    tier="speed",
                    temperature=0.0,
                    max_tokens=900,
                )
            except Exception:
                summary_text = ""

        if not summary_text.strip():
            summary_text = self._deterministic_digest(events)

        self._summaries[span_key] = {
            "start_id": start_id,
            "end_id": end_id,
            "event_count": len(events),
            "summary": summary_text.strip(),
        }
        self._save_cache()
        return summary_text.strip()

    def _format_event_turn(self, ev: AgentEvent, max_chars: int = 3500) -> Dict[str, str]:
        role = "assistant" if ev.kind == "action" else "user"
        header = f"[Event #{ev.id} | DP {ev.dp_index} | {ev.kind.upper()}" + (f" | {ev.model}]" if ev.model else "]")
        body = _clip_text(ev.content, max_chars)
        return {"role": role, "content": f"{header}\n{body}"}

    def compile_history(
        self,
        stream: EventStream,
        target_token_budget: int = 16000,
        active_dp_index: Optional[int] = None,
        router: Optional[CloudFirstRouter] = None,
    ) -> Dict[str, Any]:
        """
        Builds a model-agnostic `history` list (alternating user/assistant messages)
        from `stream` that fits strictly inside `target_token_budget`.
        """
        events = stream.events
        if not events:
            return {
                "history": [],
                "estimated_tokens": 0,
                "condensed_count": 0,
                "kept_verbatim_count": 0,
            }

        pinned_ids: set[int] = set()
        head_events: List[AgentEvent] = []

        goal_ev = next((e for e in events if e.kind == "goal"), None)
        if goal_ev is not None:
            pinned_ids.add(goal_ev.id)
            head_events.append(goal_ev)

        plan_ev = stream.latest_by_kind("plan")
        if plan_ev is not None and plan_ev.id not in pinned_ids:
            pinned_ids.add(plan_ev.id)
            head_events.append(plan_ev)

        if active_dp_index is not None:
            dp_start_ev = stream.latest_by_kind("dp_start", dp_index=active_dp_index)
            if dp_start_ev is not None and dp_start_ev.id not in pinned_ids:
                pinned_ids.add(dp_start_ev.id)
                head_events.append(dp_start_ev)

        latest_verified = stream.latest_by_kind("dp_verified")
        if latest_verified is not None and latest_verified.id not in pinned_ids:
            pinned_ids.add(latest_verified.id)
            head_events.append(latest_verified)

        non_pinned = [e for e in events if e.id not in pinned_ids]
        tail_slice = non_pinned[-self.keep_tail_events :] if self.keep_tail_events > 0 else []
        middle_slice = non_pinned[: max(0, len(non_pinned) - len(tail_slice))]

        # Check raw size before condensing
        def _assemble(
            head_list: List[AgentEvent],
            middle_summary: str,
            tail_list: List[AgentEvent],
            per_item_chars: int,
        ) -> List[Dict[str, str]]:
            raw_msgs: List[Dict[str, str]] = []
            for h in head_list:
                raw_msgs.append(self._format_event_turn(h, max_chars=per_item_chars))
            if middle_summary:
                raw_msgs.append({
                    "role": "user",
                    "content": f"[CONDENSED PRIOR HISTORY]\n{_clip_text(middle_summary, per_item_chars * 2)}",
                })
            for t in tail_list:
                raw_msgs.append(self._format_event_turn(t, max_chars=per_item_chars))

            # Coalesce consecutive same-role messages so every provider API accepts the sequence cleanly
            coalesced: List[Dict[str, str]] = []
            for msg in raw_msgs:
                if coalesced and coalesced[-1]["role"] == msg["role"]:
                    coalesced[-1]["content"] += "\n\n" + msg["content"]
                else:
                    coalesced.append(dict(msg))
            return coalesced

        raw_candidate = _assemble(head_events, "", middle_slice + tail_slice, per_item_chars=3500)
        raw_tokens = sum(_estimate_text_tokens(m["content"]) for m in raw_candidate)

        condensed_summary = ""
        condensed_count = 0
        active_tail = list(tail_slice)

        if len(middle_slice) >= self.condense_batch_min or raw_tokens > target_token_budget:
            if middle_slice:
                condensed_summary = self._summarize_span(middle_slice, router=router)
                condensed_count = len(middle_slice)
        else:
            active_tail = middle_slice + tail_slice

        # Progressive budget-fit loop so even tight-context fallback models never overflow
        per_item_chars = 3500
        compiled = _assemble(head_events, condensed_summary, active_tail, per_item_chars=per_item_chars)
        est_tokens = sum(_estimate_text_tokens(m["content"]) for m in compiled)

        while est_tokens > target_token_budget and (len(active_tail) > 2 or per_item_chars > 500):
            if len(active_tail) > 2:
                dropped = active_tail.pop(0)
                middle_slice.append(dropped)
                condensed_summary = self._summarize_span(middle_slice, router=router)
                condensed_count = len(middle_slice)
            else:
                per_item_chars = max(500, int(per_item_chars * 0.65))
            compiled = _assemble(head_events, condensed_summary, active_tail, per_item_chars=per_item_chars)
            est_tokens = sum(_estimate_text_tokens(m["content"]) for m in compiled)

        return {
            "history": compiled,
            "estimated_tokens": est_tokens,
            "condensed_count": condensed_count,
            "kept_verbatim_count": len(head_events) + len(active_tail),
        }

