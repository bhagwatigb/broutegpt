import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from broutegpt import _estimate_text_tokens
from .event_stream import AgentEvent, EventStream
from .router_adapter import CloudFirstRouter


MEMORY_MAX_LINES = 200
MEMORY_MAX_BYTES = 25 * 1024


def compute_dynamic_token_budget(
    model_input_limit: Optional[int],
    fallback_budget: int = 16000,
    reserve_tokens: int = 4096,
) -> int:
    """
    Compute a safe history token budget from a model's input_limit while leaving
    headroom for system prompts, repo maps, turn prompts, and completion tokens.
    """
    if model_input_limit is None:
        return fallback_budget
    try:
        limit = int(model_input_limit)
    except (TypeError, ValueError):
        return fallback_budget
    if limit <= 0:
        return fallback_budget
    if limit <= 8192:
        return max(1024, min(int(limit * 0.65), limit - 1024))
    usable = int(limit * 0.70) - reserve_tokens
    return max(2048, min(usable, 180000))


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
        keep_tail_events: int = 24,
        condense_batch_min: int = 6,
        hook_engine: Optional[Any] = None,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.state_dir / "condenser_state.json"
        self.memory_path = self.state_dir / "MEMORY.md"
        self.keep_tail_events = keep_tail_events
        self.condense_batch_min = condense_batch_min
        self.hook_engine = hook_engine
        self._summaries: Dict[str, Dict[str, Any]] = self._load_cache()

    def read_memory(self) -> str:
        """Return persisted cross-session Auto Memory from .broutegpt/MEMORY.md."""
        if not self.memory_path.exists():
            return ""
        try:
            return self.memory_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _enforce_memory_limits(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        if lines[0].startswith("# "):
            header = [lines[0]]
            body = lines[1:]
        else:
            header = []
            body = lines

        while len(header) + len(body) > MEMORY_MAX_LINES and body:
            body.pop(0)

        candidate = "\n".join(header + body) + "\n"
        while len(candidate.encode("utf-8")) > MEMORY_MAX_BYTES and body:
            body.pop(0)
            candidate = "\n".join(header + body) + "\n"

        raw = candidate.encode("utf-8")
        if len(raw) > MEMORY_MAX_BYTES:
            candidate = raw[:MEMORY_MAX_BYTES].decode("utf-8", errors="ignore").rstrip() + "\n"
        return candidate

    def append_memory(self, entry: str, category: str = "note") -> str:
        """Append a single-line durable note to .broutegpt/MEMORY.md and enforce 200-line / 25 KB caps."""
        clean = " | ".join(part.strip() for part in str(entry).splitlines() if part.strip())
        if not clean:
            return self.read_memory()
        existing = self.read_memory().strip()
        if not existing:
            existing = "# Auto Memory"
        updated = f"{existing}\n- [{category}] {clean}\n"
        bounded = self._enforce_memory_limits(updated)
        self.memory_path.write_text(bounded, encoding="utf-8")
        return bounded

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
        hook_engine: Optional[Any] = None,
    ) -> str:
        if not events:
            return ""
        start_id = events[0].id
        end_id = events[-1].id
        span_key = f"{start_id}:{end_id}"
        if span_key in self._summaries:
            return str(self._summaries[span_key]["summary"])

        active_hooks = hook_engine if hook_engine is not None else self.hook_engine
        hook_ctx = ""
        if active_hooks is not None:
            hook_res = active_hooks.dispatch(
                "PreCompact",
                {"start_id": start_id, "end_id": end_id, "event_count": len(events)},
            )
            if hook_res.blocked:
                blocked_digest = (
                    f"[PreCompact Hook Blocked LLM Condense: {hook_res.block_reason}]\n"
                    + self._deterministic_digest(events)
                )
                return blocked_digest
            if hook_res.additional_context:
                hook_ctx = hook_res.additional_context

        raw_transcript_lines: List[str] = []
        for ev in events:
            model_tag = f" via {ev.model}" if ev.model else ""
            raw_transcript_lines.append(
                f"[Event #{ev.id} | DP {ev.dp_index} | {ev.kind}{model_tag}]\n{_clip_text(ev.content, 1200)}"
            )
        transcript = "\n\n".join(raw_transcript_lines)

        summary_text = ""
        if router is not None:
            hook_note = f"\nAdditional Hook Context:\n{hook_ctx}\n" if hook_ctx else ""
            condense_prompt = (
                "Condense the following autonomous software-engineering event log into a compact, high-signal state summary "
                "so the next coding model can continue seamlessly without losing any technical facts.\n"
                "Include strictly:\n"
                "1. Files created/modified and their roles.\n"
                "2. Commands/tests executed and their outcomes.\n"
                "3. Bugs/errors hit and exact fixes applied.\n"
                "4. Any unresolved blockers or constraints.\n"
                f"{hook_note}\n"
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

        if hook_ctx:
            summary_text = f"[PreCompact additionalContext]\n{hook_ctx}\n\n{summary_text.strip()}"

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
        target_token_budget: Optional[int] = 16000,
        active_dp_index: Optional[int] = None,
        router: Optional[CloudFirstRouter] = None,
        model_input_limit: Optional[int] = None,
        tier: str = "smart",
    ) -> Dict[str, Any]:
        """
        Builds a model-agnostic `history` list (alternating user/assistant messages)
        from `stream` that fits strictly inside the resolved token budget.
        """
        resolved_limit = model_input_limit
        if resolved_limit is None and router is not None and hasattr(router, "get_active_input_limit"):
            try:
                resolved_limit = router.get_active_input_limit(tier=tier)
            except Exception:
                resolved_limit = None

        fallback = int(target_token_budget) if target_token_budget is not None else 16000
        effective_budget = (
            compute_dynamic_token_budget(resolved_limit, fallback_budget=fallback)
            if resolved_limit is not None
            else fallback
        )

        events = stream.events
        if not events:
            return {
                "history": [],
                "estimated_tokens": 0,
                "condensed_count": 0,
                "kept_verbatim_count": 0,
                "target_token_budget": effective_budget,
            }

        pinned_ids: set[int] = set()
        head_events: List[AgentEvent] = []

        goal_ev = next((e for e in events if e.kind == "goal"), None)
        if goal_ev is not None:
            pinned_ids.add(goal_ev.id)
            head_events.append(goal_ev)

        rules_event = stream.latest_by_kind("rules")
        if rules_event is not None and rules_event.id not in pinned_ids:
            pinned_ids.add(rules_event.id)
            head_events.append(rules_event)

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
                raw_msgs.append(self._format_event_turn(h, max_chars=max(per_item_chars, 20000)))
            if middle_summary:
                raw_msgs.append({
                    "role": "user",
                    "content": f"[CONDENSED PRIOR HISTORY]\n{_clip_text(middle_summary, per_item_chars * 2)}",
                })
            recent_cutoff = max(0, len(tail_list) - 4)
            for idx_t, t in enumerate(tail_list):
                item_cap = max(per_item_chars, 28000) if idx_t >= recent_cutoff else per_item_chars
                raw_msgs.append(self._format_event_turn(t, max_chars=item_cap))

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

        if len(middle_slice) >= self.condense_batch_min or raw_tokens > effective_budget:
            if middle_slice:
                condensed_summary = self._summarize_span(middle_slice, router=router)
                condensed_count = len(middle_slice)
        else:
            active_tail = middle_slice + tail_slice

        # Progressive budget-fit loop so even tight-context fallback models never overflow
        per_item_chars = 3500
        compiled = _assemble(head_events, condensed_summary, active_tail, per_item_chars=per_item_chars)
        est_tokens = sum(_estimate_text_tokens(m["content"]) for m in compiled)

        while est_tokens > effective_budget and (len(active_tail) > 2 or per_item_chars > 250):
            if len(active_tail) > 2:
                dropped = active_tail.pop(0)
                middle_slice.append(dropped)
                condensed_summary = self._summarize_span(middle_slice, router=router)
                condensed_count = len(middle_slice)
            else:
                per_item_chars = max(250, int(per_item_chars * 0.65))
            compiled = _assemble(head_events, condensed_summary, active_tail, per_item_chars=per_item_chars)
            est_tokens = sum(_estimate_text_tokens(m["content"]) for m in compiled)

        return {
            "history": compiled,
            "estimated_tokens": est_tokens,
            "condensed_count": condensed_count,
            "kept_verbatim_count": len(head_events) + len(active_tail),
            "target_token_budget": effective_budget,
        }

