import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_PERMISSION_SETTINGS: Dict[str, Any] = {
    "permissions": {
        "allow": [
            "read_file",
            "list_dir",
            "grep_search",
            "glob_files",
            "dp_complete",
            "write_file(*)",
            "patch_file(*)",
            "run_shell(python3 *)",
            "run_shell(pytest *)",
            "run_shell(git status*)",
            "run_shell(git diff*)",
        ],
        "ask": [
            "run_shell(*)",
        ],
        "deny": [
            "run_shell(*sudo *)",
            "run_shell(*rm -rf /*)",
            "run_shell(*rm -rf ~*)",
            "run_shell(*curl *| sh*)",
            "run_shell(*curl *| bash*)",
            "run_shell(*wget *| sh*)",
            "write_file(.env*)",
            "patch_file(.env*)",
            "write_file(.git/*)",
            "patch_file(.git/*)",
            "write_file(.broutegpt/*)",
            "patch_file(.broutegpt/*)",
        ],
    }
}


def split_shell_commands(command: str) -> List[str]:
    """
    Split a compound shell command on unquoted &&, ||, ;, and | operators
    so every chained sub-command is evaluated independently against policy rules.
    """
    parts: List[str] = []
    buf: List[str] = []
    in_quote: Optional[str] = None
    escaped = False
    i = 0
    n = len(command)

    while i < n:
        ch = command[i]
        nxt = command[i + 1] if i + 1 < n else ""

        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\" and in_quote != "'":
            buf.append(ch)
            escaped = True
            i += 1
            continue

        if in_quote is not None:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            i += 1
            continue

        if (ch == "&" and nxt == "&") or (ch == "|" and nxt == "|"):
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf.clear()
            i += 2
            continue

        if ch in (";", "|"):
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf.clear()
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


@dataclass
class PermissionDecision:
    decision: str  # allow | ask | deny
    allowed: bool
    reason: str
    matched_rule: str = ""
    guard_verdict: str = ""


class PermissionPolicy:
    """
    Declarative allow/ask/deny policy engine backed by `<project_root>/.broutegpt/settings.json`
    (with optional `~/.broutegpt/settings.json` merge) and an optional zero-cost fast-tier
    LLM Auto-Guard safety classifier for unapproved shell commands.
    """

    def __init__(
        self,
        project_dir: str | Path,
        router: Optional[Any] = None,
        auto_guard: bool = False,
    ) -> None:
        self.project_root = Path(project_dir).expanduser().resolve()
        self.state_dir = self.project_root / ".broutegpt"
        self.settings_path = self.state_dir / "settings.json"
        self.user_settings_path = Path.home() / ".broutegpt" / "settings.json"
        self.router = router
        self.auto_guard = bool(auto_guard)

    def load_rules(self) -> Dict[str, List[str]]:
        allow: List[str] = list(DEFAULT_PERMISSION_SETTINGS["permissions"]["allow"])
        ask: List[str] = list(DEFAULT_PERMISSION_SETTINGS["permissions"]["ask"])
        deny: List[str] = list(DEFAULT_PERMISSION_SETTINGS["permissions"]["deny"])

        has_custom = False
        merged_allow: List[str] = []
        merged_ask: List[str] = []
        merged_deny: List[str] = []

        for cfg_path in (self.user_settings_path, self.settings_path):
            if not cfg_path.is_file():
                continue
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            perms = raw.get("permissions") if isinstance(raw.get("permissions"), dict) else raw
            if not isinstance(perms, dict):
                continue
            for key, target in (("allow", merged_allow), ("ask", merged_ask), ("deny", merged_deny)):
                vals = perms.get(key)
                if isinstance(vals, list):
                    has_custom = True
                    for item in vals:
                        s = str(item).strip()
                        if s and s not in target:
                            target.append(s)

        if has_custom:
            # Always keep built-in deny protections merged with project deny rules
            for d in deny:
                if d not in merged_deny:
                    merged_deny.append(d)
            return {"allow": merged_allow, "ask": merged_ask, "deny": merged_deny}

        return {"allow": allow, "ask": ask, "deny": deny}

    def _parse_rule(self, rule: str) -> Tuple[str, Optional[str]]:
        m = re.match(r"^([a-zA-Z0-9_-]+)\((.*)\)$", rule.strip())
        if m:
            return m.group(1).strip().lower(), m.group(2).strip()
        return rule.strip().lower(), None

    def _rule_matches_target(self, rule: str, action_name: str, arg_value: str) -> bool:
        r_tool, r_arg = self._parse_rule(rule)
        if not fnmatch.fnmatch(action_name.lower(), r_tool):
            return False
        if r_arg is None or r_arg == "*":
            return True
        return fnmatch.fnmatch(arg_value, r_arg)

    def _match_single(
        self,
        action_name: str,
        arg_value: str,
        rules: Dict[str, List[str]],
    ) -> Tuple[str, str]:
        # Precedence: deny -> allow -> ask -> default ask (for run_shell) or allow (for non-shell)
        for rule in rules.get("deny", []):
            if self._rule_matches_target(rule, action_name, arg_value):
                return "deny", rule
        for rule in rules.get("allow", []):
            if self._rule_matches_target(rule, action_name, arg_value):
                return "allow", rule
        for rule in rules.get("ask", []):
            if self._rule_matches_target(rule, action_name, arg_value):
                return "ask", rule
        if action_name == "run_shell":
            return "ask", "implicit:run_shell"
        return "allow", "implicit:default"

    def _classify_shell_with_auto_guard(self, command: str) -> Tuple[bool, str]:
        if self.router is None:
            return False, "Auto-Guard enabled but no router configured; denying unapproved shell command."

        guard_prompt = (
            "You are a strict security classifier for an autonomous coding agent running in a local project directory.\n"
            "Evaluate whether the following shell command is SAFE to run inside the project workspace without human approval.\n"
            "Block commands that exfiltrate secrets/env vars, modify system/user dotfiles, delete outside the workspace, "
            "pipe remote scripts into a shell, or alter git remotes/credentials.\n\n"
            f"COMMAND:\n{command}\n\n"
            'Reply ONLY with valid JSON: {"safe": true|false, "reason": "<max 15 words>"}'
        )
        try:
            raw, used_model = self.router.query_autonomous(
                prompt=guard_prompt,
                system_prompt="You are a binary command safety classifier. Output ONLY valid JSON.",
                tier="speed",
                temperature=0.0,
                max_tokens=160,
            )
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return False, f"Auto-Guard ({used_model}) returned unparseable output; denying."
            data = json.loads(m.group(0))
            is_safe = bool(data.get("safe") is True)
            reason = str(data.get("reason") or ("Approved by Auto-Guard" if is_safe else "Rejected by Auto-Guard")).strip()
            return is_safe, f"{used_model}: {reason}"
        except Exception as exc:
            return False, f"Auto-Guard classifier error ({exc}); failing closed."

    def evaluate(self, action_obj: Dict[str, Any]) -> PermissionDecision:
        action_name = str(action_obj.get("action") or "").strip().lower()
        rules = self.load_rules()

        if action_name == "run_shell":
            raw_cmd = str(action_obj.get("command") or "").strip()
            sub_cmds = split_shell_commands(raw_cmd) or [raw_cmd]

            # Check whole command against deny rules first
            whole_dec, whole_rule = self._match_single("run_shell", raw_cmd, rules)
            if whole_dec == "deny":
                return PermissionDecision(
                    decision="deny",
                    allowed=False,
                    reason=f"Shell command denied by rule '{whole_rule}': {raw_cmd}",
                    matched_rule=whole_rule,
                )

            ask_segments: List[Tuple[str, str]] = []
            for seg in sub_cmds:
                seg_dec, seg_rule = self._match_single("run_shell", seg, rules)
                if seg_dec == "deny":
                    return PermissionDecision(
                        decision="deny",
                        allowed=False,
                        reason=f"Shell sub-command '{seg}' denied by rule '{seg_rule}'",
                        matched_rule=seg_rule,
                    )
                if seg_dec == "ask":
                    ask_segments.append((seg, seg_rule))

            if not ask_segments:
                return PermissionDecision(
                    decision="allow",
                    allowed=True,
                    reason="All shell sub-commands matched allow policy.",
                    matched_rule="allow",
                )

            if self.auto_guard:
                is_safe, verdict = self._classify_shell_with_auto_guard(raw_cmd)
                if is_safe:
                    return PermissionDecision(
                        decision="allow",
                        allowed=True,
                        reason=f"Approved by Auto-Guard ({verdict})",
                        matched_rule=ask_segments[0][1],
                        guard_verdict=verdict,
                    )
                return PermissionDecision(
                    decision="deny",
                    allowed=False,
                    reason=f"Blocked by Auto-Guard ({verdict})",
                    matched_rule=ask_segments[0][1],
                    guard_verdict=verdict,
                )

            # Autonomous non-interactive fallback when auto_guard=False: allow ask-tier commands
            # that survived deny rules while recording the matched ask rule.
            return PermissionDecision(
                decision="ask",
                allowed=True,
                reason=f"Allowed in autonomous mode (matched ask rule '{ask_segments[0][1]}')",
                matched_rule=ask_segments[0][1],
            )

        arg_val = str(action_obj.get("path") or action_obj.get("pattern") or "")
        dec, matched = self._match_single(action_name, arg_val, rules)
        if dec == "deny":
            return PermissionDecision(
                decision="deny",
                allowed=False,
                reason=f"Action '{action_name}({arg_val})' denied by rule '{matched}'",
                matched_rule=matched,
            )
        return PermissionDecision(
            decision=dec,
            allowed=True,
            reason=f"Action '{action_name}' permitted ({dec}: {matched})",
            matched_rule=matched,
        )

