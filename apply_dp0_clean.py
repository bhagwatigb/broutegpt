from pathlib import Path

TARGET = Path.home() / "broutegpt" / "broutegpt.py"
src = TARGET.read_text(encoding="utf-8")

PATCHES = [
    (
        'def _is_deep_reasoning_model(model: Dict[str, Any]) -> bool:\n'
        '    mid = model.get("id", "").lower()\n'
        '    return any(k in mid for k in ("70b", "72b", "120b", "235b", "405b", "r1", "qwen3", "qwq", "versatile", "llama-3.3", "llama-4", "maverick", "flash", "27b", "large", "plus"))',
        'def _is_deep_reasoning_model(model: Dict[str, Any]) -> bool:\n'
        '    mid = model.get("id", "").lower()\n'
        '    tokens = set(re.split(r"[-_/:]+", mid))\n'
        '    if tokens & {"0.5b", "1b", "1.5b", "2b", "3b", "4b", "7b", "8b", "9b", "12b", "lite", "mini", "nano", "tiny"}:\n'
        '        return False\n'
        '    if "r1" in tokens:\n'
        '        return True\n'
        '    return any(k in mid for k in ("70b", "72b", "120b", "235b", "405b", "qwen3", "qwq", "versatile", "llama-3.3", "llama-4", "maverick", "flash", "27b", "large", "plus"))',
    ),
    (
        '        # Rank 1 = fastest sufficient model (speed & space optimized), followed by bigger models 1 by 1\n'
        '        ranked_eligible = sorted(\n'
        '            eligible,\n'
        '            key=lambda m: (-m["eff_tps"], m["total_capacity"]),\n'
        '        )',
        '        if out_stats["reasoning_depth"] == "deep":\n'
        '            ranked_eligible = sorted(\n'
        '                eligible,\n'
        '                key=lambda m: (not _is_deep_reasoning_model(m), -m["total_capacity"], -m["eff_tps"]),\n'
        '            )\n'
        '        else:\n'
        '            ranked_eligible = sorted(\n'
        '                eligible,\n'
        '                key=lambda m: (-m["eff_tps"], m["total_capacity"]),\n'
        '            )',
    ),
    (
        '        self.eligible_models: List[Dict[str, Any]] = []\n'
        '        self.last_task_plan: Optional[Dict[str, Any]] = None',
        '        self.eligible_models: List[Dict[str, Any]] = []\n'
        '        self.last_task_plan: Optional[Dict[str, Any]] = None\n'
        '        self._cooldown_until: Dict[tuple[str, str], float] = {}',
    ),
    (
        '        if opt in self.OPTIONS or opt in ("7", "8", "10"):\n'
        '            self.selected_option = opt\n'
        '            print(f"[Router] Non-interactive pipeline selected: Option {opt}")\n'
        '            return',
        '        if opt in self.OPTIONS or opt in ("7", "8", "10", "11", "deep"):\n'
        '            self.selected_option = "deep" if opt in ("11", "deep") else opt\n'
        '            print(f"[Router] Non-interactive pipeline selected: Option {self.selected_option}")\n'
        '            return',
    ),
    (
        '        sources: Any = None,\n'
        '        image_paths: Optional[List[str]] = None,\n'
        '        history: Optional[List[Dict[str, str]]] = None,\n'
        '    ) -> tuple[str, str]:',
        '        sources: Any = None,\n'
        '        image_paths: Optional[List[str]] = None,\n'
        '        history: Optional[List[Dict[str, str]]] = None,\n'
        '        validator: Optional[Any] = None,\n'
        '    ) -> tuple[str, str]:',
    ),
    (
        '            if self.selected_option == "7":\n'
        '                candidates = self._get_ranked_models(top_n=10, vision_only=vision_only)\n'
        '            elif self.selected_option == "8":\n'
        '                candidates = self._get_speed_ranked_models(top_n=10, vision_only=vision_only)',
        '            if self.selected_option == "7":\n'
        '                candidates = self._get_ranked_models(top_n=10, vision_only=vision_only)\n'
        '            elif self.selected_option == "8":\n'
        '                candidates = self._get_speed_ranked_models(top_n=10, vision_only=vision_only)\n'
        '            elif self.selected_option == "deep":\n'
        '                lookup = {(m["provider"], m["id"]): m for m in self._all_discovered_models() if m.get("chat", True) and (not vision_only or m.get("vision"))}\n'
        '                candidates = [lookup[(r["provider"], r["id"])] for r in (self.role_config.get("deep_synthesis") or []) if (r["provider"], r["id"]) in lookup]\n'
        '                seen = {(m["provider"], m["id"]) for m in candidates}\n'
        '                candidates.extend(m for m in self._get_ranked_models(top_n=None, vision_only=vision_only) if (m["provider"], m["id"]) not in seen)',
    ),
    (
        '            errors = []\n'
        '            for idx, model in enumerate(candidates):\n'
        '                mid, provider = model["id"], model["provider"]\n'
        '                total = model.get("input_limit", 0) + model.get("output_limit", 0)\n'
        '                tps = model.get("measured_tps") or model.get("est_tps") or _estimate_model_tps(provider, mid)\n'
        '                effective_max_tokens = max_tokens\n'
        '                if task_plan is not None:\n'
        '                    target_out = max_tokens or task_plan["safe_output_tokens"]\n'
        '                    effective_max_tokens = min(target_out, int(model.get("output_limit") or target_out))\n'
        '                print(f"[Router] Trying ({idx + 1}/{len(candidates)}) {provider}:{mid} [total_limit={total:,}, ~{float(tps):.0f} tok/s]")\n'
        '                try:\n'
        '                    text = self._call_model(provider, mid, prompt, system_prompt, image_path, temperature, effective_max_tokens, image_paths=all_images, history=history)\n'
        '                    return text, f"{provider}:{mid}"\n'
        '                except Exception as exc:\n'
        '                    errors.append(f"{provider}:{mid} -> {exc}")\n'
        '                    print(f"[Fallback] {provider}:{mid} failed ({exc}). Rerouting to next best model...")',
        '            errors = []\n'
        '            now = time.time()\n'
        '            active_candidates = [m for m in candidates if self._cooldown_until.get((m["provider"], m["id"]), 0.0) <= now] or candidates\n'
        '            for idx, model in enumerate(active_candidates):\n'
        '                mid, provider = model["id"], model["provider"]\n'
        '                total = model.get("input_limit", 0) + model.get("output_limit", 0)\n'
        '                tps = model.get("measured_tps") or model.get("est_tps") or _estimate_model_tps(provider, mid)\n'
        '                effective_max_tokens = max_tokens\n'
        '                if task_plan is not None:\n'
        '                    target_out = max_tokens or task_plan["safe_output_tokens"]\n'
        '                    effective_max_tokens = min(target_out, int(model.get("output_limit") or target_out))\n'
        '                print(f"[Router] Trying ({idx + 1}/{len(active_candidates)}) {provider}:{mid} [total_limit={total:,}, ~{float(tps):.0f} tok/s]")\n'
        '                try:\n'
        '                    text = self._call_model(provider, mid, prompt, system_prompt, image_path, temperature, effective_max_tokens, image_paths=all_images, history=history)\n'
        '                    if any(text.count(text[i : i + 25]) > 8 for i in range(0, min(len(text), 4000), 25) if len(text[i : i + 25].strip()) >= 15):\n'
        '                        raise ValueError("Degenerate substring repetition loop detected")\n'
        '                    if validator is not None:\n'
        '                        val_err = validator(text)\n'
        '                        if val_err:\n'
        '                            raise ValueError(f"Validator rejected output: {val_err}")\n'
        '                    return text, f"{provider}:{mid}"\n'
        '                except Exception as exc:\n'
        '                    if "429" in str(exc):\n'
        '                        self._cooldown_until[(provider, mid)] = time.time() + 60.0\n'
        '                    errors.append(f"{provider}:{mid} -> {exc}")\n'
        '                    print(f"[Fallback] {provider}:{mid} failed ({exc}). Rerouting to next best model...")',
    ),
    (
        '            out_parts = resp.json()["candidates"][0]["content"]["parts"]\n'
        '            text = "".join(p.get("text", "") for p in out_parts)\n'
        '            if not text:\n'
        '                raise ValueError("Empty Gemini response")\n'
        '            return text',
        '            cand0 = resp.json()["candidates"][0]\n'
        '            out_parts = cand0["content"]["parts"]\n'
        '            text = "".join(p.get("text", "") for p in out_parts)\n'
        '            if not text:\n'
        '                raise ValueError("Empty Gemini response")\n'
        '            if cand0.get("finishReason") == "MAX_TOKENS" and text.count("```") % 2 == 1:\n'
        '                raise ValueError("Gemini output truncated mid-code-block (MAX_TOKENS)")\n'
        '            return text',
    ),
    (
        '        resp = requests.post(url, headers={"Authorization": f"Bearer {os.getenv(env_var)}", "Content-Type": "application/json"}, json=payload, timeout=self.timeout)\n'
        '        resp.raise_for_status()\n'
        '        text = resp.json()["choices"][0]["message"]["content"]\n'
        '        if not text:\n'
        '            raise ValueError("Empty response")\n'
        '        return text',
        '        resp = requests.post(url, headers={"Authorization": f"Bearer {os.getenv(env_var)}", "Content-Type": "application/json"}, json=payload, timeout=self.timeout)\n'
        '        resp.raise_for_status()\n'
        '        choice0 = resp.json()["choices"][0]\n'
        '        text = choice0["message"]["content"]\n'
        '        if not text:\n'
        '            raise ValueError("Empty response")\n'
        '        if choice0.get("finish_reason") == "length" and text.count("```") % 2 == 1:\n'
        '            raise ValueError("Output truncated mid-code-block (finish_reason=length)")\n'
        '        return text',
    ),
]

for idx, (old, new) in enumerate(PATCHES, 1):
    assert old in src, f"Patch {idx} anchor not found!"
    src = src.replace(old, new, 1)

TARGET.write_text(src, encoding="utf-8")
print("All 9 patch blocks applied cleanly.")

