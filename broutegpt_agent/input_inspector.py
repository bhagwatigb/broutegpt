import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from broutegpt import FreeLLMRouter, _load_sources_payload


class InputInspector:
    """
    Inspects user-supplied input files, folders, or URLs to extract both
    compact structural schemas (for DP planning prompts) and full payloads
    (for data-processing DPs).
    """

    def __init__(self, router: Optional[FreeLLMRouter] = None) -> None:
        self.router = router

    def _inspect_single_file(self, fpath: Path) -> Dict[str, Any]:
        ext = fpath.suffix.lower()
        info: Dict[str, Any] = {
            "name": fpath.name,
            "path": str(fpath.resolve()),
            "ext": ext,
            "size_bytes": fpath.stat().st_size,
            "structure": "",
            "sample": "",
        }

        try:
            if ext in (".csv", ".tsv"):
                delim = "\t" if ext == ".tsv" else ","
                with fpath.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                    reader = list(csv.reader(fh, delimiter=delim))
                row_count = len(reader)
                headers = reader[0] if reader else []
                sample_rows = reader[1:4]
                info["structure"] = f"{max(row_count - 1, 0)} data rows, {len(headers)} columns: {headers}"
                info["sample"] = "\n".join(str(r) for r in sample_rows)
                return info

            if ext == ".json":
                data = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, dict):
                    info["structure"] = f"JSON object with {len(data)} keys: {list(data.keys())[:25]}"
                elif isinstance(data, list):
                    first_keys = list(data[0].keys())[:20] if data and isinstance(data[0], dict) else type(data[0]).__name__ if data else "empty"
                    info["structure"] = f"JSON array of {len(data)} items (item schema: {first_keys})"
                else:
                    info["structure"] = f"JSON scalar ({type(data).__name__})"
                info["sample"] = json.dumps(data, indent=2, ensure_ascii=False)[:600]
                return info

            if ext == ".jsonl":
                lines = [ln for ln in fpath.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                first_obj = json.loads(lines[0]) if lines else {}
                keys = list(first_obj.keys())[:20] if isinstance(first_obj, dict) else type(first_obj).__name__
                info["structure"] = f"JSONL with {len(lines)} records (record keys: {keys})"
                info["sample"] = "\n".join(lines[:3])[:600]
                return info

            if ext in (".py", ".js", ".ts", ".sh", ".sql", ".md", ".txt", ".yaml", ".yml", ".tex", ".html"):
                text = fpath.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                info["structure"] = f"Text/Code file ({len(lines)} lines, {len(text):,} chars)"
                info["sample"] = "\n".join(lines[:15])[:600]
                return info

        except Exception as exc:
            info["structure"] = f"File ({info['size_bytes']:,} bytes) [schema preview note: {exc}]"
            return info

        info["structure"] = f"Binary/Document file ({info['size_bytes']:,} bytes)"
        return info

    def inspect(self, inputs: Any) -> Dict[str, Any]:
        payload = _load_sources_payload(inputs, router=self.router)
        manifest: List[Dict[str, Any]] = []

        raw_items: List[str] = []
        if isinstance(inputs, (str, Path)):
            s = str(inputs).strip()
            if s:
                raw_items = [p.strip() for p in s.split(",")] if ("," in s and not Path(s).expanduser().exists()) else [s]
        elif isinstance(inputs, (list, tuple, set)):
            raw_items = [str(x).strip() for x in inputs if str(x).strip()]

        for item in raw_items:
            if item.startswith(("http://", "https://")):
                manifest.append({
                    "name": item,
                    "path": item,
                    "ext": "url",
                    "size_bytes": 0,
                    "structure": "Remote URL source",
                    "sample": "",
                })
                continue
            cand = Path(item[1:] if item.startswith("@") else item).expanduser()
            if cand.is_dir():
                for fpath in sorted(p for p in cand.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts)):
                    manifest.append(self._inspect_single_file(fpath))
            elif cand.is_file():
                manifest.append(self._inspect_single_file(cand))

        if not manifest and not payload["file_count"]:
            brief = "No external input files provided."
        else:
            lines = [f"Total inputs: {payload['file_count']} item(s), ~{payload['tokens']:,} tokens."]
            for entry in manifest:
                lines.append(f"- `{entry['name']}` ({entry['path']}): {entry['structure']}")
                if entry["sample"]:
                    lines.append(f"  Sample:\n  ```\n  {entry['sample'][:350]}\n  ```")
            brief = "\n".join(lines)

        return {
            "manifest": manifest,
            "brief_summary": brief,
            "full_payload": payload,
        }

