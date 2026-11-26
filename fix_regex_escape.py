from pathlib import Path

target = Path.home() / "broutegpt" / "broutegpt.py"
src = target.read_text(encoding="utf-8")

old = 'if re.search(r"(?:^|[-_/:])(?:0.5|1.5|[1-9]|12)b(?:$|[-_/:])", mid)'
new = 'if re.search(r"(?:^|[-_/:])(?:0\\.5|1\\.5|[1-9]|12)b(?:$|[-_/:])", mid)'

assert old in src, "Anchor not found"
target.write_text(src.replace(old, new, 1), encoding="utf-8")
print("Escaped decimal points in regex.")

