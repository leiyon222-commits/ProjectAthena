import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "research" / "preregistrations.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft")
    args = parser.parse_args()
    draft_path = (ROOT / args.draft).resolve()
    if not draft_path.is_relative_to(ROOT):
        raise RuntimeError("ProjectAthena外のdraftは禁止です")
    record = json.loads(draft_path.read_text(encoding="utf-8"))
    script_path = (ROOT / record["script_path"]).resolve()
    if not script_path.is_relative_to(ROOT) or not script_path.is_file():
        raise RuntimeError("script_pathが不正です")
    existing = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(item["experiment_id"] == record["experiment_id"] for item in existing):
        raise RuntimeError("実験IDは既に事前登録されています")
    record["script_sha256"] = sha256_file(script_path)
    dependency_hashes = {}
    for dependency in record.get("dependency_paths", []):
        dependency_path = (ROOT / dependency).resolve()
        if not dependency_path.is_relative_to(ROOT) or not dependency_path.is_file():
            raise RuntimeError(f"dependency_pathが不正です: {dependency}")
        dependency_hashes[dependency] = sha256_file(dependency_path)
    record["dependency_sha256"] = dependency_hashes
    record["preregistered_at"] = datetime.now().astimezone().isoformat()
    record["status"] = "PREREGISTERED"
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record["preregistration_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
