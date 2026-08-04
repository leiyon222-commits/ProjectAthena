import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "research" / "preregistrations.jsonl"
RESULTS = ROOT / "research" / "results.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    args = parser.parse_args()
    records = [json.loads(line) for line in PREREG.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [item for item in records if item["experiment_id"] == args.experiment_id]
    if len(matches) != 1:
        raise RuntimeError("一意な事前登録がありません")
    prereg = matches[0]
    script = (ROOT / prereg["script_path"]).resolve()
    current_hash = sha256_file(script)
    if current_hash != prereg["script_sha256"]:
        raise RuntimeError("SCRIPT_HASH_MISMATCH: 実験を無効化して停止します")
    for dependency, expected_hash in prereg.get("dependency_sha256", {}).items():
        dependency_path = (ROOT / dependency).resolve()
        if not dependency_path.is_relative_to(ROOT):
            raise RuntimeError("DEPENDENCY_PATH_INVALID: 実験を停止します")
        if sha256_file(dependency_path) != expected_hash:
            raise RuntimeError("DEPENDENCY_HASH_MISMATCH: 実験を無効化して停止します")
    existing = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(item["experiment_id"] == args.experiment_id for item in existing):
        raise RuntimeError("結果は既に記録されています")
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(output_lines[-1])
    result = {
        "experiment_id": args.experiment_id,
        "completed_at": datetime.now().astimezone().isoformat(),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "script_sha256_at_execution": current_hash,
        "dependency_sha256_at_execution": prereg.get("dependency_sha256", {}),
        "trade_count": payload["trade_count"],
        "buy_count": payload["buy_count"],
        "sell_count": payload["sell_count"],
        "total_r": payload["total_r"],
        "average_r": payload["average_r"],
        "average_r_95pct_lower": payload["average_r_95pct_lower"],
        "profit_factor": payload["profit_factor"],
        "max_drawdown_r": payload["max_drawdown_r"],
        "positive_folds": payload["positive_folds"],
        "result": payload,
        "status": "COMPLETED",
    }
    with RESULTS.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    print(completed.stdout, end="")
    print("RESULT_RECORD=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
