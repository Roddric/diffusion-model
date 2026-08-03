"""Create an immutable, hash-pinned protocol record before an experiment runs."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state(cwd):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def register(
    output,
    experiment_id,
    hypothesis,
    data_boundary,
    decision_rule,
    configs=(),
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Protocol already exists: {output}")
    config_records = []
    for config in configs:
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config does not exist: {path}")
        config_records.append(
            {"path": str(path), "sha256": _sha256(path)}
        )
    report = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned_before_execution",
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "data_boundary": data_boundary,
        "decision_rule": decision_rule,
        "configs": config_records,
        "version_control": _git_state(Path.cwd()),
        "integrity_note": (
            "Do not modify this file after registration. Record outcomes in a "
            "separate result artifact, including failures."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--data-boundary", required=True)
    parser.add_argument("--decision-rule", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = register(
        output=args.output,
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        data_boundary=args.data_boundary,
        decision_rule=args.decision_rule,
        configs=args.config,
    )
    print(
        f"Registered {report['experiment_id']} at "
        f"{report['registered_at_utc']}"
    )


if __name__ == "__main__":
    main()
