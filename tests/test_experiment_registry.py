"""Pre-execution experiment registration invariants."""

import json

import pytest

from research.register_experiment import register


def test_register_experiment_hashes_configs_and_refuses_overwrite(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    output = tmp_path / "protocol.json"
    report = register(
        output=output,
        experiment_id="future_test",
        hypothesis="Candidate improves energy.",
        data_boundary="Outcomes after 2026-08-04 only.",
        decision_rule="Energy ratio below 0.99.",
        configs=[config],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["status"] == "planned_before_execution"
    assert len(saved["configs"][0]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        register(
            output=output,
            experiment_id="future_test",
            hypothesis="Changed after seeing data.",
            data_boundary="Unknown",
            decision_rule="Unknown",
        )
