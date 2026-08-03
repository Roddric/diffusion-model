"""CSI external-replication immutability and decision rules."""

from types import SimpleNamespace

import pytest

from research.freeze_csi_phase3b import _seed_weights
from research.replicate_csi_phase3b import (
    _external_success,
    run,
)


def test_seed_weights_normalize_json_keys():
    assert _seed_weights({"42": 0.25, "314": 0.5}, "test") == {
        42: 0.25,
        314: 0.5,
    }
    with pytest.raises(ValueError):
        _seed_weights({}, "test")


def test_external_success_requires_every_frozen_gate():
    ratios = {"state_energy_score": 0.98, "state_rmse": 0.99}
    success, checks = _external_success(0.985, ratios, 0.04, 1.01)
    assert success
    assert all(checks.values())

    assert not _external_success(0.985, ratios, 0.11, 1.01)[0]
    assert not _external_success(0.985, ratios, 0.04, 1.03)[0]


def test_replication_refuses_to_overwrite_existing_result(tmp_path):
    output = tmp_path / "replication.json"
    output.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(output=str(output))
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        run(args)
