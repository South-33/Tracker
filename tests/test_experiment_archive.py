import json
from pathlib import Path

import pytest

from tracker.experiment_archive import start_experiment, finish_experiment


def setup_source(root):
    script = root / "experiments/example/probe.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('original experiment')\n")
    library = root / "src/tracker/model.py"
    library.parent.mkdir(parents=True)
    library.write_text("value = 1\n")
    return script


def test_source_survives_later_edits_and_result_is_archived(tmp_path):
    script = setup_source(tmp_path)
    record = start_experiment(tmp_path, "example", script, ["python", str(script)],
                              {"cycle": "test"}, "result.json")
    script.write_text("print('different experiment')\n")
    (tmp_path / "src/tracker/model.py").write_text("value = 2\n")
    (tmp_path / "result.json").write_text('{"accuracy":0.2}')
    result = finish_experiment(tmp_path, record, 0)
    assert result["status"] == "complete"
    assert (record.parent / "source/experiments/example/probe.py").read_text() == "print('original experiment')\n"
    assert (record.parent / "source/src/tracker/model.py").read_text() == "value = 1\n"
    assert json.loads((record.parent / "result.json").read_text())["accuracy"] == .2


def test_zero_exit_without_result_is_not_complete(tmp_path):
    script = setup_source(tmp_path)
    record = start_experiment(tmp_path, "missing", script, [], {}, "missing.json")
    assert finish_experiment(tmp_path, record, 0)["status"] == "missing_result"


def test_failure_keeps_the_original_source_and_exit_status(tmp_path):
    script = setup_source(tmp_path)
    record = start_experiment(tmp_path, "failed", script, [], {}, "result.json")
    result = finish_experiment(tmp_path, record, 1)
    assert result["status"] == "failed"
    assert result["returncode"] == 1
    assert (record.parent / "source/experiments/example/probe.py").is_file()


def test_invalid_json_is_not_a_successful_result(tmp_path):
    script = setup_source(tmp_path)
    record = start_experiment(tmp_path, "invalid", script, [], {}, "result.json")
    (tmp_path / "result.json").write_text("partial{")
    assert finish_experiment(tmp_path, record, 0)["status"] == "invalid_result"


def test_external_output_and_path_traversal_are_rejected(tmp_path):
    script = setup_source(tmp_path)
    with pytest.raises(ValueError, match="leaves"):
        start_experiment(tmp_path, "escape", script, [], {}, "../result.json")
    with pytest.raises(ValueError, match="name"):
        start_experiment(tmp_path, "../escape", script, [], {}, "result.json")


def test_preexisting_result_cannot_make_a_noop_run_successful(tmp_path):
    script = setup_source(tmp_path)
    (tmp_path / "result.json").write_text('{"old":true}')
    record = start_experiment(tmp_path, "stale", script, [], {}, "result.json")
    assert finish_experiment(tmp_path, record, 0)["status"] == "stale_result"


def test_context_is_saved_as_json_without_duplicate_markdown(tmp_path):
    import hashlib
    script = setup_source(tmp_path)
    original = "# Tracker\r\n\r\nCafé\n".encode("utf-8")
    (tmp_path / "tracker.md").write_bytes(original)
    record_path = start_experiment(tmp_path, "context", script, [], {}, "result.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    restored = record["context_utf8"]["tracker.md"].encode("utf-8")
    assert restored == original
    assert hashlib.sha256(restored).hexdigest() == record["source_sha256"]["tracker.md"]
    assert not list(record_path.parent.rglob("*.md"))
