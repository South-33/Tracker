import json
from tracker.validation_cache import test_fingerprint as fingerprint, passing_result


def test_changes_to_code_and_dependencies_invalidate_test_result(tmp_path):
    code = tmp_path / "src/tracker/code.py"
    code.parent.mkdir(parents=True)
    code.write_text("value = 1\n")
    env = tmp_path / "env"
    original = fingerprint(tmp_path, env)
    cache = tmp_path / "runs/.lab-tests.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"fingerprint": original, "returncode": 0}))
    assert passing_result(tmp_path, original)
    code.write_text("value = 2\n")
    changed = fingerprint(tmp_path, env)
    assert original != changed
    assert passing_result(tmp_path, changed) is None
    metadata = env / "Lib/site-packages/example.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Version: 2\n")
    assert fingerprint(tmp_path, env) != changed


def test_result_artifacts_do_not_trigger_full_unit_suite(tmp_path):
    before = fingerprint(tmp_path, tmp_path / "env")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/loss.jsonl").write_text('{"loss": 1}')
    assert fingerprint(tmp_path, tmp_path / "env") == before


def test_failed_or_corrupt_cache_does_not_skip_tests(tmp_path):
    cache = tmp_path / "runs/.lab-tests.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"fingerprint": "same", "returncode": 1}))
    assert passing_result(tmp_path, "same") is None
    cache.write_text("partial{")
    assert passing_result(tmp_path, "same") is None
