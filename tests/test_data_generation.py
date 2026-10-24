import json

import pytest

from conftest import load_module

utils = load_module("data-generation/src/utils.py", "datagen_utils")


def _write_input(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"page_title": f"T{i}", "text": f"text {i}"}) + "\n")


def _prepare(tmp_path, n, splits, max_lines=None):
    src = tmp_path / "in.jsonl"
    _write_input(src, n)
    return utils.prepare_batch_file(
        str(src),
        str(tmp_path / "out.jsonl"),
        lambda: "system",
        lambda d: d["text"],
        "test-model",
        max_lines,
        splits,
    )


def _ids(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["custom_id"] for line in f]


def test_split_with_remainder_creates_exactly_n_files(tmp_path):
    # Regression: 5 lines / 2 splits produced 3 files (out_0, out_1, out_2).
    outputs = _prepare(tmp_path, 5, 2)
    assert sorted(outputs) == [str(tmp_path / "out_0.jsonl"), str(tmp_path / "out_1.jsonl")]
    all_ids = sum((_ids(p) for p in sorted(outputs)), [])
    assert all_ids == ["0", "1", "2", "3", "4"]


def test_more_splits_than_lines_does_not_divide_by_zero(tmp_path):
    outputs = _prepare(tmp_path, 1, 2)
    assert len(outputs) == 1


def test_rerun_does_not_duplicate_requests(tmp_path):
    # Regression: outputs were opened in append mode, so a re-run duplicated custom_ids.
    _prepare(tmp_path, 4, 1)
    outputs = _prepare(tmp_path, 4, 1)
    assert _ids(next(iter(outputs))) == ["0", "1", "2", "3"]


def test_max_lines_is_respected(tmp_path):
    outputs = _prepare(tmp_path, 10, 1, max_lines=3)
    assert _ids(next(iter(outputs))) == ["0", "1", "2"]


def test_request_shape(tmp_path):
    outputs = _prepare(tmp_path, 1, 1)
    with open(next(iter(outputs)), encoding="utf-8") as f:
        req = json.loads(f.readline())
    assert req["url"] == "/v1/chat/completions"
    assert req["body"]["model"] == "test-model"
    assert [m["role"] for m in req["body"]["messages"]] == ["system", "user"]


def test_api_key_prefers_environment(tmp_path):
    key = "test-" + "key-from-env"
    assert utils.get_api_key({"OPENAI_API_KEY": key}, str(tmp_path / "missing.env")) == key


def test_api_key_falls_back_to_dotenv(tmp_path):
    env = tmp_path / ".env"
    key = "test-" + "key-from-file"
    env.write_text(f"# comment\nOPEN_AI_KEY={key}\n")
    assert utils.get_api_key({}, str(env)) == key


def test_missing_api_key_is_a_clear_error_not_a_crash(tmp_path):
    # Regression: a missing .env raised FileNotFoundError / KeyError.
    with pytest.raises(SystemExit) as exc:
        utils.get_api_key({}, str(tmp_path / "missing.env"))
    assert "OPENAI_API_KEY" in str(exc.value.code)
