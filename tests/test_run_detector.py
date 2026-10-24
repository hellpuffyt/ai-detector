import io

import pytest

from conftest import load_module

rd = load_module("run_detector.py", "run_detector_under_test")


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_label_threshold():
    assert rd.label_for(0.51) == "AI-generated"
    assert rd.label_for(0.5) == "Human-written"
    assert rd.label_for(0.0) == "Human-written"


def test_file_is_one_text(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("line one\nline two\n", encoding="utf-8")
    assert rd.read_texts(["prog", str(f)], io.StringIO()) == ["line one\nline two\n"]


def test_missing_file_exits_nonzero_before_loading_model(tmp_path, monkeypatch):
    # Regression: a missing file used to load (and download) the model first, then exit 0.
    monkeypatch.setattr(rd, "load_detector", lambda: pytest.fail("model must not load"))
    with pytest.raises(SystemExit) as exc:
        rd.main(["prog", str(tmp_path / "nope.txt")], io.StringIO())
    assert "not found" in str(exc.value.code)


def test_empty_file_is_rejected(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("  \n", encoding="utf-8")
    with pytest.raises(SystemExit):
        rd.read_texts(["prog", str(f)], io.StringIO())


def test_piped_stdin_without_blank_line_does_not_crash():
    # Regression: `printf 'a\nb\n' | python run_detector.py` raised EOFError.
    assert rd.read_texts(["prog"], io.StringIO("a\nb\n")) == ["a\nb\n"]


def test_empty_piped_stdin_is_rejected():
    with pytest.raises(SystemExit):
        rd.read_texts(["prog"], io.StringIO(""))


def test_interactive_lines_are_joined_into_one_text():
    # Regression: each interactive line used to be classified as a separate text.
    assert rd.read_texts(["prog"], FakeTTY("first line\nsecond line\n\nignored\n")) == [
        "first line\nsecond line"
    ]


def test_interactive_eof_ends_input():
    assert rd.read_texts(["prog"], FakeTTY("only line\n")) == ["only line"]


def test_interactive_nothing_entered_uses_examples():
    assert rd.read_texts(["prog"], FakeTTY("\n")) == rd.DEFAULT_EXAMPLES
