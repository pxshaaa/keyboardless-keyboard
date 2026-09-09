"""Unit tests for key normalization (privacy allowlist) and the JSONL writer."""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase0.capture.recorder import (  # noqa: E402
    UNKNOWN,
    JsonlWriter,
    key_name,
    normalize_key,
    resolve_camera,
)

LEGAL = set("abcdefghijklmnopqrstuvwxyz0123456789") | {
    "space",
    "enter",
    "tab",
    "backspace",
    "shift",
    "ctrl",
    "alt",
    "cmd",
    "esc",
    UNKNOWN,
}


class FakeCode:
    """Stand-in for pynput.keyboard.KeyCode (has .char, no .name)."""

    def __init__(self, char):
        self.char = char


class FakeKey:
    """Stand-in for pynput.keyboard.Key members (has .name)."""

    def __init__(self, name):
        self.name = name


# --- key_name -------------------------------------------------------------


def test_key_name_prefers_name_over_char():
    k = FakeKey("shift_r")
    k.char = "X"
    assert key_name(k) == "shift_r"


@pytest.mark.parametrize("bad", [FakeCode(None), FakeCode(""), object()])
def test_key_name_unrecoverable_is_unknown(bad):
    assert key_name(bad) == UNKNOWN


# --- allowlist ------------------------------------------------------------


@pytest.mark.parametrize("ch", list("abcxyz0159"))
def test_letters_and_digits_pass_through(ch):
    assert normalize_key(FakeCode(ch)) == ch


@pytest.mark.parametrize("ch", list("ABCXYZ"))
def test_uppercase_is_lowercased(ch):
    assert normalize_key(FakeCode(ch)) == ch.lower()


def test_shift_does_not_suppress_content():
    assert normalize_key(FakeCode("A"), {"shift"}) == "a"


@pytest.mark.parametrize(
    "ch", list(".,;:'\"/\\-_=+!@#$%^&*()[]{}<>?`~|") + ["ü", "é", "€", "√"]
)
def test_punctuation_and_non_ascii_never_leak(ch):
    assert normalize_key(FakeCode(ch)) == UNKNOWN


@pytest.mark.parametrize(
    "name,expected",
    [
        ("space", "space"),
        ("enter", "enter"),
        ("tab", "tab"),
        ("backspace", "backspace"),
        ("esc", "esc"),
        ("shift", "shift"),
        ("shift_l", "shift"),
        ("shift_r", "shift"),
        ("ctrl", "ctrl"),
        ("ctrl_r", "ctrl"),
        ("alt", "alt"),
        ("alt_gr", "alt"),
        ("cmd", "cmd"),
        ("cmd_r", "cmd"),
    ],
)
def test_special_and_modifier_names(name, expected):
    assert normalize_key(FakeKey(name)) == expected


@pytest.mark.parametrize(
    "name", ["f1", "f12", "media_play_pause", "caps_lock", "home", "page_down", "up"]
)
def test_other_named_keys_are_unknown(name):
    assert normalize_key(FakeKey(name)) == UNKNOWN


@pytest.mark.parametrize("mod", ["ctrl", "alt", "cmd"])
def test_any_non_shift_modifier_suppresses_the_character(mod):
    assert normalize_key(FakeCode("c"), {mod}) == UNKNOWN
    assert normalize_key(FakeCode("5"), {mod, "shift"}) == UNKNOWN


def test_modifier_key_itself_is_still_named_while_held():
    assert normalize_key(FakeKey("cmd"), {"cmd"}) == "cmd"


@pytest.mark.parametrize("mods", [None, set(), {"shift"}, {"ctrl"}, {"cmd", "alt"}])
def test_output_is_always_contract_legal(mods):
    probes = [FakeCode(c) for c in "aZ9.^ü"] + [
        FakeKey(n) for n in ["space", "f7", "esc", "print_screen"]
    ]
    for p in probes:
        assert normalize_key(p, mods) in LEGAL


# --- JsonlWriter ----------------------------------------------------------


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_writer_appends_one_json_object_per_line(tmp_path):
    p = tmp_path / "keys.jsonl"
    with JsonlWriter(p, flush_every=1) as w:
        w.write({"t": 1.5, "event": "down", "key": "a"})
        w.write({"t": 1.7, "event": "up", "key": "a"})
    assert _rows(p) == [
        {"t": 1.5, "event": "down", "key": "a"},
        {"t": 1.7, "event": "up", "key": "a"},
    ]


def test_writer_counts_rows(tmp_path):
    w = JsonlWriter(tmp_path / "f.jsonl", flush_every=5)
    for i in range(7):
        w.write({"i": i, "t": float(i)})
    assert w.count == 7
    w.close()


def test_writer_flushes_periodically_so_a_crash_keeps_data(tmp_path):
    p = tmp_path / "f.jsonl"
    w = JsonlWriter(p, flush_every=2)
    w.write({"i": 0, "t": 0.0})
    w.write({"i": 1, "t": 0.1})
    assert len(_rows(p)) == 2  # visible on disk without close()
    w.close()


def test_writer_opens_in_append_mode(tmp_path):
    p = tmp_path / "f.jsonl"
    JsonlWriter(p, flush_every=1).write({"i": 0, "t": 0.0})
    with JsonlWriter(p, flush_every=1) as w:
        w.write({"i": 1, "t": 0.1})
    assert [r["i"] for r in _rows(p)] == [0, 1]


def test_writer_close_is_idempotent_and_write_after_close_is_a_noop(tmp_path):
    p = tmp_path / "f.jsonl"
    w = JsonlWriter(p, flush_every=1)
    w.write({"i": 0, "t": 0.0})
    w.close()
    w.close()
    w.write({"i": 1, "t": 1.0})
    assert len(_rows(p)) == 1


def test_writer_is_thread_safe(tmp_path):
    p = tmp_path / "f.jsonl"
    w = JsonlWriter(p, flush_every=3)
    threads = [
        threading.Thread(target=lambda n=n: [w.write({"i": n, "t": float(j)}) for j in range(50)])
        for n in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    w.close()
    rows = _rows(p)
    assert len(rows) == 400
    assert all(set(r) == {"i", "t"} for r in rows)


# --- camera spec parsing (no camera hardware touched) ---------------------


def test_resolve_camera_accepts_integer_index(monkeypatch):
    monkeypatch.setattr("phase0.capture.recorder.list_mac_cameras", lambda: [])
    assert resolve_camera("2") == (2, "index:2")


def test_resolve_camera_matches_name_substring_case_insensitively(monkeypatch):
    monkeypatch.setattr(
        "phase0.capture.recorder.list_mac_cameras",
        lambda: ["FaceTime HD Camera", "Pasha's iPhone Camera"],
    )
    assert resolve_camera("iphone") == (1, "Pasha's iPhone Camera")


def test_resolve_camera_raises_on_no_match(monkeypatch):
    monkeypatch.setattr("phase0.capture.recorder.list_mac_cameras", lambda: ["FaceTime"])
    with pytest.raises(SystemExit):
        resolve_camera("nosuchcam")


# --- prompter (Amendment 1) -----------------------------------------------


def _run_prompter(tmp_path, phrases, phrase_seconds=0.02, advance=None):
    from phase0.capture.recorder import Prompter

    p = tmp_path / "phrases.jsonl"
    w = JsonlWriter(p, flush_every=1)
    pr = Prompter(
        phrases,
        w,
        threading.Event(),
        advance or threading.Event(),
        phrase_seconds=phrase_seconds,
        out=io.StringIO(),
    )
    pr.run()
    w.close()
    return pr, _rows(p)


def test_load_phrases_skips_blanks_and_comments(tmp_path):
    from phase0.capture.recorder import load_phrases

    f = tmp_path / "phrases.txt"
    f.write_text("# header\n\nthe quick brown fox\n  jumps over  \n\n")
    assert load_phrases(f) == ["the quick brown fox", "jumps over"]


def test_prompter_writes_shown_and_done_per_phrase(tmp_path):
    pr, rows = _run_prompter(tmp_path, ["alpha beta", "gamma"])
    assert pr.completed == 2
    assert [(r["event"], r["idx"], r["phrase"]) for r in rows] == [
        ("shown", 0, "alpha beta"),
        ("done", 0, "alpha beta"),
        ("shown", 1, "gamma"),
        ("done", 1, "gamma"),
    ]
    assert all(set(r) == {"t", "event", "phrase", "idx"} for r in rows)
    assert all(isinstance(r["t"], float) for r in rows)


def test_prompter_windows_are_monotonic_and_non_overlapping(tmp_path):
    _, rows = _run_prompter(tmp_path, ["a", "b", "c"])
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts)
    shown = [r["t"] for r in rows if r["event"] == "shown"]
    done = [r["t"] for r in rows if r["event"] == "done"]
    assert all(d >= s for s, d in zip(shown, done))
    assert all(done[i] <= shown[i + 1] for i in range(len(done) - 1))


def test_prompter_sets_stop_when_the_list_is_exhausted(tmp_path):
    from phase0.capture.recorder import Prompter

    w = JsonlWriter(tmp_path / "phrases.jsonl", flush_every=1)
    stop = threading.Event()
    pr = Prompter(["only"], w, stop, threading.Event(), 0.01, out=io.StringIO())
    pr.run()
    w.close()
    assert stop.is_set()


def test_prompter_stops_early_when_stop_is_set(tmp_path):
    from phase0.capture.recorder import Prompter

    p = tmp_path / "phrases.jsonl"
    w = JsonlWriter(p, flush_every=1)
    stop = threading.Event()
    stop.set()
    Prompter(["a", "b"], w, stop, threading.Event(), 5.0, out=io.StringIO()).run()
    w.close()
    assert _rows(p) == []


def test_manual_advance_key_shortens_the_window(tmp_path):
    import time as _t

    adv = threading.Event()
    threading.Timer(0.05, adv.set).start()  # simulates a space/enter press
    t0 = _t.monotonic()
    pr, rows = _run_prompter(tmp_path, ["a"], phrase_seconds=30.0, advance=adv)
    assert pr.completed == 1
    assert _t.monotonic() - t0 < 5.0
    assert rows[1]["t"] - rows[0]["t"] < 5.0


def test_a_stale_advance_does_not_skip_the_next_phrase(tmp_path):
    adv = threading.Event()
    adv.set()  # left over from the previous phrase; must be cleared, not honoured
    pr, rows = _run_prompter(tmp_path, ["a", "b"], phrase_seconds=0.05, advance=adv)
    assert pr.completed == 2
    assert rows[1]["t"] - rows[0]["t"] >= 0.04


def test_advance_keys_are_allowlisted_names():
    from phase0.capture.recorder import Prompter

    assert Prompter.ADVANCE_KEYS <= LEGAL
