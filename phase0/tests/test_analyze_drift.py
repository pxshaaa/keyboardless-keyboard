import math

import pytest

from phase0.analysis.analyze_drift import (
    CONFIDENT_MATCH_COST,
    FALLBACK_PITCH_PX_DEFAULT,
    KeyStats,
    align_phrase,
    calibrate_pitch,
    normalize_char,
    pair_events,
)


def greedy_nearest(key_times, tap_times, window=0.080):
    # the naive implementation the contract's wording invites; kept only to prove it is wrong.
    out = []
    for i, kt in enumerate(key_times):
        best = min(range(len(tap_times)), key=lambda j: abs(tap_times[j] - kt))
        if abs(tap_times[best] - kt) <= window:
            out.append((i, best))
    return out


# --- pairing --------------------------------------------------------------
def test_greedy_double_assigns_but_pair_events_does_not():
    keys = [0.000, 0.040]
    taps = [0.030, 0.075]
    assert greedy_nearest(keys, taps) == [(0, 0), (1, 0)]  # same tap twice
    pairs = pair_events(keys, taps)
    assert pairs == [(0, 0), (1, 1)]
    assert len({t for _, t in pairs}) == len(pairs)


def test_fast_typing_burst_never_reuses_a_tap():
    keys = [0.000, 0.045, 0.090, 0.135, 0.180]
    taps = [0.010, 0.050, 0.100, 0.140, 0.190]
    pairs = pair_events(keys, taps)
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def test_prefers_more_pairs_over_a_single_tighter_one():
    # greedy would grab tap 0 for key 1 (10 ms) and orphan key 0 entirely.
    keys = [0.000, 0.040]
    taps = [0.035, 0.110]
    pairs = pair_events(keys, taps)
    assert len(pairs) == 2


def test_window_is_enforced():
    assert pair_events([0.0], [0.081]) == []
    assert pair_events([0.0], [0.079]) == [(0, 0)]


def test_missing_and_extra_taps_are_left_unpaired():
    keys = [0.0, 1.0, 2.0]
    taps = [0.01, 1.5, 2.02, 5.0]
    pairs = pair_events(keys, taps)
    assert pairs == [(0, 0), (2, 2)]  # key 1 unpaired, taps 1 and 3 unpaired


def test_empty_inputs():
    assert pair_events([], [1.0]) == []
    assert pair_events([1.0], []) == []


def test_unsorted_input_rejected():
    with pytest.raises(Exception):
        pair_events([1.0, 0.0], [0.0, 1.0])


# --- key pitch ------------------------------------------------------------
def _ks(key, x, y, n=20):
    return KeyStats(key, n, x, y, 4.0, 2.0, 1.0, "index", 1.0)


def test_pitch_from_home_row_is_the_median_adjacent_distance():
    pitch = 40.0
    xs = {"a": 0, "s": 40, "d": 80, "f": 120, "j": 300, "k": 340, "l": 380}
    stats = {k: _ks(k, x, 500.0) for k, x in xs.items()}
    px, note = calibrate_pitch(stats, 1920)
    assert px == pytest.approx(pitch)
    assert "home-row pairs (5/5)" in note


def test_pitch_uses_median_so_one_bad_pair_does_not_move_it():
    xs = {"a": 0, "s": 40, "d": 80, "f": 120, "j": 300, "k": 340, "l": 900}
    stats = {k: _ks(k, x, 500.0) for k, x in xs.items()}
    px, _ = calibrate_pitch(stats, 1920)
    assert px == pytest.approx(40.0)


def test_pitch_falls_back_to_nearest_neighbour_when_home_row_missing():
    stats = {k: _ks(k, 50.0 * i, 500.0) for i, k in enumerate("qwer")}
    px, note = calibrate_pitch(stats, 1920)
    assert px == pytest.approx(50.0)
    assert note.startswith("FALLBACK")


def test_pitch_falls_back_to_assumption_when_almost_nothing_is_there():
    px, note = calibrate_pitch({"a": _ks("a", 0.0, 0.0)}, None)
    assert px == pytest.approx(FALLBACK_PITCH_PX_DEFAULT)
    assert note.startswith("ASSUMED")


def test_pitch_weak_when_one_home_row_pair():
    px, note = calibrate_pitch({"a": _ks("a", 0.0, 0.0), "s": _ks("s", 42.0, 0.0)}, 1920)
    assert px == pytest.approx(42.0)
    assert note.startswith("WEAK")


# --- desk phrase alignment ------------------------------------------------
def chars_of(phrase):
    return [c for c in (normalize_char(ch) for ch in phrase) if c]


def taps_for(chars, drop=(), extra=()):
    """Synthesise one tap per character (thumb for spaces), then drop/insert some."""
    taps = []
    for i, c in enumerate(chars):
        if i in drop:
            continue
        taps.append({"t": 100.0 + 0.2 * i, "x": 10.0 * i, "y": 500.0, "finger": 4 if c == "space" else 8, "hand": 0})
    for pos, finger in extra:
        taps.append({"t": 100.0 + 0.2 * pos + 0.1, "x": 10.0 * pos, "y": 505.0, "finger": finger, "hand": 0})
    taps.sort(key=lambda t: t["t"])
    return taps


def test_perfect_transcription_aligns_one_to_one():
    chars = chars_of("the quick brown fox")
    al = align_phrase(chars, taps_for(chars))
    assert len(al.matches) == len(chars)
    assert al.frac_confident == 1.0
    assert al.norm_cost == pytest.approx(0.0, abs=1e-9)
    assert [c for c, _, _ in al.matches] == list(range(len(chars)))


def test_alignment_is_monotonic():
    chars = chars_of("pack my box with five dozen liquor jugs")
    al = align_phrase(chars, taps_for(chars, drop=(3, 11, 20), extra=((7, 8), (15, 8))))
    cs = [c for c, _, _ in al.matches]
    ts = [t for _, t, _ in al.matches]
    assert cs == sorted(cs) and ts == sorted(ts)
    assert len(set(cs)) == len(cs) and len(set(ts)) == len(ts)


def test_missed_taps_leave_those_characters_unlabelled_not_shifted():
    chars = chars_of("the five boxing wizards jump quickly")
    dropped = {5, 12, 19}
    al = align_phrase(chars, taps_for(chars, drop=tuple(dropped)))
    matched = {c for c, _, cost in al.matches if cost < CONFIDENT_MATCH_COST}
    assert dropped.isdisjoint(matched)
    assert matched == set(range(len(chars))) - dropped


def test_hallucinated_taps_are_absorbed_as_insertions():
    chars = chars_of("sphinx of black quartz")
    al = align_phrase(chars, taps_for(chars, extra=((4, 8), (9, 8), (14, 8))))
    assert len(al.matches) <= len(chars)
    assert al.frac_confident > 0.7


def test_space_thumb_disagreement_costs_more():
    chars = ["a", "space", "b"]
    good = [{"t": 0.0, "x": 0, "y": 0, "finger": 8}, {"t": 1.0, "x": 0, "y": 0, "finger": 4}, {"t": 2.0, "x": 0, "y": 0, "finger": 8}]
    bad = [dict(g, finger=(8 if i == 1 else 4)) for i, g in enumerate(good)]
    assert align_phrase(chars, good).total_cost < align_phrase(chars, bad).total_cost


def test_garbage_taps_give_a_low_confidence_alignment():
    chars = chars_of("a journey of a thousand miles begins with a single step")
    taps = [{"t": 100.0 + 0.2 * i, "x": 0, "y": 0, "finger": 4} for i in range(len(chars))]
    al = align_phrase(chars, taps)
    assert al.frac_confident < 1.0  # every tap claims to be a space
    assert al.norm_cost > 0.0


def test_alignment_handles_empty_sides():
    assert align_phrase([], []).matches == []
    assert align_phrase(["a"], []).matches == []
    assert math.isinf(align_phrase(["a"], []).norm_cost)


def test_normalize_char():
    assert normalize_char(" ") == "space"
    assert normalize_char("A") == "a"
    assert normalize_char("7") == "7"
    assert normalize_char(",") is None
