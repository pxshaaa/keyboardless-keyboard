import json

import numpy as np
import pytest

from phase0.analysis import hirecall as HR


def _double_peak() -> np.ndarray:
    """Two taps merged into one peak: the second is a shoulder, never a local maximum."""
    x = np.arange(40.0)
    return 0.9 * np.exp(-((x - 15) ** 2) / 6) + 0.55 * np.exp(-((x - 19) ** 2) / 6)


def test_strict_nms_emits_one_event_for_a_merged_pair():
    p = _double_peak()
    assert len(HR.pick(p, 0.1, 1, 1, "peak")) == 1


def test_shoulder_mode_recovers_the_buried_second_tap():
    p = _double_peak()
    ev = HR.pick(p, 0.1, 1, 1, "shoulder")
    assert len(ev) == 2
    assert 17 <= ev[1] <= 21


def test_shoulder_mode_is_a_superset_of_strict_peaks():
    rng = np.random.default_rng(0)
    p = np.convolve(rng.random(500), np.ones(5) / 5, mode="same")
    a = HR.pick(p, 0.3, 1, 1, "peak")
    b = HR.pick(p, 0.3, 1, 1, "shoulder")
    assert set(a.tolist()) <= set(b.tolist())


def test_lowering_the_threshold_only_ever_adds_events():
    rng = np.random.default_rng(1)
    p = np.convolve(rng.random(400), np.ones(3) / 3, mode="same")
    for nms in ("peak", "shoulder"):
        hi = set(HR.pick(p, 0.6, 1, 1, nms).tolist())
        assert hi <= set(HR.pick(p, 0.2, 1, 1, nms).tolist())


def test_refractory_is_honoured_by_the_greedy_suppressor():
    p = _double_peak()
    ev = HR.pick(p, 0.1, 6, 1, "shoulder")
    assert len(ev) == 1 and ev[0] == 15


def test_nms_keeps_the_taller_candidate_of_a_close_pair():
    h = np.array([0.4, 0.9, 0.5])
    assert HR._nms(np.array([10, 12, 14]), h, 4, 40).tolist() == [12]


def test_density_bins_partition_the_rows(monkeypatch):
    monkeypatch.setitem(HR._NCH, HR.DESK, 100)
    rows = [{"n_taps": n, "per": []} for n in (60, 100, 120, 200, 350)]
    got = [len(HR.bin_rows("t", rows, lo, hi)) for lo, hi in zip(HR.BINS[:-1], HR.BINS[1:])]
    assert sum(got) == len(rows)
    assert HR.density("t", rows[1]) == 1.0


def test_subsample_spans_the_density_range_and_never_sorts_by_cer():
    cands = [({"i": i}, np.arange(n)) for i, n in enumerate([300, 50, 200, 120, 70])]
    got = HR.subsample(cands, 3)
    sizes = sorted(len(ev) for _, ev in got)
    assert len(got) == 3 and sizes[0] == 50 and sizes[-1] == 300
    assert HR.subsample(cands, 9) is cands


def test_merged_tables_rekey_stream_ids_without_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(HR, "HR_CACHE", tmp_path)
    monkeypatch.setattr(HR, "_ST", {})
    for tag, n in (("a", 2), ("b", 3)):
        (tmp_path / f"streams_{tag}.json").write_text(json.dumps(
            [{"sid": i, "cfg": {"src": tag}, "ev": [i]} for i in range(n)]))
        (tmp_path / f"table_{tag}_none.json").write_text(json.dumps(
            [{"sid": i, "per": [(1, 2)], "n_taps": 1} for i in range(n)]))
    a = type("A", (), {"tags": "a,b", "mode": "none"})()
    tag, rows = HR._merged(a)
    assert sorted(r["sid"] for r in rows) == [0, 1, 2, 3, 4]
    assert [s["cfg"]["src"] for s in HR.load_streams(tag)] == ["a", "a", "b", "b", "b"]
    assert HR.ev_of(tag, 4).tolist() == [2]


def test_the_sweep_grid_reaches_well_past_the_saturated_operating_point():
    assert min(HR.THR) < 0.30 and min(HR.GRID["refractory"]) < 3
    assert 0.0 in HR.GRID["gate_thr"] and 1 in HR.GRID["n_consec"]
    assert HR.BAND[1] > 1.13 * 2


@pytest.mark.parametrize("nms", ("peak", "shoulder"))
def test_events_are_sorted_and_unique(nms):
    rng = np.random.default_rng(2)
    p = np.convolve(rng.random(600), np.ones(4) / 4, mode="same")
    ev = HR.pick(p, 0.25, 3, 1, nms)
    assert np.all(np.diff(ev) >= 3)
    assert len(set(ev.tolist())) == len(ev)
