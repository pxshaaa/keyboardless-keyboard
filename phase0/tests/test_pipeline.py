import numpy as np
import pytest

from phase0.analysis import pipeline as pl


def _rows(errs, refs):
    return [("x" * r, "", e, r) for e, r in zip(errs, refs)]


def test_pooled_is_character_weighted_not_phrase_averaged():
    rows = _rows([1, 9], [10, 90])
    assert pl.pooled(rows) == pytest.approx(0.1)
    assert pl.pooled(_rows([5, 0], [10, 90])) == pytest.approx(0.05)


def test_boot_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    refs = rng.integers(20, 45, 20)
    rows = _rows((refs * 0.6).astype(int), refs)
    c, lo, hi = pl.boot_ci(rows, n=2000)
    assert lo < c < hi
    assert 0.0 < lo and hi < 1.0


def test_boot_ci_is_degenerate_when_every_phrase_is_identical():
    rows = _rows([5] * 20, [10] * 20)
    c, lo, hi = pl.boot_ci(rows, n=500)
    assert (c, lo, hi) == pytest.approx((0.5, 0.5, 0.5))


def test_boot_delta_sign_is_b_minus_a():
    a = _rows([5] * 10, [10] * 10)
    b = _rows([3] * 10, [10] * 10)
    d, lo, hi = pl.boot_delta(a, b, n=500)
    assert d == pytest.approx(-0.2)
    assert lo <= d <= hi


def test_boot_delta_uses_a_shared_denominator_so_it_is_paired():
    # unequal phrase lengths: an unpaired delta would not be exactly -0.1 here
    a = [("x" * 10, "", 5, 10), ("x" * 90, "", 45, 90)]
    b = [("x" * 10, "", 4, 10), ("x" * 90, "", 36, 90)]
    d, _, _ = pl.boot_delta(a, b, n=200)
    assert d == pytest.approx(-0.1)


def test_select_cfg_ignores_the_held_out_phrase():
    rows = [{"cfg": {"id": "a"}, "deletion": -9.0, "per": [(0, 10), (10, 10)]},
            {"cfg": {"id": "b"}, "deletion": -4.0, "per": [(10, 10), (0, 10)]}]
    assert pl.select_cfg(rows, {0})["cfg"]["id"] == "a"
    assert pl.select_cfg(rows, {1})["cfg"]["id"] == "b"
    assert pl.select_cfg(rows, {0, 1})["cfg"]["id"] == "a"
    assert pl.select_cfg(rows, {1})["deletion"] == -4.0


def test_stack_weights_carry_the_insertion_and_deletion_costs():
    w = pl.Stack(insertion=-4.0, deletion=-5.0).weights()
    assert (w.insertion, w.deletion) == (-4.0, -5.0)


def test_full_stack_does_not_include_the_contact_expert():
    assert pl.FULL.contact_w == 0.0
    assert pl.SWEEP.em is False and pl.SWEEP.coral is True


def test_ablation_ladder_ends_at_the_full_stack_row():
    names = [n for n, _, _ in pl.ABLATIONS]
    assert names[0].startswith("no camera")
    assert pl.ABLATIONS[4][1] == pl.FULL


def test_candidate_grid_covers_the_desk_refractory_and_low_thresholds():
    # desk_tune found 160 ms suits desk rhythm; low thresholds are the recall-weighted end
    assert 160.0 in pl.GRID["refractory_ms"]
    assert min(pl.GRID["thr"]) <= 0.30
    assert 0.25 in pl.GRID["gate_thr"]


@pytest.mark.slow
def test_taps_from_and_segments_agree_on_every_phrase(tmp_path):
    s = pl.session_path(pl.DESK)
    d = pl.frame_probs(s)
    taps = pl.taps_from(d, pl.event_idx(d, pl.champion_cfg(s)))
    segs = pl.segments(s, taps)
    assert len(segs) == 20
    assert sum(len(t) for t, _ in segs) == 676
    assert all(0 <= i < len(taps) for _, rows in segs for i in rows)
