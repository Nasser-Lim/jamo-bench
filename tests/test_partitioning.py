# -*- coding: utf-8 -*-
import math

from jamo_bench.decompose import ALL_18_CELLS, decompose
from jamo_bench.partitioning import minimal_pair_reserve_map, partition


def test_core_540_exact_and_unique():
    result = partition(seed=42)
    assert len(result.core_540) == 540
    assert len(set(result.core_540)) == 540


def test_every_cell_hits_quota_30():
    result = partition(seed=42)
    for cell in ALL_18_CELLS:
        assert len(result.core_by_cell[cell]) == 30, cell


def test_reproducible_with_same_seed():
    r1 = partition(seed=123)
    r2 = partition(seed=123)
    assert r1.to_json() == r2.to_json()


def test_private_holdout_disjoint_from_core():
    result = partition(seed=1)
    assert set(result.private_test_holdout).isdisjoint(set(result.core_540))


def test_reserve_syllables_land_in_core_540():
    result = partition(seed=1)
    reserve = minimal_pair_reserve_map()
    core_set = set(result.core_540)
    missing = [c for c in reserve if c not in core_set]
    assert missing == [], f"reserve syllables missing from Core 540: {missing}"


def test_cross_shared_250_is_subset_of_core():
    result = partition(seed=1)
    assert len(result.cross_shared_250) == 250
    assert len(set(result.cross_shared_250)) == 250
    assert set(result.cross_shared_250).issubset(set(result.core_540))


def test_cross_struct_180_is_subset_of_core():
    result = partition(seed=1)
    assert len(result.cross_struct_180) == 180
    assert len(set(result.cross_struct_180)) == 180
    assert set(result.cross_struct_180).issubset(set(result.core_540))
    for cell6, chars in result.cross_struct_by_cell6.items():
        assert len(chars) == 30, cell6


def test_tensed_double_quota_in_simple_t_cells():
    result = partition(seed=1)
    for cell in ALL_18_CELLS:
        v, t, o = cell
        if t != "simple_T":
            continue
        chars = result.core_by_cell[cell]
        n_tensed = sum(1 for c in chars if decompose(c).coda_class_4 == "tensed_double")
        assert n_tensed >= math.ceil(0.20 * 30), cell


def test_vertical_derived_band_in_simple_v_cells():
    result = partition(seed=1)
    for cell in ALL_18_CELLS:
        v, t, o = cell
        if v != "simple_V":
            continue
        chars = result.core_by_cell[cell]
        n_vd = sum(1 for c in chars if decompose(c).vowel_shape == "vertical_derived")
        assert math.ceil(0.30 * 30) <= n_vd <= math.floor(0.40 * 30), (cell, n_vd)


def test_no_plan_b_prime_trigger_by_default():
    result = partition(seed=1)
    assert result.plan_b_prime_triggered is False
    assert result.warnings == ()


def test_banned_syllables_excluded_everywhere():
    result = partition(seed=1, banned_syllables=frozenset({"씨발"[:1], "가"}))
    assert "가" not in result.core_540
    assert "가" not in result.private_test_holdout


def test_to_json_roundtrip_serializable():
    import json

    result = partition(seed=5)
    text = result.to_json_str()
    parsed = json.loads(text)
    assert len(parsed["core_540"]) == 540
