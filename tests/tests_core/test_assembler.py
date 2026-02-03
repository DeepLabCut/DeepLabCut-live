from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dlclive.core.inferenceutils import Assembler, Assembly, Joint, Link, _conv_square_to_condensed_indices

HYPOTHESIS_SETTINGS = settings(max_examples=300, deadline=None)


def _bag_from_frame(frame: dict) -> dict[int, list]:
    """Build joints bag {label: [Joint, ...]} from a single frame."""
    bag: dict[int, list] = {}
    for j in Assembler._flatten_detections(frame):
        bag.setdefault(j.label, []).append(j)
    return bag


# _conv_square_to_condensed_indices
@HYPOTHESIS_SETTINGS
@given(
    n=st.integers(min_value=2, max_value=50),
    i=st.integers(min_value=0, max_value=49),
    j=st.integers(min_value=0, max_value=49),
)
def test_condensed_index_properties(n, i, j):
    i = i % n
    j = j % n

    if i == j:
        with pytest.raises(ValueError):
            _conv_square_to_condensed_indices(i, j, n)
        return

    k1 = _conv_square_to_condensed_indices(i, j, n)
    k2 = _conv_square_to_condensed_indices(j, i, n)

    assert k1 == k2
    assert 0 <= k1 < (n * (n - 1)) // 2


# --------------------------------------------------------------------------------------
# Basic metadata and __getitem__
# --------------------------------------------------------------------------------------


def test_parse_metadata_and_getitem(assembler_data, make_assembler):
    adat = assembler_data
    # Parsing
    asm = make_assembler(
        adat.data,
        max_n_individuals=2,
        n_multibodyparts=2,
    )

    assert asm.metadata["num_joints"] == 2
    assert asm.metadata["paf_graph"] == adat.graph
    assert list(asm.metadata["paf"]) == adat.paf_inds
    assert set(asm.metadata["imnames"]) == {"0", "1"}
    # __getitem__
    assert "coordinates" in asm[0]
    assert "confidence" in asm[0]
    assert "costs" in asm[0]


def test_empty_classmethod(assembler_graph_and_pafs):
    paf = assembler_graph_and_pafs
    empty = Assembler.empty(
        max_n_individuals=1,
        n_multibodyparts=1,
        n_uniquebodyparts=0,
        graph=paf.graph,
        paf_inds=paf.paf_inds,
    )
    assert isinstance(empty, Assembler)
    assert empty.n_keypoints == 1


# --------------------------------------------------------------------------------------
# _flatten_detections
# --------------------------------------------------------------------------------------
def test_flatten_detections_no_identity(simple_two_label_scene):
    frame = simple_two_label_scene
    joints = list(Assembler._flatten_detections(frame))

    assert len(joints) == 4
    assert sorted(j.label for j in joints) == [0, 0, 1, 1]
    assert set(j.group for j in joints) == {-1}


def test_flatten_detections_with_identity(scene_copy):
    frame = scene_copy
    id0 = np.array([[10.0, 0.0], [0.0, 10.0]])
    id1 = np.array([[10.0, 0.0], [0.0, 10.0]])
    frame["identity"] = [id0, id1]

    joints = list(Assembler._flatten_detections(frame))
    groups = [j.group for j in joints]

    assert set(groups) == {0, 1}
    assert groups.count(0) == 2
    assert groups.count(1) == 2


@st.composite
def coords_and_conf(draw, max_n=5):
    n = draw(st.integers(1, max_n))
    coords = draw(
        arrays(
            dtype=np.float64,
            shape=(n, 2),
            elements=st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False),
        )
    )
    conf = draw(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    return coords, conf


@HYPOTHESIS_SETTINGS
@given(
    c0=coords_and_conf(),
    c1=coords_and_conf(),
)
def test_flatten_detections_counts(c0, c1):
    coords0, conf0 = c0
    coords1, conf1 = c1

    frame = {
        "coordinates": [[coords0, coords1]],
        "confidence": [conf0, conf1],
        "costs": {},
    }

    joints = list(Assembler._flatten_detections(frame))

    # Should yield exactly one Joint per detection
    assert len(joints) == (len(coords0) + len(coords1))
    assert sum(j.label == 0 for j in joints) == len(coords0)
    assert sum(j.label == 1 for j in joints) == len(coords1)


# --------------------------------------------------------------------------------------
# extract_best_links
# --------------------------------------------------------------------------------------
def test_extract_best_links_optimal_assignment(assembler_data_single_frame, make_assembler):
    sframe_data = assembler_data_single_frame
    asm = make_assembler(
        sframe_data.data,
        greedy=False,  # use Hungarian (maximize)
        min_n_links=1,
    )

    frame0 = sframe_data.data["0"]
    bag = _bag_from_frame(frame0)

    links = asm.extract_best_links(bag, frame0["costs"], trees=None)
    assert len(links) == 2

    endpoints = [{tuple(l.j1.pos), tuple(l.j2.pos)} for l in links]
    assert {(0.0, 0.0), (5.0, 0.0)} in endpoints
    assert {(100.0, 100.0), (110.0, 100.0)} in endpoints

    vals = sorted((l.affinity for l in links), reverse=True)
    assert vals[0] == pytest.approx(0.95, rel=1e-6)
    assert vals[1] == pytest.approx(0.90, rel=1e-6)


def test_extract_best_links_greedy_with_thresholds(assembler_data_single_frame, make_assembler):
    sframe_data = assembler_data_single_frame
    asm = make_assembler(
        sframe_data.data,
        max_n_individuals=1,  # greedy will stop after 1 disjoint pair chosen
        greedy=True,
        pcutoff=0.5,  # conf product must exceed 0.25
        min_affinity=0.5,  # low-affinity pairs excluded
        min_n_links=1,
    )

    frame0 = sframe_data.data["0"]
    bag = _bag_from_frame(frame0)

    links = asm.extract_best_links(bag, frame0["costs"], trees=None)
    assert len(links) == 1

    s = {tuple(links[0].j1.pos), tuple(links[0].j2.pos)}
    assert s in (
        {(0.0, 0.0), (5.0, 0.0)},
        {(100.0, 100.0), (110.0, 100.0)},
    )


@HYPOTHESIS_SETTINGS
@given(
    n=st.integers(min_value=1, max_value=4),
    pcutoff=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_aff=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    conf0=st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=4),
    conf1=st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=4),
)
def test_extract_best_links_greedy_invariants_with_threshold_gates(n, pcutoff, min_aff, conf0, conf1):
    # Normalize confidences to exactly n items
    conf0 = (conf0 + [0.0] * n)[:n]
    conf1 = (conf1 + [0.0] * n)[:n]
    conf0 = np.array(conf0, dtype=float)
    conf1 = np.array(conf1, dtype=float)

    # Random-ish affinity matrix (still stable), in [0,1]
    rng = np.random.default_rng(0)  # deterministic noise
    aff = rng.random((n, n))  # uniform [0,1)
    # Ensure at least one "good" candidate sometimes; otherwise test is vacuously true.
    # We'll only assert gated properties on returned links anyway.
    # But for better coverage, bias the diagonal upward a bit:
    np.fill_diagonal(aff, np.maximum(np.diag(aff), 0.8))
    dist = np.ones((n, n), dtype=float)

    graph = [(0, 1)]
    paf_inds = [0]
    data = {
        "metadata": {"all_joints_names": ["b0", "b1"], "PAFgraph": graph, "PAFinds": paf_inds},
        "0": {},
    }

    asm = Assembler(
        data,
        max_n_individuals=n,
        n_multibodyparts=2,
        greedy=True,
        pcutoff=pcutoff,
        min_affinity=min_aff,
        min_n_links=1,
        method="m1",
    )

    dets0 = [Joint((float(i), 0.0), float(conf0[i]), label=0, idx=i) for i in range(n)]
    dets1 = [Joint((float(i), 1.0), float(conf1[i]), label=1, idx=100 + i) for i in range(n)]
    joints_dict = {0: dets0, 1: dets1}
    costs = {0: {"distance": dist, "m1": aff}}

    links = asm.extract_best_links(joints_dict, costs, trees=None)

    assert len(links) <= n

    used_src = set()
    used_tgt = set()

    for link in links:
        # Invariant 1: affinity gate
        assert link.affinity >= min_aff

        # Invariant 2: pcutoff gate (confidence product)
        assert link.j1.confidence * link.j2.confidence >= pcutoff * pcutoff

        # Invariant 3: disjointness in greedy selection
        assert link.j1.idx not in used_src
        assert link.j2.idx not in used_tgt
        used_src.add(link.j1.idx)
        used_tgt.add(link.j2.idx)


# --------------------------------------------------------------------------------------
# build_assemblies
# --------------------------------------------------------------------------------------


def test_build_assemblies_from_links(assembler_data_single_frame, make_assembler):
    sframe_data = assembler_data_single_frame
    asm = make_assembler(sframe_data.data, greedy=False, min_n_links=1)

    frame0 = sframe_data.data["0"]
    bag = _bag_from_frame(frame0)

    links = asm.extract_best_links(bag, frame0["costs"])
    assemblies, _ = asm.build_assemblies(links)

    assert len(assemblies) == 2
    for a in assemblies:
        assert a.n_links == 1
        assert len(a) == 2
        assert a.affinity == pytest.approx(a._affinity / a.n_links)


# --------------------------------------------------------------------------------------
# _assemble (per-frame) – main path
# --------------------------------------------------------------------------------------


def test__assemble_main_no_calibration_returns_two_assemblies(assembler_data_single_frame, make_assembler):
    sframe_data = assembler_data_single_frame
    asm = make_assembler(
        sframe_data.data,
        greedy=False,
        min_n_links=1,
        max_overlap=0.99,
        window_size=0,
    )

    assemblies, unique = asm._assemble(sframe_data.data["0"], 0)
    assert unique is None
    assert len(assemblies) == 2
    assert all(len(a) == 2 for a in assemblies)


def test__assemble_returns_none_when_no_detections(assembler_data_no_detections, make_assembler):
    nodet_data = assembler_data_no_detections
    asm = make_assembler(nodet_data.data, max_n_individuals=2, n_multibodyparts=2)

    assemblies, unique = asm._assemble(nodet_data.data["0"], 0)
    assert assemblies is None and unique is None


# --------------------------------------------------------------------------------------
# assemble() over multiple frames + KD-tree caching
# --------------------------------------------------------------------------------------


def test_assemble_across_frames_updates_temporal_trees(assembler_data_two_frames_nudged, make_assembler):
    twofr_data = assembler_data_two_frames_nudged
    asm = make_assembler(
        twofr_data.data,
        window_size=1,  # enable temporal memory
        min_n_links=1,
    )

    asm.assemble(chunk_size=0)

    assert 0 in asm._trees or 1 in asm._trees
    assert isinstance(asm.assemblies, dict)
    assert set(asm.assemblies.keys()).issubset({0, 1})


# --------------------------------------------------------------------------------------
# identity_only=True branch
# --------------------------------------------------------------------------------------


def test_identity_only_branch_groups_by_identity(assembler_data_single_frame, scene_copy, make_assembler):
    sframe_data = assembler_data_single_frame

    base = scene_copy
    id0 = np.array([[4.0, 1.0], [1.0, 4.0]])
    id1 = np.array([[4.0, 1.0], [1.0, 4.0]])
    base["identity"] = [id0, id1]
    sframe_data.data["0"] = base

    asm = make_assembler(
        sframe_data.data,
        max_n_individuals=3,
        identity_only=True,
        pcutoff=0.1,
    )

    assemblies, _ = asm._assemble(sframe_data.data["0"], 0)
    assert assemblies is not None
    assert all(len(a) >= 1 for a in assemblies)


# --------------------------------------------------------------------------------------
# Mahalanobis & link probability
# --------------------------------------------------------------------------------------


@dataclass
class _FakeKDE:
    mean: np.ndarray
    inv_cov: np.ndarray
    covariance: np.ndarray
    d: int


def test_calc_assembly_mahalanobis_and_link_probability_with_fake_kde(assembler_data_single_frame, make_assembler):
    sframe_data = assembler_data_single_frame
    asm = make_assembler(sframe_data.data, min_n_links=1)

    j0 = Joint((0.0, 0.0), 1.0, 0, 0)
    j1 = Joint((3.0, 4.0), 1.0, 1, 1)
    link = Link(j0, j1, 1.0)

    a = Assembly(size=2)
    a.add_link(link)

    fake = _FakeKDE(
        mean=np.array([25.0]),
        inv_cov=np.array([[1.0]]),
        covariance=np.array([[1.0]]),
        d=1,
    )
    asm._kde = fake
    asm.safe_edge = True

    d = asm.calc_assembly_mahalanobis_dist(a)
    assert d == pytest.approx(0.0, abs=1e-6)

    p = asm.calc_link_probability(link)
    assert p == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------------------
# I/O: pickle / h5
# --------------------------------------------------------------------------------------


def test_to_pickle_and_from_pickle(tmp_path, assembler_data_single_frame, make_assembler, assembler_graph_and_pafs):
    sframe_data = assembler_data_single_frame
    asm = make_assembler(sframe_data.data, min_n_links=1)
    assemblies, _ = asm._assemble(sframe_data.data["0"], 0)
    asm.assemblies = {0: assemblies}

    pkl = tmp_path / "assemb.pkl"
    asm.to_pickle(str(pkl))

    new_asm = Assembler.empty(
        max_n_individuals=2,
        n_multibodyparts=2,
        n_uniquebodyparts=0,
        graph=sframe_data.graph,
        paf_inds=sframe_data.paf_inds,
    )
    new_asm.from_pickle(str(pkl))

    assert 0 in new_asm.assemblies
    assert isinstance(new_asm.assemblies[0], list)


@pytest.mark.skipif(
    pytest.importorskip("tables", reason="PyTables required for HDF5") is None,
    reason="requires PyTables",
)
def test_to_h5_roundtrip(tmp_path, assembler_data_single_frame, make_assembler):
    sframe_data = assembler_data_single_frame

    asm = make_assembler(sframe_data.data, min_n_links=1)
    assemblies, _ = asm._assemble(sframe_data.data["0"], 0)
    asm.assemblies = {0: assemblies}

    h5 = tmp_path / "assemb.h5"
    asm.to_h5(str(h5))

    df = pd.read_hdf(str(h5), key="ass")
    assert df.shape[0] == 1
    assert df.columns.nlevels == 4
    assert set(df.columns.get_level_values("coords")) == {"x", "y", "likelihood"}
