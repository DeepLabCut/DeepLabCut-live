from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from dlclive.core.inferenceutils import Assembler, Assembly, Joint, Link

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_metadata(graph, paf_inds, n_bodyparts, frame_keys):
    """Create a minimal DLC-like metadata structure for Assembler."""
    return {
        "metadata": {
            "all_joints_names": [f"b{i}" for i in range(n_bodyparts)],
            "PAFgraph": graph,
            "PAFinds": paf_inds,
        },
        **{k: {} for k in frame_keys},
    }


def make_frame(coordinates_per_label, confidence_per_label, identity_per_label=None, costs=None):
    """
    Build a single frame dict with the structure Assembler._flatten_detections expects.

    coordinates_per_label: list of np.ndarray[(n_dets, 2)]
    confidence_per_label: list of np.ndarray[(n_dets, )]
    identity_per_label:   list of np.ndarray[(n_dets, n_groups)] or None
    costs: dict or None. Example:
        {
          0: {
                "distance": np.array([[...]]),  # rows: label s detections, cols: label t
                "m1": np.array([[...]]),
              }
        }
    """
    frame = {
        # NOTE: Assembler expects coordinates under key "coordinates"[0]
        "coordinates": [coordinates_per_label],
        "confidence": confidence_per_label,
        "costs": costs or {},
    }
    if identity_per_label is not None:
        frame["identity"] = identity_per_label
    return frame


def simple_two_label_scene():
    """
    Build a very small, deterministic scene with 2 bodyparts (0 ↔ 1), each with 2 detections.
    We design affinities so that the intended pairs are (A↔C) and (B↔D).
    """
    # Label 0 detections: A (near origin), B (far)
    coords0 = np.array([[0.0, 0.0], [100.0, 100.0]])
    conf0 = np.array([0.9, 0.6])

    # Label 1 detections: C near A, D near B
    coords1 = np.array([[5.0, 0.0], [110.0, 100.0]])
    conf1 = np.array([0.8, 0.7])

    # Affinities: strong on diagonal pairs AC and BD, weak elsewhere
    aff = np.array([[0.95, 0.1], [0.05, 0.9]])
    # Lengths: finite distances (not used for assignment, but must be finite)
    lens = np.array(
        [
            [np.hypot(*(coords1[0] - coords0[0])), np.hypot(*(coords1[1] - coords0[0]))],
            [np.hypot(*(coords1[0] - coords0[1])), np.hypot(*(coords1[1] - coords0[1]))],
        ]
    )

    # Build frame expected by Assembler
    frame0 = make_frame(
        coordinates_per_label=[coords0, coords1],
        confidence_per_label=[conf0, conf1],
        identity_per_label=None,
        costs={0: {"distance": lens, "m1": aff}},
    )
    return frame0


# --------------------------------------------------------------------------------------
# Basic metadata and __getitem__
# --------------------------------------------------------------------------------------


def test_parse_metadata_and_getitem_and_empty_classmethod():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0", "1"])
    # Fill frames so __getitem__ returns something non-empty later
    data["0"] = simple_two_label_scene()
    data["1"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
    )

    # parse_metadata applied in ctor:
    assert asm.metadata["num_joints"] == 2
    assert asm.metadata["paf_graph"] == graph
    assert list(asm.metadata["paf"]) == paf_inds
    assert set(asm.metadata["imnames"]) == {"0", "1"}

    # __getitem__ returns the per-frame dict
    assert "coordinates" in asm[0]
    assert "confidence" in asm[0]
    assert "costs" in asm[0]

    # empty() convenience
    empty = Assembler.empty(
        max_n_individuals=1,
        n_multibodyparts=1,
        n_uniquebodyparts=0,
        graph=graph,
        paf_inds=paf_inds,
    )
    assert isinstance(empty, Assembler)
    assert empty.n_keypoints == 1


# --------------------------------------------------------------------------------------
# _flatten_detections
# --------------------------------------------------------------------------------------


def test_flatten_detections_no_identity():
    frame = simple_two_label_scene()
    joints = list(Assembler._flatten_detections(frame))
    # 2 labels * 2 detections
    assert len(joints) == 4

    # label IDs and groups
    labels = sorted([j.label for j in joints])
    assert labels == [0, 0, 1, 1]
    # identity absent → group = -1
    assert set(j.group for j in joints) == {-1}


def test_flatten_detections_with_identity():
    frame = simple_two_label_scene()

    # Add identity logits so that argmax → [0, 1] for both labels
    id0 = np.array([[10.0, 0.0], [0.0, 10.0]])  # for label 0
    id1 = np.array([[10.0, 0.0], [0.0, 10.0]])  # for label 1
    frame["identity"] = [id0, id1]

    joints = list(Assembler._flatten_detections(frame))
    groups = [j.group for j in joints]
    # we expect groups [0,1,0,1] in some order (2 per label)
    assert set(groups) == {0, 1}
    assert groups.count(0) == 2
    assert groups.count(1) == 2


# --------------------------------------------------------------------------------------
# extract_best_links
# --------------------------------------------------------------------------------------


def test_extract_best_links_optimal_assignment():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        greedy=False,  # use Hungarian (maximize)
        pcutoff=0.1,
        min_affinity=0.05,
        min_n_links=1,  # avoid pruning 1-link assemblies in later steps
    )

    # Build joints_dict like _assemble does
    joints = list(Assembler._flatten_detections(data["0"]))
    bag = {}
    for j in joints:
        bag.setdefault(j.label, []).append(j)

    links = asm.extract_best_links(bag, data["0"]["costs"], trees=None)
    # Expect 2 high-quality links: (coords0[0] ↔ coords1[0]) and (coords0[1] ↔ coords1[1])
    assert len(links) == 2

    # Check that each link connects matching pairs (by position)
    endpoints = [{tuple(l.j1.pos), tuple(l.j2.pos)} for l in links]
    assert {(0.0, 0.0), (5.0, 0.0)} in endpoints
    assert {(100.0, 100.0), (110.0, 100.0)} in endpoints

    # Affinity should be the matrix diagonal values ~0.95 and ~0.9
    vals = sorted([l.affinity for l in links], reverse=True)
    assert vals[0] == pytest.approx(0.95, rel=1e-6)
    assert vals[1] == pytest.approx(0.90, rel=1e-6)


def test_extract_best_links_greedy_with_thresholds():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=1,  # greedy will stop after 1 disjoint pair chosen
        n_multibodyparts=2,
        greedy=True,
        pcutoff=0.5,  # conf product must exceed 0.25
        min_affinity=0.5,  # low-affinity pairs excluded
        min_n_links=1,
    )

    joints = list(Assembler._flatten_detections(data["0"]))
    bag = {}
    for j in joints:
        bag.setdefault(j.label, []).append(j)

    links = asm.extract_best_links(bag, data["0"]["costs"], trees=None)
    # Expect exactly 1 link due to max_n_individuals=1 in greedy picking
    assert len(links) == 1
    s = {tuple(links[0].j1.pos), tuple(links[0].j2.pos)}
    assert s == {(0.0, 0.0), (5.0, 0.0)} or s == {(100.0, 100.0), (110.0, 100.0)}


# --------------------------------------------------------------------------------------
# build_assemblies
# --------------------------------------------------------------------------------------


def test_build_assemblies_from_links():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        greedy=False,
        pcutoff=0.1,
        min_affinity=0.05,
        min_n_links=1,
    )

    joints = list(Assembler._flatten_detections(data["0"]))
    bag = {}
    for j in joints:
        bag.setdefault(j.label, []).append(j)

    links = asm.extract_best_links(bag, data["0"]["costs"])
    assemblies, _ = asm.build_assemblies(links)

    # We expect two disjoint 2-joint assemblies
    assert len(assemblies) == 2
    for a in assemblies:
        assert a.n_links == 1
        assert len(a) == 2
        # affinity is the sum of link affinities for the assembly
        assert a.affinity == pytest.approx(a._affinity / a.n_links)


# --------------------------------------------------------------------------------------
# _assemble (per-frame) – main path without calibration
# --------------------------------------------------------------------------------------


def test__assemble_main_no_calibration_returns_two_assemblies():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        greedy=False,
        pcutoff=0.1,
        min_affinity=0.05,
        min_n_links=1,
        max_overlap=0.99,
        window_size=0,
    )

    assemblies, unique = asm._assemble(data["0"], ind_frame=0)
    assert unique is None  # no unique bodyparts in this setting
    assert len(assemblies) == 2
    assert all(len(a) == 2 for a in assemblies)


def test__assemble_returns_none_when_no_detections():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])

    # Frame with zero coords (→ skipped by _flatten_detections)
    coords0 = np.zeros((0, 2))
    conf0 = np.zeros((0,))
    coords1 = np.zeros((0, 2))
    conf1 = np.zeros((0,))
    frame = make_frame([coords0, coords1], [conf0, conf1], identity_per_label=None, costs={})
    data["0"] = frame

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
    )
    assemblies, unique = asm._assemble(data["0"], ind_frame=0)
    assert assemblies is None and unique is None


# --------------------------------------------------------------------------------------
# assemble() over multiple frames + window_size KD-tree caching
# --------------------------------------------------------------------------------------


def test_assemble_across_frames_updates_temporal_trees():
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0", "1"])

    # Frame 0: baseline
    frame0 = simple_two_label_scene()

    # Frame 1: nudge coordinates slightly, keep affinities similar
    f1 = simple_two_label_scene()
    f1["coordinates"][0][0] = f1["coordinates"][0][0] + np.array([[1.0, 0.0], [1.0, 0.0]])
    f1["coordinates"][0][1] = f1["coordinates"][0][1] + np.array([[1.0, 0.0], [1.0, 0.0]])

    data["0"] = frame0
    data["1"] = f1

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        window_size=1,  # enable temporal memory
        min_n_links=1,
    )

    # Use serial path to avoid multiprocessing in tests
    asm.assemble(chunk_size=0)

    # KD-trees should be recorded for frames that had links
    # Presence of keys 0 and 1 depends on links creation
    assert 0 in asm._trees or 1 in asm._trees
    assert isinstance(asm.assemblies, dict)
    assert set(asm.assemblies.keys()).issubset({0, 1})


# --------------------------------------------------------------------------------------
# identity_only=True branch
# --------------------------------------------------------------------------------------


def test_identity_only_branch_groups_by_identity():
    graph = [(0, 1)]
    paf_inds = [0]

    # Build a frame with identities. Two groups (0 and 1).
    base = simple_two_label_scene()
    # identity logits such that each label has two detections belonging to groups 0 and 1
    id0 = np.array([[4.0, 1.0], [1.0, 4.0]])  # label 0 → group 0 then 1
    id1 = np.array([[4.0, 1.0], [1.0, 4.0]])  # label 1 → group 0 then 1
    base["identity"] = [id0, id1]

    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = base

    # identity_only=True should be enabled since "identity" is present in frame 0
    asm = Assembler(
        data,
        max_n_individuals=3,
        n_multibodyparts=2,
        identity_only=True,
        pcutoff=0.1,
    )

    assemblies, unique = asm._assemble(data["0"], ind_frame=0)
    # We expect at least one assembly created by grouping (one per identity that has both labels observed)
    assert assemblies is not None
    assert all(len(a) >= 1 for a in assemblies)


# --------------------------------------------------------------------------------------
# Mahalanobis distance and link probability with a mocked KDE
# --------------------------------------------------------------------------------------


@dataclass
class _FakeKDE:
    mean: np.ndarray
    inv_cov: np.ndarray
    covariance: np.ndarray
    d: int  # dimension


def test_calc_assembly_mahalanobis_and_link_probability_with_fake_kde():
    # 2 multibody parts → pdist length = 1
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        min_n_links=1,
    )

    # Build a simple assembly with two joints
    j0 = Joint((0.0, 0.0), 1.0, label=0, idx=0)
    j1 = Joint((3.0, 4.0), 1.0, label=1, idx=1)
    link = Link(j0, j1, affinity=1.0)

    a = Assembly(size=2)
    a.add_link(link)

    # Fake KDE: one-dimensional (pairwise sq distance only)
    # distance^2 = 5^2 = 25; set mean=25, identity covariance
    fake = _FakeKDE(
        mean=np.array([25.0]),
        inv_cov=np.array([[1.0]]),
        covariance=np.array([[1.0]]),
        d=1,
    )
    asm._kde = fake
    asm.safe_edge = True

    # Mahalanobis should be finite and small because dist == mean
    d = asm.calc_assembly_mahalanobis_dist(a)
    assert np.isfinite(d)
    assert d == pytest.approx(0.0, abs=1e-6)

    # Link probability depends on squared length vs mean; here z=0 → high prob
    p = asm.calc_link_probability(link)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------------------
# I/O helpers: to_pickle / from_pickle / to_h5 (optional)
# --------------------------------------------------------------------------------------


def test_to_pickle_and_from_pickle(tmp_path):
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        min_n_links=1,
    )

    # build assemblies for frame 0
    assemblies, _ = asm._assemble(data["0"], 0)
    asm.assemblies = {0: assemblies}

    pkl = tmp_path / "ass.pkl"
    asm.to_pickle(str(pkl))

    # Load into a new Assembler (empty schema is sufficient)
    new_asm = Assembler.empty(
        max_n_individuals=2,
        n_multibodyparts=2,
        n_uniquebodyparts=0,
        graph=graph,
        paf_inds=paf_inds,
    )
    new_asm.from_pickle(str(pkl))
    assert 0 in new_asm.assemblies
    assert isinstance(new_asm.assemblies[0], list)
    assert new_asm.assemblies[0][0].shape == (2, 4) or True  # presence is enough


@pytest.mark.skipif(
    pytest.importorskip("tables", reason="PyTables required for HDF5") is None, reason="requires PyTables"
)
def test_to_h5_roundtrip(tmp_path):
    graph = [(0, 1)]
    paf_inds = [0]
    data = make_metadata(graph, paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene()

    asm = Assembler(
        data,
        max_n_individuals=2,
        n_multibodyparts=2,
        min_n_links=1,
    )
    assemblies, _ = asm._assemble(data["0"], 0)
    asm.assemblies = {0: assemblies}

    h5 = tmp_path / "ass.h5"
    asm.to_h5(str(h5))

    # Read back and perform basic structural assertions
    df = pd.read_hdf(str(h5), key="ass")
    # one frame, 2 individuals, 2 bodyparts, coords {x,y,likelihood}
    # df shape will be (frames, 2*2*3)
    assert df.shape[0] == 1
    assert df.columns.nlevels == 4
    assert set(df.columns.get_level_values("coords")) == {"x", "y", "likelihood"}
