from __future__ import annotations

import copy
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from dlclive.core.inferenceutils import Assembler


# --------------------------------------------------------------------------------------
# Headless display fixture
# --------------------------------------------------------------------------------------
@pytest.fixture
def headless_display_env(monkeypatch):
    """
    Patch dlclive.display so tkinter + ImageTk are replaced by MagicMocks.

    Returns an object with:
      - mod: the imported dlclive.display module
      - tk_ctor: MagicMock constructor for Tk
      - tk: MagicMock instance for the window
      - label_ctor: MagicMock constructor for Label
      - label: MagicMock instance for the label widget
      - photo_ctor: MagicMock function for ImageTk.PhotoImage
      - photo: MagicMock instance representing created image
    """
    import dlclive.display as display_mod

    # Ensure display path is enabled
    monkeypatch.setattr(display_mod, "_TKINTER_AVAILABLE", True, raising=False)

    # Tk / Label mocks
    tk = MagicMock(name="TkInstance")
    tk_ctor = MagicMock(name="Tk", return_value=tk)

    label = MagicMock(name="LabelInstance")
    label_ctor = MagicMock(name="Label", return_value=label)

    # ImageTk.PhotoImage mock
    photo = MagicMock(name="PhotoImageInstance")
    photo_ctor = MagicMock(name="PhotoImage", return_value=photo)

    class FakeImageTkModule:
        PhotoImage = photo_ctor

    monkeypatch.setattr(display_mod, "Tk", tk_ctor, raising=False)
    monkeypatch.setattr(display_mod, "Label", label_ctor, raising=False)
    monkeypatch.setattr(display_mod, "ImageTk", FakeImageTkModule, raising=False)

    return SimpleNamespace(
        mod=display_mod,
        tk_ctor=tk_ctor,
        tk=tk,
        label_ctor=label_ctor,
        label=label,
        photo_ctor=photo_ctor,
        photo=photo,
    )


# --------------------------------------------------------------------------------------
# Assembler/assembly test fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def assembler_graph_and_pafs() -> SimpleNamespace:
    """Standard 2‑joint graph used throughout the test suite."""
    graph = [(0, 1)]
    paf_inds = [0]
    return SimpleNamespace(graph=graph, paf_inds=paf_inds)


@pytest.fixture
def make_assembler_metadata() -> Callable[..., dict[str, Any]]:
    """Return a factory that builds minimal Assembler metadata dictionaries."""

    def _factory(graph, paf_inds, n_bodyparts, frame_keys):
        return {
            "metadata": {
                "all_joints_names": [f"b{i}" for i in range(n_bodyparts)],
                "PAFgraph": graph,
                "PAFinds": paf_inds,
            },
            **{k: {} for k in frame_keys},
        }

    return _factory


@pytest.fixture
def make_assembler_frame() -> Callable[..., dict[str, Any]]:
    """Return a factory that builds a frame dict compatible with _flatten_detections."""

    def _factory(
        coordinates_per_label,
        confidence_per_label,
        identity_per_label=None,
        costs=None,
    ):
        frame = {
            "coordinates": [coordinates_per_label],
            "confidence": confidence_per_label,
            "costs": costs or {},
        }
        if identity_per_label is not None:
            frame["identity"] = identity_per_label
        return frame

    return _factory


@pytest.fixture
def simple_two_label_scene(make_assembler_frame) -> dict[str, Any]:
    """Deterministic scene with predictable affinities for testing."""
    coords0 = np.array([[0.0, 0.0], [100.0, 100.0]])
    coords1 = np.array([[5.0, 0.0], [110.0, 100.0]])
    conf0 = np.array([0.9, 0.6])
    conf1 = np.array([0.8, 0.7])

    aff = np.array([[0.95, 0.1], [0.05, 0.9]])

    lens = np.array(
        [
            [np.hypot(*(coords1[0] - coords0[0])), np.hypot(*(coords1[1] - coords0[0]))],
            [np.hypot(*(coords1[0] - coords0[1])), np.hypot(*(coords1[1] - coords0[1]))],
        ]
    )

    return make_assembler_frame(
        coordinates_per_label=[coords0, coords1],
        confidence_per_label=[conf0, conf1],
        identity_per_label=None,
        costs={0: {"distance": lens, "m1": aff}},
    )


@pytest.fixture
def scene_copy(simple_two_label_scene) -> dict[str, Any]:
    """Return a deep copy of the simple_two_label_scene fixture."""
    return copy.deepcopy(simple_two_label_scene)


@pytest.fixture
def assembler_data(
    assembler_graph_and_pafs,
    make_assembler_metadata,
    simple_two_label_scene,
) -> SimpleNamespace:
    """Full metadata + two identical frames ('0', '1')."""
    paf = assembler_graph_and_pafs
    data = make_assembler_metadata(paf.graph, paf.paf_inds, n_bodyparts=2, frame_keys=["0", "1"])
    data["0"] = simple_two_label_scene
    data["1"] = simple_two_label_scene
    return SimpleNamespace(data=data, graph=paf.graph, paf_inds=paf.paf_inds)


@pytest.fixture
def assembler_data_single_frame(
    assembler_graph_and_pafs,
    make_assembler_metadata,
    simple_two_label_scene,
) -> SimpleNamespace:
    """Metadata + a single frame ('0'). Used by most tests."""
    paf = assembler_graph_and_pafs
    data = make_assembler_metadata(paf.graph, paf.paf_inds, n_bodyparts=2, frame_keys=["0"])
    data["0"] = simple_two_label_scene
    return SimpleNamespace(data=data, graph=paf.graph, paf_inds=paf.paf_inds)


@pytest.fixture
def assembler_data_two_frames_nudged(
    assembler_graph_and_pafs,
    make_assembler_metadata,
    simple_two_label_scene,
) -> SimpleNamespace:
    """Two frames where frame '1' is a nudged copy of frame '0'."""
    paf = assembler_graph_and_pafs
    data = make_assembler_metadata(paf.graph, paf.paf_inds, n_bodyparts=2, frame_keys=["0", "1"])

    frame0 = simple_two_label_scene
    frame1 = copy.deepcopy(simple_two_label_scene)
    frame1["coordinates"][0][0] += np.array([[1.0, 0.0], [1.0, 0.0]])
    frame1["coordinates"][0][1] += np.array([[1.0, 0.0], [1.0, 0.0]])

    data["0"] = frame0
    data["1"] = frame1
    return SimpleNamespace(data=data, graph=paf.graph, paf_inds=paf.paf_inds)


@pytest.fixture
def assembler_data_no_detections(
    assembler_graph_and_pafs,
    make_assembler_metadata,
    make_assembler_frame,
) -> SimpleNamespace:
    """Metadata + a single frame ('0') with zero detections for both labels."""
    paf = assembler_graph_and_pafs
    data = make_assembler_metadata(paf.graph, paf.paf_inds, n_bodyparts=2, frame_keys=["0"])

    frame = make_assembler_frame(
        coordinates_per_label=[np.zeros((0, 2)), np.zeros((0, 2))],
        confidence_per_label=[np.zeros((0,)), np.zeros((0,))],
        identity_per_label=None,
        costs={},
    )
    data["0"] = frame
    # return data, graph, paf_inds
    return SimpleNamespace(data=data, graph=paf.graph, paf_inds=paf.paf_inds)


@pytest.fixture
def make_assembler() -> Callable[..., Assembler]:
    """
    Factory to create an Assembler with sensible defaults for this test suite.
    Override any parameter per-test via kwargs.
    """

    def _factory(data: dict[str, Any], **overrides) -> Assembler:
        defaults = dict(
            max_n_individuals=2,
            n_multibodyparts=2,
            min_n_links=1,
            pcutoff=0.1,
            min_affinity=0.05,
        )
        defaults.update(overrides)
        return Assembler(data, **defaults)

    return _factory


# --------------------------------------------------------------------------------------
# Assembly / Joint / Link test fixtures
# --------------------------------------------------------------------------------------
from dlclive.core.inferenceutils import Assembly, Joint, Link  # noqa: E402


@pytest.fixture
def make_assembly() -> Callable[..., Assembly]:
    """Factory to create an Assembly with the given size."""

    def _factory(size: int) -> Assembly:
        return Assembly(size=size)

    return _factory


@pytest.fixture
def make_joint() -> Callable[..., Joint]:
    """Factory to create a Joint with sensible defaults."""

    def _factory(
        pos=(0.0, 0.0),
        confidence: float = 1.0,
        label: int = 0,
        idx: int = 0,
        group: int = -1,
    ) -> Joint:
        return Joint(pos=pos, confidence=confidence, label=label, idx=idx, group=group)

    return _factory


@pytest.fixture
def make_link() -> Callable[..., Link]:
    """Factory to create a Link between two joints."""

    def _factory(j1: Joint, j2: Joint, affinity: float = 1.0) -> Link:
        return Link(j1, j2, affinity=affinity)

    return _factory


@pytest.fixture
def two_overlap_assemblies(make_assembly) -> tuple[Assembly, Assembly]:
    """Two assemblies with partial overlap used by intersection tests."""
    assemb1 = make_assembly(2)
    assemb1.data[0, :2] = [0, 0]
    assemb1.data[1, :2] = [10, 10]
    assemb1._visible.update({0, 1})

    assemb2 = make_assembly(2)
    assemb2.data[0, :2] = [5, 5]
    assemb2.data[1, :2] = [15, 15]
    assemb2._visible.update({0, 1})
    return assemb1, assemb2


@pytest.fixture
def soft_identity_assembly(make_assembly) -> Assembly:
    """Assembly configured for soft_identity tests."""
    assemb = make_assembly(3)
    assemb.data[:] = np.nan
    assemb.data[0] = [0, 0, 1.0, 0]
    assemb.data[1] = [5, 5, 0.5, 0]
    assemb.data[2] = [10, 10, 1.0, 1]
    assemb._visible = {0, 1, 2}
    return assemb


@pytest.fixture
def four_joint_chain(make_joint, make_link) -> SimpleNamespace:
    """Four joints and two links: (0-1) and (2-3)."""
    j0 = make_joint((0, 0), 1.0, label=0, idx=10)
    j1 = make_joint((1, 0), 1.0, label=1, idx=11)
    j2 = make_joint((2, 0), 1.0, label=2, idx=12)
    j3 = make_joint((3, 0), 1.0, label=3, idx=13)
    l01 = make_link(j0, j1, affinity=0.5)
    l23 = make_link(j2, j3, affinity=0.8)
    return SimpleNamespace(j0=j0, j1=j1, j2=j2, j3=j3, l01=l01, l23=l23)
