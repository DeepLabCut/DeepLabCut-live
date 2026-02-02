import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dlclive.core.inferenceutils import Assembly

HYPOTHESIS_SETTINGS = settings(max_examples=200, deadline=None)


# ---------------------------
# Basic construction
# ---------------------------
def test_assembly_init(make_assembly):
    assemb = make_assembly(size=5)
    assert assemb.data.shape == (5, 4)

    # col 0,1,3 are NaN, col 2 is confidence=0
    # this is due to confidence being initialized to 0
    # by default in Assembly.__init__
    assert np.isnan(assemb.data[:, 0]).all()
    assert np.isnan(assemb.data[:, 1]).all()
    assert (assemb.data[:, 2] == 0).all()
    assert np.isnan(assemb.data[:, 3]).all()

    assert assemb._affinity == 0
    assert assemb._links == []
    assert assemb._visible == set()
    assert assemb._idx == set()


# ---------------------------
# from_array
# ---------------------------
@HYPOTHESIS_SETTINGS
@given(
    st.integers(min_value=1, max_value=30).flatmap(
        lambda n_rows: st.sampled_from([2, 3]).flatmap(
            lambda n_cols: arrays(
                dtype=np.float64,
                shape=(n_rows, n_cols),
                elements=st.floats(allow_infinity=False, allow_nan=True, width=32),
            )
        )
    )
)
def test_from_array_invariants(arr):
    n_rows, n_cols = arr.shape

    assemb = Assembly.from_array(arr.copy())
    assert assemb.data.shape == (n_rows, 4)

    # Row is "valid" iff it has no NaN among the provided columns
    row_valid = ~np.isnan(arr).any(axis=1)
    visible = set(np.flatnonzero(row_valid).tolist())
    assert assemb._visible == visible

    out = assemb.data

    # For invalid rows: x/y must be NaN
    assert np.all(np.isnan(out[~row_valid, 0]))
    assert np.all(np.isnan(out[~row_valid, 1]))

    # Confidence behavior differs depending on number of columns
    if n_cols == 2:
        # XY-only input: confidence starts at 0 and is set to 1 for visible rows only
        assert np.all(out[row_valid, 2] == pytest.approx(1.0))
        assert np.all(out[~row_valid, 2] == pytest.approx(0.0))
    else:
        # XY+confidence input: confidence is preserved for visible rows
        assert np.allclose(out[row_valid, 2], arr[row_valid, 2], equal_nan=False)
        # Invalid rows become NaN in all provided columns, including confidence
        assert np.all(np.isnan(out[~row_valid, 2]))

    # Visible rows preserve xy
    assert np.allclose(out[row_valid, :2], arr[row_valid, :2], equal_nan=False)


def test_assembly_from_array_with_nans():
    arr = np.array(
        [
            [10.0, 20.0, 0.9],
            [np.nan, 5.0, 0.8],  # one NaN → entire row becomes NaN
        ]
    )
    assemb = Assembly.from_array(arr.copy())

    assert np.allclose(assemb.data[0], [10.0, 20.0, 0.9, np.nan], equal_nan=True)
    assert np.isnan(assemb.data[1]).all()

    # visible only includes fully non-NaN rows
    assert assemb._visible == {0}


# ---------------------------
# extent, area, xy
# ---------------------------
@HYPOTHESIS_SETTINGS
@given(
    coords=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 30), st.just(2)),
        elements=st.floats(allow_nan=True, allow_infinity=False, width=32),
    )
)
def test_extent_matches_visible_points(coords):
    xy = coords.copy()
    a = Assembly(size=xy.shape[0])
    a.data[:] = np.nan
    a.data[:, :2] = xy
    a._visible = set(np.flatnonzero(~np.isnan(xy).any(axis=1)).tolist())

    visible_mask = ~np.isnan(coords).any(axis=1)
    assume(visible_mask.any())

    expected = np.array(
        [
            coords[visible_mask, 0].min(),
            coords[visible_mask, 1].min(),
            coords[visible_mask, 0].max(),
            coords[visible_mask, 1].max(),
        ]
    )
    assert np.allclose(a.extent, expected)
    assert a.area >= 0


# ---------------------------
# add_joint / remove_joint
# ---------------------------
def test_add_joint_and_remove_joint(make_assembly, make_joint):
    assemb = make_assembly(size=3)
    j0 = make_joint(pos=(1.0, 2.0), confidence=0.5, label=0, idx=10)
    j1 = make_joint(pos=(3.0, 4.0), confidence=0.8, label=1, idx=11)

    # adding first joint
    assert assemb.add_joint(j0) is True
    assert assemb._visible == {0}
    assert assemb._idx == {10}
    assert np.allclose(assemb.data[0], [1.0, 2.0, 0.5, j0.group])

    # adding second joint
    assert assemb.add_joint(j1) is True
    assert assemb._visible == {0, 1}
    assert assemb._idx == {10, 11}

    # adding same joint label again → ignored
    assert assemb.add_joint(j0) is False

    # removing joint
    assert assemb.remove_joint(j1) is True
    assert assemb._visible == {0}
    assert assemb._idx == {10}
    assert np.isnan(assemb.data[1]).all()

    # remove nonexistent → False
    assert assemb.remove_joint(j1) is False


# ---------------------------
# add_link (simple)
# ---------------------------
def test_add_link_adds_joints_and_affinity(make_assembly, make_joint, make_link):
    assemb = make_assembly(size=3)

    j0 = make_joint(pos=(0.0, 0.0), confidence=1.0, label=0, idx=100)
    j1 = make_joint(pos=(1.0, 0.0), confidence=1.0, label=1, idx=101)
    link = make_link(j0, j1, affinity=0.7)

    # New link → adds both joints
    result = assemb.add_link(link)
    assert result is True
    assert assemb.n_links == 1
    assert assemb._affinity == pytest.approx(0.7)
    assert assemb._visible == {0, 1}
    assert assemb._idx == {100, 101}

    # Add same link again → both idx already present → only increases affinity, no new joints
    result = assemb.add_link(link)
    assert result is False  # as per code path
    assert assemb.n_links == 2  # link appended again
    assert assemb._affinity == pytest.approx(1.4)  # 0.7 + 0.7


# ---------------------------
# intersection_with
# ---------------------------
def test_intersection_with_partial_overlap(two_overlap_assemblies):
    ass1, ass2 = two_overlap_assemblies
    assert ass1.intersection_with(ass2) == pytest.approx(0.5)


# ---------------------------
# confidence property
# ---------------------------
def test_confidence_property(make_assembly):
    assemb = make_assembly(size=3)
    assemb.data[:] = np.nan
    assemb.data[:, 2] = [0.2, 0.4, np.nan]  # mean of finite = (0.2+0.4)/2 = 0.3
    assert assemb.confidence == pytest.approx(0.3)

    assemb.confidence = 0.9
    assert np.allclose(assemb.data[:, 2], [0.9, 0.9, 0.9], equal_nan=True)


# ---------------------------
# soft_identity
# ---------------------------
def test_soft_identity_simple(soft_identity_assembly):
    assemb = soft_identity_assembly
    soft = assemb.soft_identity
    assert set(soft.keys()) == {0, 1}
    s0, s1 = soft[0], soft[1]
    assert pytest.approx(s0 + s1) == 1.0
    assert s1 > s0


# ---------------------------
# intersection operator: __contains__
# ---------------------------
def test_contains_checks_shared_idx(make_assembly, make_joint):
    ass1 = make_assembly(size=3)
    ass2 = make_assembly(size=3)

    j0 = make_joint((0, 0), confidence=1.0, label=0, idx=10)
    j1 = make_joint((1, 1), confidence=1.0, label=1, idx=99)

    ass1.add_joint(j0)
    ass2.add_joint(j1)

    # different idx sets → no intersection
    assert (ass2 in ass1) is False

    ass2.add_joint(j0)
    # now share idx=10
    assert (ass2 in ass1) is True


# ---------------------------
# assembly addition (__add__)
# ---------------------------
def test_assembly_addition_combines_links(make_assembly, four_joint_chain):
    a1 = make_assembly(size=4)
    a2 = make_assembly(size=4)

    chain = four_joint_chain

    a1.add_link(chain.l01)
    a2.add_link(chain.l23)

    # now they share NO joints → addition should succeed
    result = a1 + a2

    assert result.n_links == 2
    assert result._affinity == pytest.approx(1.3)

    # original assemblies unchanged
    assert a1.n_links == 1
    assert a2.n_links == 1

    # now purposely make them share a joint → should raise
    a2.add_joint(chain.j0)
    with pytest.raises(ArithmeticError):
        _ = a1 + a2
