import numpy as np
import pytest

from dlclive.core.inferenceutils import Assembly, Joint, Link

# ---------------------------
# Basic construction
# ---------------------------


def test_assembly_init():
    assemb = Assembly(size=5)
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


def test_assembly_from_array_basic_xy_only():
    arr = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )
    assemb = Assembly.from_array(arr.copy())

    # full shape (n_bodyparts, 4)
    assert assemb.data.shape == (2, 4)

    # xy preserved
    assert np.allclose(assemb.data[:, :2], arr)

    # confidence auto-set to 1 where xy is present
    assert np.allclose(assemb.data[:, 2], np.array([1.0, 1.0]))

    # labels visible
    assert assemb._visible == {0, 1}


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
# add_joint / remove_joint
# ---------------------------


def test_add_joint_and_remove_joint():
    assemb = Assembly(size=3)
    j0 = Joint(pos=(1.0, 2.0), confidence=0.5, label=0, idx=10)
    j1 = Joint(pos=(3.0, 4.0), confidence=0.8, label=1, idx=11)

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


def test_add_link_adds_joints_and_affinity():
    assemb = Assembly(size=3)

    j0 = Joint(pos=(0.0, 0.0), confidence=1.0, label=0, idx=100)
    j1 = Joint(pos=(1.0, 0.0), confidence=1.0, label=1, idx=101)
    link = Link(j0, j1, affinity=0.7)

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
# extent, area, xy
# ---------------------------


def test_extent_and_area():
    assemb = Assembly(size=3)
    # manually set data: [x, y, conf, group]
    assemb.data[:] = np.nan
    assemb.data[0, :2] = [10, 10]
    assemb.data[1, :2] = [20, 40]
    assemb._visible.update({0, 1})

    # extent = (min_x, min_y, max_x, max_y)
    assert np.allclose(assemb.extent, [10, 10, 20, 40])

    # area = dx * dy = (20-10) * (40-10) = 10 * 30
    assert assemb.area == pytest.approx(300)


# ---------------------------
# intersection_with
# ---------------------------


def test_intersection_with_partial_overlap():
    ass1 = Assembly(size=2)
    ass1.data[0, :2] = [0, 0]
    ass1.data[1, :2] = [10, 10]
    ass1._visible.update({0, 1})

    ass2 = Assembly(size=2)
    ass2.data[0, :2] = [5, 5]
    ass2.data[1, :2] = [15, 15]
    ass2._visible.update({0, 1})

    # They overlap in a square of area 5x5 around (5,5)-(10,10).
    # Each assembly has 2 points. Points inside overlap:
    # ass1: both (0,0) no, (10,10) yes → 1 / 2 = 0.5
    # ass2: (5,5) yes, (15,15) no → 1 / 2 = 0.5
    assert ass1.intersection_with(ass2) == pytest.approx(0.5)


# ---------------------------
# confidence property
# ---------------------------


def test_confidence_property():
    assemb = Assembly(size=3)
    assemb.data[:] = np.nan
    assemb.data[:, 2] = [0.2, 0.4, np.nan]  # mean of finite = (0.2+0.4)/2 = 0.3
    assert assemb.confidence == pytest.approx(0.3)

    assemb.confidence = 0.9
    assert np.allclose(assemb.data[:, 2], [0.9, 0.9, 0.9], equal_nan=True)


# ---------------------------
# soft_identity
# ---------------------------


def test_soft_identity_simple():
    # data format: x, y, conf, group
    assemb = Assembly(size=3)
    assemb.data[:] = np.nan
    assemb.data[0] = [0, 0, 1.0, 0]
    assemb.data[1] = [5, 5, 0.5, 0]
    assemb.data[2] = [10, 10, 1.0, 1]
    assemb._visible = {0, 1, 2}

    # groups: 0 → weights 1.0 and 0.5 (avg=0.75)
    #          1 → weight 1.0
    # softmax([0.75, 1.0]) ≈ [...]
    soft = assemb.soft_identity
    assert set(soft.keys()) == {0, 1}
    s0, s1 = soft[0], soft[1]
    assert pytest.approx(s0 + s1) == 1.0
    assert s1 > s0


# ---------------------------
# intersection operator: __contains__
# ---------------------------


def test_contains_checks_shared_idx():
    ass1 = Assembly(size=3)
    ass2 = Assembly(size=3)

    j0 = Joint((0, 0), confidence=1.0, label=0, idx=10)
    j1 = Joint((1, 1), confidence=1.0, label=1, idx=99)

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


def test_assembly_addition_combines_links():
    a1 = Assembly(size=4)
    a2 = Assembly(size=4)

    j0 = Joint((0, 0), 1.0, label=0, idx=10)
    j1 = Joint((1, 0), 1.0, label=1, idx=11)
    j2 = Joint((2, 0), 1.0, label=2, idx=12)
    j3 = Joint((3, 0), 1.0, label=3, idx=13)

    l01 = Link(j0, j1, affinity=0.5)
    l23 = Link(j2, j3, affinity=0.8)

    a1.add_link(l01)
    a2.add_link(l23)

    # now they share NO joints → addition should succeed
    result = a1 + a2

    assert result.n_links == 2
    assert result._affinity == pytest.approx(1.3)

    # original assemblies unchanged
    assert a1.n_links == 1
    assert a2.n_links == 1

    # now purposely make them share a joint → should raise
    a2.add_joint(j0)
    with pytest.raises(ArithmeticError):
        _ = a1 + a2
