import numpy as np
import pytest

from src.node import Tree


def build_example():
    """Cây ở ví dụ mục 3.7: S→A→B→C→D→E, rồi thêm U nối trực tiếp từ S."""
    tree = Tree((0, 0), 8)
    a = tree.add((0, 4), 0)
    b = tree.add((4, 4), a)
    c = tree.add((4, 1), b)
    d = tree.add((5, 1), c)
    e = tree.add((6, 1), d)
    u = tree.add((3, 0), 0)
    return tree, dict(a=a, b=b, c=c, d=d, e=e, u=u)


def test_costs_before_reparent():
    tree, ids = build_example()
    assert tree.nodes[ids["c"]].cost == pytest.approx(11)
    assert tree.nodes[ids["e"]].cost == pytest.approx(13)
    assert tree.nodes[ids["u"]].cost == pytest.approx(3)


def test_reparent_updates_all_descendants():
    tree, ids = build_example()
    updated = tree.reparent(ids["c"], ids["u"])
    assert updated == 3  # C, D, E
    assert tree.nodes[ids["c"]].cost == pytest.approx(3 + np.sqrt(2))
    assert tree.nodes[ids["d"]].cost == pytest.approx(3 + np.sqrt(2) + 1)
    assert tree.nodes[ids["e"]].cost == pytest.approx(3 + np.sqrt(2) + 2)
    assert ids["c"] not in tree.nodes[ids["b"]].children
    assert ids["c"] in tree.nodes[ids["u"]].children


def test_reparent_rejects_cycle_and_root():
    tree, ids = build_example()
    with pytest.raises(ValueError):
        tree.reparent(ids["b"], ids["e"])  # E là hậu duệ của B → tạo chu trình
    with pytest.raises(ValueError):
        tree.reparent(0, ids["u"])  # không được đổi cha của gốc


def test_add_rejects_unknown_parent():
    tree = Tree((0, 0), 2)
    with pytest.raises(ValueError):
        tree.add((1, 1), 5)


def test_path_to_returns_root_first():
    tree, ids = build_example()
    path = tree.path_to(ids["e"])
    assert np.array_equal(path[0], (0, 0)) and np.array_equal(path[-1], (6, 1))
