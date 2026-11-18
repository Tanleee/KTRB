import numpy as np
import pytest

from src import PlannerConfig, RRT, RRTStar, load_environment
from src.utils import validate_path

from .helpers import check_tree_invariants

MAPS = ["simple", "complex", "narrow"]
ITER = 700


def make(algorithm, map_name, seed, **kw):
    env = load_environment(map_name)
    return algorithm(env, PlannerConfig(max_iterations=ITER, **kw), seed)


@pytest.mark.parametrize("map_name", MAPS)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("algorithm", [RRT, RRTStar])
def test_tree_invariants_hold(algorithm, map_name, seed):
    planner = make(algorithm, map_name, seed)
    planner.run()
    check_tree_invariants(planner)


@pytest.mark.parametrize("map_name", MAPS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_node_positions_for_rrt_and_rrt_star(map_name, seed):
    """Mệnh đề mục 7.1: cùng seed → cùng tập vị trí nút; RRT* chỉ khác cạnh/chi phí."""
    rrt, star = make(RRT, map_name, seed), make(RRTStar, map_name, seed)
    rrt.run(), star.run()
    assert np.array_equal(rrt.tree.positions, star.tree.positions)


@pytest.mark.parametrize("map_name", MAPS)
def test_deterministic_for_same_seed(map_name):
    a, b = make(RRTStar, map_name, 4), make(RRTStar, map_name, 4)
    ra, rb = a.run(), b.run()
    assert ra.path_length == rb.path_length
    assert ra.rewires == rb.rewires
    assert ra.first_solution_iteration == rb.first_solution_iteration
    assert np.array_equal(a.tree.positions, b.tree.positions)


@pytest.mark.parametrize("map_name", ["simple", "complex"])
@pytest.mark.parametrize("algorithm", [RRT, RRTStar])
def test_returned_path_is_valid(algorithm, map_name):
    planner = make(algorithm, map_name, 1)
    result = planner.run()
    if result.success:
        assert validate_path(result.path, planner.environment, planner.config.robot_radius)
        assert np.allclose(result.path[0], planner.environment.start)
        assert np.array_equal(result.path[-1], planner.environment.goal)


@pytest.mark.parametrize("algorithm", [RRT, RRTStar])
def test_best_cost_never_increases_per_run(algorithm):
    planner = make(algorithm, "simple", 3)
    planner.run()
    costs = [r.best_cost for r in planner.history if r.best_cost is not None]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))


def test_rrt_never_rewires_and_rrt_star_does():
    assert make(RRT, "simple", 0).run().rewires == 0
    assert make(RRTStar, "simple", 0).run().rewires > 0


def test_rrt_star_not_longer_than_rrt_on_simple():
    rrt, star = make(RRT, "simple", 0).run(), make(RRTStar, "simple", 0).run()
    assert rrt.success and star.success
    assert star.path_length <= rrt.path_length + 1e-9


def test_stop_on_first_stops_early():
    planner = make(RRT, "simple", 0, stop_on_first=True)
    result = planner.run()
    assert result.success and result.iterations == result.first_solution_iteration


def test_neighbor_radius_equals_eta_at_2500_nodes_with_default_gamma():
    """Ghi nhận trong báo cáo: với γ mặc định, bán kính chưa giảm dưới η trong 2500 vòng."""
    planner = make(RRTStar, "simple", 0)
    assert planner.gamma() == pytest.approx(18.24, abs=0.01)
    assert planner.neighbor_radius(2501) == pytest.approx(planner.config.step_size)
    assert planner.neighbor_radius(5000) < planner.config.step_size


@pytest.mark.parametrize("bad", [
    {"step_size": 0}, {"step_size": float("nan")}, {"goal_sample_rate": 1.0},
    {"goal_sample_rate": -0.1}, {"max_iterations": 0}, {"robot_radius": -1},
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        PlannerConfig(**bad)


def test_invalid_start_rejected():
    env = load_environment("simple")
    with pytest.raises(ValueError):
        RRT(env, PlannerConfig(robot_radius=5.0), 0)  # Start không còn nằm trong miền tự do


@pytest.mark.parametrize("algorithm", [RRT, RRTStar])
def test_narrow_gap_closed_when_radius_is_0_3(algorithm):
    """Mục 6: r = 0,3 làm khe 0,6 m đóng lại (cấm tiếp xúc) → không thể có đường."""
    planner = make(algorithm, "narrow", 1, robot_radius=0.3)
    assert not planner.run().success
