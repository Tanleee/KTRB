import numpy as np
import pytest

from dataclasses import replace

from src import RRTStar, PlannerConfig, load_environment
from src.smoothing import shortcut_path
from src.tracking import PurePursuitConfig, cross_track_errors, track_path
from src.utils import path_length, validate_path


def free_env():
    return replace(load_environment("simple"), obstacles=())


def test_straight_path_is_tracked_exactly():
    env = free_env()
    path = np.array([[1.0, 1.0], [11.0, 1.0]])
    result = track_path(path, env, PurePursuitConfig(lookahead=0.6))
    assert result.success
    assert result.max_cte < 1e-6
    # Thời gian ~ chiều dài / v_max (cộng đoạn giảm tốc gần đích)
    assert 10 / 0.5 <= result.time < 10 / 0.5 + 8


def test_speed_and_turn_rate_limits_respected():
    env = free_env()
    path = np.array([[1.0, 1.0], [6.0, 1.0], [6.0, 6.0]])
    config = PurePursuitConfig(lookahead=0.6, v_max=0.5, omega_max=1.0)
    result = track_path(path, env, config)
    xy, theta = result.trajectory[:, :2], result.trajectory[:, 2]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / config.dt
    omega = np.abs(np.diff(theta)) / config.dt
    assert speed.max() <= config.v_max + 1e-9
    assert omega.max() <= config.omega_max + 1e-9
    assert result.reached


def test_collision_detected_when_path_hugs_obstacle():
    env = load_environment("simple")
    # Đường đi qua tâm vật cản tròn => chắc chắn va chạm.
    obstacle = env.obstacles[1]
    path = np.array([[obstacle.x - 3, obstacle.y], [obstacle.x + 3, obstacle.y]])
    result = track_path(path, env, PurePursuitConfig(robot_radius=0.2))
    assert result.collided and not result.success


def test_timeout_when_max_time_too_short():
    env = free_env()
    path = np.array([[1.0, 1.0], [11.0, 1.0]])
    result = track_path(path, env, PurePursuitConfig(max_time=2.0))
    assert not result.reached and not result.collided and not result.success


def test_cross_track_errors_known_values():
    path = np.array([[0.0, 0.0], [10.0, 0.0]])
    errors = cross_track_errors(np.array([[5.0, 1.0], [-2.0, 0.0], [10.0, -3.0]]), path)
    assert np.allclose(errors, [1.0, 2.0, 3.0])


def test_tracking_is_deterministic():
    env = free_env()
    path = np.array([[1.0, 1.0], [6.0, 3.0], [11.0, 11.0]])
    a, b = track_path(path, env), track_path(path, env)
    assert np.array_equal(a.trajectory, b.trajectory)


@pytest.mark.parametrize("map_name", ["simple", "complex"])
def test_shortcut_is_valid_not_longer_and_keeps_endpoints(map_name):
    env = load_environment(map_name)
    result = RRTStar(env, PlannerConfig(max_iterations=700), 1).run()
    assert result.success
    smooth = shortcut_path(result.path, env, 0.2)
    assert validate_path(smooth, env, 0.2)
    assert path_length(smooth) <= path_length(result.path) + 1e-9
    assert len(smooth) <= len(result.path)


def test_shortcut_on_two_points_is_identity():
    env = free_env()
    path = np.array([[1.0, 1.0], [11.0, 11.0]])
    assert np.array_equal(shortcut_path(path, env, 0.2), path)


def test_safety_margin_avoids_corner_cutting_collision():
    # Ghi nhận trong báo cáo: đường sát biên vật cản nở (margin = 0) có thể gây va chạm khi bám đường.
    env = load_environment("complex")
    tracking = PurePursuitConfig(lookahead=0.6, robot_radius=0.2)
    tight = RRTStar(env, PlannerConfig(max_iterations=2500, robot_radius=0.2), 0).run()
    safe = RRTStar(env, PlannerConfig(max_iterations=2500, robot_radius=0.25), 0).run()
    assert track_path(tight.path, env, tracking).collided
    assert track_path(safe.path, env, tracking).success


@pytest.mark.parametrize("bad", [
    {"lookahead": 0}, {"v_max": -1}, {"omega_max": float("nan")}, {"dt": 0}, {"robot_radius": -0.1},
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        PurePursuitConfig(**bad)

