import numpy as np
import pytest

from src import load_environment
from src.collision import Circle, Rectangle, segment_hits_obstacle

RADIUS = 0.2


def clearance(point, obstacle):
    """Khoảng cách từ tâm robot đến vật cản trừ bán kính vật cản (âm nếu nằm trong)."""
    if isinstance(obstacle, Circle):
        return float(np.hypot(point[0] - obstacle.x, point[1] - obstacle.y)) - obstacle.radius
    dx = max(obstacle.x - point[0], 0, point[0] - (obstacle.x + obstacle.width))
    dy = max(obstacle.y - point[1], 0, point[1] - (obstacle.y + obstacle.height))
    return float(np.hypot(dx, dy))


@pytest.mark.parametrize("map_name", ["simple", "complex", "narrow"])
def test_edge_check_matches_dense_sampling(map_name):
    """So kiểm tra giải tích với cách đo thô (lấy mẫu dày dọc cạnh)."""
    env = load_environment(map_name)
    rng = np.random.default_rng(123)
    ts = np.linspace(0, 1, 600)
    for _ in range(1500):
        a = rng.uniform(0.3, 11.7, 2)
        b = a + rng.uniform(-0.8, 0.8, 2)
        for obstacle in env.obstacles:
            exact = segment_hits_obstacle(a, b, obstacle, RADIUS)
            samples = a[None, :] + ts[:, None] * (b - a)[None, :]
            min_clear = min(clearance(p, obstacle) for p in samples)
            if min_clear <= RADIUS - 1e-9:
                assert exact, "lấy mẫu thấy va chạm nhưng kiểm tra giải tích bỏ sót"
            if exact:
                assert min_clear <= RADIUS + 0.01, "kiểm tra giải tích loại thừa cạnh hợp lệ"


def test_touching_counts_as_collision():
    circle = Circle(5, 5, 1)
    # đường đi song song, cách tâm đúng R + r → chạm → bị loại
    assert segment_hits_obstacle((3, 5 + 1 + RADIUS), (7, 5 + 1 + RADIUS), circle, RADIUS)
    assert not segment_hits_obstacle((3, 5 + 1 + RADIUS + 1e-6), (7, 5 + 1 + RADIUS + 1e-6), circle, RADIUS)


def test_rectangle_rounded_corner_not_over_rejected():
    rect = Rectangle(0, 0, 2, 2)
    # điểm cách góc (2,2) khoảng 0.25 theo đường chéo > r = 0.2 → hợp lệ
    p = (2 + 0.25 / np.sqrt(2), 2 + 0.25 / np.sqrt(2))
    assert not segment_hits_obstacle(p, p, rect, RADIUS)
    # một hình chữ nhật mở rộng thô sẽ loại điểm này; mô hình góc tròn thì không


def test_degenerate_edge_zero_length():
    circle = Circle(0, 0, 1)
    assert segment_hits_obstacle((0.5, 0), (0.5, 0), circle, RADIUS)
    assert not segment_hits_obstacle((3, 3), (3, 3), circle, RADIUS)


def test_boundary_uses_shrunk_workspace():
    env = load_environment("simple")
    assert env.contains((1, 1), RADIUS)
    assert not env.contains((0.2, 5), RADIUS)  # tâm cách tường đúng r → chạm
    assert not env.edge_free((1, 1), (12.5, 1), RADIUS)
