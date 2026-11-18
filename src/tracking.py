"""Bám đường bằng Pure Pursuit cho robot 2 bánh vi sai (mô hình unicycle).

Mô hình:      x' = v cos(theta),  y' = v sin(theta),  theta' = omega
Pure Pursuit: chọn điểm mục tiêu trên đường cách điểm gần nhất một đoạn L (look-ahead),
              alpha = góc giữa hướng robot và hướng tới điểm mục tiêu,
              độ cong kappa = 2 sin(alpha) / d  (d = khoảng cách tới điểm mục tiêu),
              omega = v * kappa.
Giới hạn:     |v| <= v_max, |omega| <= omega_max (v giảm khi cua gấp: v <= omega_max / |kappa|),
              v giảm tuyến tính khi tới gần đích. Nếu |alpha| > pi/2, robot quay tại chỗ.
Đo đạc:       sai số bám đường (cross-track error) so với đường tham chiếu, thời gian hoàn thành,
              và kiểm tra va chạm liên tục của quỹ đạo thực với đúng bán kính robot.
"""

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin

import numpy as np


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead: float = 0.6        # L (m)
    v_max: float = 0.5            # m/s
    omega_max: float = 1.5        # rad/s
    goal_tolerance: float = 0.1   # m
    goal_slowdown: float = 1.0    # s^-1: v <= goal_slowdown * khoảng cách tới đích
    dt: float = 0.05              # s
    max_time: float = 200.0       # s
    robot_radius: float = 0.2     # bán kính thật dùng để kiểm tra va chạm

    def __post_init__(self):
        for name in ("lookahead", "v_max", "omega_max", "goal_tolerance", "goal_slowdown", "dt", "max_time"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.robot_radius) or self.robot_radius < 0:
            raise ValueError("robot_radius must be finite and non-negative")


@dataclass
class TrackResult:
    reached: bool
    collided: bool
    success: bool                 # tới đích trong thời gian cho phép và không va chạm
    time: float                   # thời gian mô phỏng (s)
    traveled_length: float        # quãng đường thực (m)
    rms_cte: float                # căn bậc hai trung bình bình phương sai số bám đường (m)
    max_cte: float                # sai số bám đường lớn nhất (m)
    mean_cte: float
    trajectory: np.ndarray        # N x 3: x, y, theta
    reference_length: float


def _arc_lengths(path):
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segments)))


def _point_at(path, arc, s):
    s = min(max(s, 0.0), arc[-1])
    k = int(np.searchsorted(arc, s, side="right")) - 1
    k = min(max(k, 0), len(path) - 2)
    span = arc[k + 1] - arc[k]
    t = 0.0 if span <= 0 else (s - arc[k]) / span
    return path[k] + t * (path[k + 1] - path[k])


def _closest_arc(path, arc, position, s_low, s_high):
    """Hình chiếu của vị trí lên đường, chỉ xét các đoạn trong cửa sổ [s_low, s_high] (đơn điệu theo s)."""
    best_s, best_d = s_low, float("inf")
    for k in range(len(path) - 1):
        if arc[k + 1] < s_low or arc[k] > s_high:
            continue
        a, b = path[k], path[k + 1]
        ab = b - a
        length_sq = float(ab @ ab)
        t = 0.0 if length_sq == 0 else float(np.clip((position - a) @ ab / length_sq, 0.0, 1.0))
        s = arc[k] + t * float(np.sqrt(length_sq))
        s = min(max(s, s_low), s_high)
        d = float(np.linalg.norm(position - _point_at(path, arc, s)))
        if d < best_d:
            best_s, best_d = s, d
    return best_s


def cross_track_errors(points, path):
    """Khoảng cách từ mỗi điểm tới đường gấp khúc `path` (vector hóa)."""
    points = np.asarray(points, dtype=float)
    a, b = path[:-1], path[1:]
    ab = b - a
    length_sq = np.einsum("ij,ij->i", ab, ab)
    length_sq = np.where(length_sq == 0, 1.0, length_sq)
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nki,ki->nk", rel, ab) / length_sq[None, :], 0.0, 1.0)
    nearest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - nearest, axis=2).min(axis=1)


def track_path(path, environment, config=None, initial_heading=None):
    """Mô phỏng Pure Pursuit bám `path`; trả về TrackResult. Mặc định robot hướng theo đoạn đầu."""
    config = config or PurePursuitConfig()
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must be an array of at least two 2D points")
    arc = _arc_lengths(path)
    total = float(arc[-1])
    goal = path[-1]

    first = path[1] - path[0]
    theta = atan2(first[1], first[0]) if initial_heading is None else float(initial_heading)
    position = path[0].copy()
    states = [(position[0], position[1], theta)]
    s_current = 0.0
    traveled = 0.0
    collided = reached = False
    steps = int(config.max_time / config.dt)

    for step in range(steps):
        distance_goal = float(np.linalg.norm(position - goal))
        if distance_goal <= config.goal_tolerance:
            reached = True
            break

        window = 2 * config.lookahead + 1.0
        s_current = _closest_arc(path, arc, position, s_current, min(total, s_current + window))
        target = _point_at(path, arc, s_current + config.lookahead)
        dx, dy = target - position
        d_target = max(hypot(dx, dy), 1e-6)
        alpha = atan2(dy, dx) - theta
        alpha = (alpha + pi) % (2 * pi) - pi

        if abs(alpha) > pi / 2:
            v, omega = 0.0, config.omega_max * (1 if alpha > 0 else -1)
        else:
            kappa = 2 * sin(alpha) / d_target
            v = min(config.v_max, config.goal_slowdown * distance_goal)
            if abs(kappa) > 1e-9:
                v = min(v, config.omega_max / abs(kappa))
            omega = max(-config.omega_max, min(config.omega_max, v * kappa))

        theta_mid = theta + 0.5 * omega * config.dt
        new_position = position + np.array([cos(theta_mid), sin(theta_mid)]) * v * config.dt
        theta += omega * config.dt

        if not environment.edge_free(position, new_position, config.robot_radius):
            collided = True
            position = new_position
            states.append((position[0], position[1], theta))
            break
        traveled += float(np.linalg.norm(new_position - position))
        position = new_position
        states.append((position[0], position[1], theta))

    trajectory = np.asarray(states)
    errors = cross_track_errors(trajectory[:, :2], path)
    return TrackResult(
        reached=reached,
        collided=collided,
        success=reached and not collided,
        time=(len(states) - 1) * config.dt,
        traveled_length=traveled,
        rms_cte=float(np.sqrt(np.mean(errors ** 2))),
        max_cte=float(errors.max()),
        mean_cte=float(errors.mean()),
        trajectory=trajectory,
        reference_length=total,
    )
