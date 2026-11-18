from dataclasses import dataclass
from math import isfinite

EPS = 1e-10


@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        if not all(isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("Rectangle coordinates must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rectangle dimensions must be positive")


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float

    def __post_init__(self):
        if not all(isfinite(value) for value in (self.x, self.y, self.radius)) or self.radius <= 0:
            raise ValueError("Circle must have finite coordinates and positive radius")


def point_segment_distance_sq(point, start, end):
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    px, py = float(point[0] - start[0]), float(point[1] - start[1])
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return px * px + py * py
    projection = (px * dx + py * dy) / length_sq
    projection = max(0.0, min(1.0, projection))
    return (px - projection * dx) ** 2 + (py - projection * dy) ** 2


def segment_intersects_box(start, end, xmin, ymin, xmax, ymax):
    enter, leave = 0.0, 1.0
    for axis, lower, upper in ((0, xmin, xmax), (1, ymin, ymax)):
        origin = float(start[axis])
        direction = float(end[axis] - start[axis])
        if direction == 0:
            if origin < lower - EPS or origin > upper + EPS:
                return False
            continue
        near = (lower - EPS - origin) / direction
        far = (upper + EPS - origin) / direction
        if near > far:
            near, far = far, near
        enter = max(enter, near)
        leave = min(leave, far)
        if enter > leave:
            return False
    return True


def segment_hits_obstacle(start, end, obstacle, robot_radius=0.0):
    # Touching an obstacle counts as a collision.
    if isinstance(obstacle, Circle):
        radius = obstacle.radius + robot_radius + EPS
        center = (obstacle.x, obstacle.y)
        return point_segment_distance_sq(center, start, end) <= radius * radius
    if not isinstance(obstacle, Rectangle):
        raise TypeError(f"Unsupported obstacle: {type(obstacle)}")

    x0, y0 = obstacle.x, obstacle.y
    x1, y1 = x0 + obstacle.width, y0 + obstacle.height
    radius = robot_radius
    if max(start[0], end[0]) < x0 - radius - EPS or min(start[0], end[0]) > x1 + radius + EPS:
        return False
    if max(start[1], end[1]) < y0 - radius - EPS or min(start[1], end[1]) > y1 + radius + EPS:
        return False

    # Expanding a rectangle by a disk gives two strips and four rounded corners.
    if segment_intersects_box(start, end, x0 - radius, y0, x1 + radius, y1):
        return True
    if segment_intersects_box(start, end, x0, y0 - radius, x1, y1 + radius):
        return True
    corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    return any(
        point_segment_distance_sq(corner, start, end) <= (radius + EPS) ** 2
        for corner in corners
    )

