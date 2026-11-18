from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path

from .collision import EPS, Circle, Rectangle, segment_hits_obstacle

MAP_DIR = Path(__file__).resolve().parents[1] / "configs" / "maps"


@dataclass(frozen=True)
class Environment:
    name: str
    bounds: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacles: tuple[Rectangle | Circle, ...] = ()

    def __post_init__(self):
        if len(self.bounds) != 4 or not all(isfinite(value) for value in self.bounds):
            raise ValueError("Bounds must be four finite numbers")
        xmin, xmax, ymin, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Bounds must have positive width and height")
        for label, point in (("Start", self.start), ("Goal", self.goal)):
            if len(point) != 2 or not all(isfinite(value) for value in point):
                raise ValueError(f"{label} must contain two finite coordinates")
        if any(not isinstance(obstacle, (Rectangle, Circle)) for obstacle in self.obstacles):
            raise TypeError("Only rectangles and circles are supported")

    @property
    def area(self):
        xmin, xmax, ymin, ymax = self.bounds
        return (xmax - xmin) * (ymax - ymin)

    def contains(self, point, robot_radius=0.0):
        if len(point) != 2 or not all(isfinite(value) for value in point):
            return False
        xmin, xmax, ymin, ymax = self.bounds
        return (
            xmin + robot_radius + EPS < point[0] < xmax - robot_radius - EPS
            and ymin + robot_radius + EPS < point[1] < ymax - robot_radius - EPS
        )

    def edge_free(self, start, end, robot_radius=0.0):
        if not self.contains(start, robot_radius) or not self.contains(end, robot_radius):
            return False
        # A segment between two points inside the rectangular boundary stays inside.
        return not any(
            segment_hits_obstacle(start, end, obstacle, robot_radius)
            for obstacle in self.obstacles
        )

    def point_free(self, point, robot_radius=0.0):
        return self.edge_free(point, point, robot_radius)

    def validate_query(self, robot_radius):
        if not isfinite(robot_radius) or robot_radius < 0:
            raise ValueError("Robot radius must be finite and non-negative")
        for label, point in (("Start", self.start), ("Goal", self.goal)):
            if not self.point_free(point, robot_radius):
                raise ValueError(f"{label} is invalid for robot radius {robot_radius}")

    def to_dict(self):
        return {
            "name": self.name,
            "bounds": list(self.bounds),
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [
                {
                    "type": "rectangle" if isinstance(obstacle, Rectangle) else "circle",
                    **asdict(obstacle),
                }
                for obstacle in self.obstacles
            ],
        }


def load_environment(name_or_path="simple"):
    path = Path(name_or_path)
    if not path.is_file():
        path = MAP_DIR / f"{name_or_path}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    obstacles = []
    for entry in data.get("obstacles", []):
        values = dict(entry)
        kind = values.pop("type")
        if kind == "rectangle":
            obstacles.append(Rectangle(**values))
        elif kind == "circle":
            obstacles.append(Circle(**values))
        else:
            raise ValueError(f"Unknown obstacle type: {kind}")
    return Environment(
        name=data["name"],
        bounds=tuple(data["bounds"]),
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
        obstacles=tuple(obstacles),
    )

