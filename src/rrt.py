from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from .collision import EPS
from .node import Tree
from .utils import path_length


@dataclass(frozen=True)
class PlannerConfig:
    step_size: float = 0.8
    goal_sample_rate: float = 0.05
    max_iterations: int = 2500
    robot_radius: float = 0.2
    goal_radius: float | None = None
    neighbor_gamma: float | None = None
    stop_on_first: bool = False

    def __post_init__(self):
        for name in ("step_size", "goal_radius", "neighbor_gamma"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.goal_sample_rate) or not 0 <= self.goal_sample_rate < 1:
            raise ValueError("goal_sample_rate must be in [0, 1)")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not isfinite(self.robot_radius) or self.robot_radius < 0:
            raise ValueError("robot_radius must be finite and non-negative")


@dataclass(frozen=True)
class StepRecord:
    iteration: int
    nodes: int
    elapsed: float
    best_cost: float | None
    added_node: int | None
    rewired: tuple[int, ...]


@dataclass
class PlanResult:
    algorithm: str
    seed: int
    success: bool
    path: np.ndarray | None
    path_length: float | None
    planning_time: float
    time_to_first_solution: float | None
    first_solution_iteration: int | None
    iterations: int
    number_of_nodes: int
    rewires: int


class RRT:
    name = "RRT"

    def __init__(self, environment, config=None, seed=0):
        self.environment = environment
        self.config = config or PlannerConfig()
        environment.validate_query(self.config.robot_radius)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tree = Tree(environment.start, self.config.max_iterations + 1)
        self.goal = np.asarray(environment.goal, dtype=float)
        self.iteration = 0
        self.elapsed = 0.0
        self.first_solution_time = None
        self.first_solution_iteration = None
        self.goal_candidates = {}
        self.best_goal_parent = None
        self.best_cost = None
        self.rewire_count = 0

        started = perf_counter()
        self._consider_goal(0)
        self._refresh_goal()
        self.elapsed = perf_counter() - started
        if self.best_cost is not None:
            self.first_solution_time = self.elapsed
            self.first_solution_iteration = 0
        self.history = [StepRecord(
            iteration=0,
            nodes=1,
            elapsed=self.elapsed,
            best_cost=self.best_cost,
            added_node=None,
            rewired=(),
        )]

    @property
    def done(self):
        reached_budget = self.iteration >= self.config.max_iterations
        found_first = self.config.stop_on_first and self.best_cost is not None
        return reached_budget or found_first

    def sample(self):
        # Both planners consume three numbers each step to share the same samples.
        goal_draw, x_draw, y_draw = self.rng.random(3)
        if goal_draw < self.config.goal_sample_rate:
            return self.goal.copy()
        xmin, xmax, ymin, ymax = self.environment.bounds
        radius = self.config.robot_radius
        return np.array([
            xmin + radius + x_draw * (xmax - xmin - 2 * radius),
            ymin + radius + y_draw * (ymax - ymin - 2 * radius),
        ])

    def nearest(self, point):
        offsets = self.tree.positions - point
        distances_squared = np.einsum("ij,ij->i", offsets, offsets)
        return int(np.argmin(distances_squared))

    def steer(self, origin, target):
        direction = target - origin
        distance = float(np.linalg.norm(direction))
        if distance <= EPS:
            return origin.copy()
        return origin + direction * min(1.0, self.config.step_size / distance)

    def _edge_free(self, start, end):
        return self.environment.edge_free(start, end, self.config.robot_radius)

    def _insert(self, point, nearest):
        return self.tree.add(point, nearest), ()

    def _consider_goal(self, node):
        point = self.tree.nodes[node].position
        distance = float(np.linalg.norm(point - self.goal))
        radius = self.config.goal_radius or self.config.step_size
        if distance <= radius and self._edge_free(point, self.goal):
            self.goal_candidates[node] = distance

    def _refresh_goal(self):
        # Rewiring an ancestor can lower the cost of any goal candidate.
        if not self.goal_candidates:
            return
        best = min(
            self.goal_candidates,
            key=lambda index: (
                self.tree.nodes[index].cost + self.goal_candidates[index], index
            ),
        )
        self.best_goal_parent = best
        self.best_cost = self.tree.nodes[best].cost + self.goal_candidates[best]

    def step(self):
        if self.done:
            raise StopIteration("Planning finished")

        started = perf_counter()
        self.iteration += 1
        sample = self.sample()
        nearest = self.nearest(sample)
        origin = self.tree.nodes[nearest].position
        point = self.steer(origin, sample)
        added_node = None
        rewired = ()

        if float(np.linalg.norm(point - origin)) > EPS and self._edge_free(origin, point):
            added_node, rewired = self._insert(point, nearest)
            self._consider_goal(added_node)
            self._refresh_goal()

        # Measure planning work without rendering or saving files.
        self.elapsed += perf_counter() - started
        if self.best_cost is not None and self.first_solution_time is None:
            self.first_solution_time = self.elapsed
            self.first_solution_iteration = self.iteration

        record = StepRecord(
            iteration=self.iteration,
            nodes=len(self.tree.nodes),
            elapsed=self.elapsed,
            best_cost=self.best_cost,
            added_node=added_node,
            rewired=tuple(rewired),
        )
        self.history.append(record)
        return record

    def run(self):
        while not self.done:
            self.step()
        return self.result()

    def path(self):
        if self.best_goal_parent is None:
            return None
        points = self.tree.path_to(self.best_goal_parent)
        if not np.array_equal(points[-1], self.goal):
            points = np.vstack((points, self.goal))
        return points

    def result(self):
        path = self.path()
        return PlanResult(
            algorithm=self.name,
            seed=self.seed,
            success=path is not None,
            path=path,
            path_length=path_length(path),
            planning_time=self.elapsed,
            time_to_first_solution=self.first_solution_time,
            first_solution_iteration=self.first_solution_iteration,
            iterations=self.iteration,
            number_of_nodes=len(self.tree.nodes),
            rewires=self.rewire_count,
        )

