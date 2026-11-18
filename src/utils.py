import numpy as np


def path_length(path):
    if path is None:
        return None
    segments = np.diff(np.asarray(path), axis=0)
    return float(np.linalg.norm(segments, axis=1).sum())


def validate_path(path, environment, robot_radius):
    if path is None or len(path) == 0:
        return False
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return False
    if not np.allclose(points[0], environment.start, atol=1e-9, rtol=0):
        return False
    if not np.allclose(points[-1], environment.goal, atol=1e-9, rtol=0):
        return False
    if not all(environment.point_free(point, robot_radius) for point in points):
        return False
    return all(
        environment.edge_free(start, end, robot_radius)
        for start, end in zip(points[:-1], points[1:])
    )

