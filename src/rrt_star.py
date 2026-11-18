from math import log, pi, sqrt

import numpy as np

from .rrt import RRT


class RRTStar(RRT):
    name = "RRT*"

    def gamma(self):
        # Workspace area gives a conservative default for the 2D radius.
        return self.config.neighbor_gamma or 2.2 * sqrt(1.5 * self.environment.area / pi)

    def neighbor_radius(self, node_count=None):
        if node_count is None:
            node_count = len(self.tree.nodes) + 1
        return min(self.config.step_size, self.gamma() * sqrt(log(node_count) / node_count))

    def _choose_parent(self, point, nearest, neighbors, distances):
        # The nearest node remains a valid fallback outside the shrinking radius.
        parent = nearest
        best_cost = self.tree.nodes[parent].cost + float(distances[parent])
        for index in neighbors:
            node = self.tree.nodes[index]
            cost = node.cost + float(distances[index])
            if cost < best_cost and self._edge_free(node.position, point):
                parent = index
                best_cost = cost
        return parent

    def _rewire(self, new_index, neighbors, distances):
        new_node = self.tree.nodes[new_index]
        rewired = []
        for index in neighbors:
            if index == 0 or index == new_node.parent:
                continue
            node = self.tree.nodes[index]
            cost = new_node.cost + float(distances[index])
            if cost >= node.cost:
                continue
            if self.tree.is_ancestor(index, new_index):
                continue
            if self._edge_free(new_node.position, node.position):
                self.tree.reparent(index, new_index)
                rewired.append(index)
        self.rewire_count += len(rewired)
        return rewired

    def _insert(self, point, nearest):
        distances = np.linalg.norm(self.tree.positions - point, axis=1)
        neighbors = np.flatnonzero(distances <= self.neighbor_radius()).tolist()
        parent = self._choose_parent(point, nearest, neighbors, distances)
        new_index = self.tree.add(point, parent)
        rewired = self._rewire(new_index, neighbors, distances)
        return new_index, rewired

