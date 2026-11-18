from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    position: np.ndarray
    parent: int | None
    cost: float = 0.0
    children: set[int] = field(default_factory=set)


class Tree:
    def __init__(self, start, capacity):
        self.nodes = [Node(np.asarray(start, dtype=float).copy(), None)]
        self._positions = np.empty((max(capacity, 1), 2), dtype=float)
        self._positions[0] = start

    @property
    def positions(self):
        return self._positions[:len(self.nodes)]

    def add(self, position, parent):
        if not 0 <= parent < len(self.nodes):
            raise ValueError("Unknown parent")
        index = len(self.nodes)
        if index == len(self._positions):
            self._positions = np.concatenate((self._positions, np.empty_like(self._positions)))
        position = np.asarray(position, dtype=float).copy()
        parent_node = self.nodes[parent]
        cost = parent_node.cost + float(np.linalg.norm(position - parent_node.position))
        self.nodes.append(Node(position, parent, cost))
        parent_node.children.add(index)
        self._positions[index] = position
        return index

    def is_ancestor(self, candidate, node):
        while node is not None:
            if node == candidate:
                return True
            node = self.nodes[node].parent
        return False

    def reparent(self, node, parent):
        if not 0 <= node < len(self.nodes) or not 0 <= parent < len(self.nodes):
            raise ValueError("Unknown node")
        if node == 0 or self.is_ancestor(node, parent):
            raise ValueError("Reparenting would change the root or create a cycle")

        old_parent = self.nodes[node].parent
        self.nodes[old_parent].children.remove(node)
        self.nodes[parent].children.add(node)
        self.nodes[node].parent = parent

        # Every descendant depends on the new cost, not just the rewired node.
        pending = [node]
        updated = 0
        while pending:
            current = self.nodes[pending.pop()]
            parent_node = self.nodes[current.parent]
            current.cost = parent_node.cost + float(np.linalg.norm(current.position - parent_node.position))
            pending.extend(sorted(current.children, reverse=True))
            updated += 1
        return updated

    def path_to(self, node):
        points = []
        while node is not None:
            points.append(self.nodes[node].position)
            node = self.nodes[node].parent
        return np.asarray(points[::-1])
