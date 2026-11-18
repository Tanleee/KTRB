import numpy as np


def check_tree_invariants(planner, tol=1e-9):
    """Kiểm tra các bất biến nêu ở mục 3.7 của báo cáo + mọi cạnh không va chạm."""
    tree, env, radius = planner.tree, planner.environment, planner.config.robot_radius
    nodes = tree.nodes
    assert nodes[0].parent is None and nodes[0].cost == 0.0, "gốc phải không có cha và chi phí 0"
    for index, node in enumerate(nodes):
        assert np.array_equal(tree.positions[index], node.position)
        if index == 0:
            continue
        # đúng một cha, cha tồn tại, và quan hệ con khớp hai chiều
        assert node.parent is not None and 0 <= node.parent < len(nodes)
        assert index in nodes[node.parent].children
        # chi phí nhất quán: cost(v) = cost(p) + |p v|
        parent = nodes[node.parent]
        expected = parent.cost + float(np.linalg.norm(node.position - parent.position))
        assert abs(node.cost - expected) <= tol * max(1.0, expected)
        # mọi cạnh hợp lệ với bán kính robot
        assert env.edge_free(parent.position, node.position, radius)
        # không có chu trình: đi ngược lên gốc trong tối đa len(nodes) bước
        steps, current = 0, index
        while current is not None:
            current = nodes[current].parent
            steps += 1
            assert steps <= len(nodes), "phát hiện chu trình"
    for index, node in enumerate(nodes):
        for child in node.children:
            assert nodes[child].parent == index
