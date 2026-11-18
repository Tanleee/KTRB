"""Làm mượt đường đi hình học bằng cắt tắt (shortcut) tham lam.

Từ điểm hiện tại, nối thẳng tới điểm *xa nhất* trên đường gốc mà cạnh nối không va chạm
(kiểm tra liên tục với đúng bán kính dùng khi lập kế hoạch), rồi lặp lại từ điểm đó.
Kết quả luôn hợp lệ và không dài hơn đường gốc (bất đẳng thức tam giác).
"""

import numpy as np


def shortcut_path(path, environment, robot_radius):
    points = np.asarray(path, dtype=float)
    if len(points) <= 2:
        return points.copy()
    kept = [points[0]]
    i = 0
    last = len(points) - 1
    while i < last:
        j = last
        # Điểm kế tiếp (i+1) luôn nối được vì nó là cạnh của đường gốc hợp lệ.
        while j > i + 1 and not environment.edge_free(points[i], points[j], robot_radius):
            j -= 1
        kept.append(points[j])
        i = j
    return np.asarray(kept)
