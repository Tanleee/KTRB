# So sánh RRT và RRT* trong lập kế hoạch đường đi 2D cho robot di động (Nhóm 04)

Mô phỏng và thực nghiệm so sánh RRT với RRT* cho robot hình đĩa trên bản đồ 2D tĩnh (kiểm tra va chạm liên tục
trên toàn cạnh, cùng chuỗi mẫu theo seed), kèm làm mượt đường và bám đường bằng Pure Pursuit cho robot 2 bánh vi sai.

## 1. Cài đặt

Yêu cầu Python ≥ 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `src/` | Lõi thuật toán: `smoothing.py` (làm mượt shortcut), `tracking.py` (Pure Pursuit, robot vi sai), `rrt.py`, `rrt_star.py`, `node.py` (cây + đổi cha), `collision.py`, `environment.py`, `visualization.py` |
| `configs/maps/` | Ba bản đồ: `simple`, `complex`, `narrow` (JSON) |
| `experiments/benchmark.py` | Chạy 3 bản đồ × 20 seed × 2 thuật toán, ghi CSV/JSON |
| `experiments/tracking_benchmark.py` | Lập kế hoạch → làm mượt → bám đường: tỷ lệ thành công, sai số bám đường (CTE), thời gian |
| `experiments/plots.py` | Tổng hợp thống kê (Wilson, Wilcoxon) và vẽ hình |
| `experiments/extra.py` | Thực nghiệm bổ sung: bán kính lân cận, độ phức tạp, bộ nhớ, baseline, độ nhạy, cùng thời gian |
| `tests/` | Kiểm thử bất biến cây, va chạm, tái lập, tính hợp lệ của đường |
| `results/` | Dữ liệu và hình của các đợt chạy |

## 3. Chạy mô phỏng một lần

```bash
python main.py --map simple --seed 0                 # in kết quả của RRT và RRT*
python main.py --map complex --seed 0 --save out.png  # lưu hình hai cây cạnh nhau
python main.py --map narrow --seed 1 --interactive   # giao diện tương tác
```

Tham số chính: `--algorithm {rrt,rrt-star,both}`, `--iterations`, `--step-size`, `--goal-bias`, `--robot-radius`, `--gamma`, `--stop-on-first`.

## 4. Chạy benchmark

```bash
# Đợt chuẩn trong báo cáo (20 seed, 2500 vòng lặp); mỗi cấu hình đo 5 lần, lấy trung vị thời gian
python -m experiments.benchmark --output results/benchmark_v2 --seeds 20 --repeats 5

# Vẽ lại hình/thống kê từ dữ liệu đã có
python -m experiments.plots --input results/benchmark_v2
```

`--output` phải là thư mục chưa có kết quả. Thông tin máy, phiên bản thư viện và γ hiệu dụng được ghi vào `settings.json`.
Số đo thời gian phụ thuộc máy; chỉ so sánh các số được đo trong cùng một đợt chạy.

## 5. Thực nghiệm bổ sung

```bash
python -m experiments.extra radius                          # r_n thực tế, ngưỡng γ
python -m experiments.extra scaling --input results/benchmark_v2        # số mũ thời gian theo vòng lặp
python -m experiments.extra first-solution --input results/benchmark_v2 # RRT dừng ở lời giải đầu vs chạy hết ngân sách
python -m experiments.extra reference --input results/benchmark_v2      # optimality gap so với đồ thị tầm nhìn
python -m experiments.extra memory --seeds 5                # bộ nhớ đỉnh (tracemalloc)
python -m experiments.extra equal-time --budget 1.0 --seeds 20          # cùng ngân sách thời gian
python -m experiments.extra sensitivity --seeds 10          # độ nhạy η, p_g, γ
```

Kết quả ghi vào `results/extra/`.

## 6. Làm mượt và bám đường bằng Pure Pursuit

```bash
python main.py --map complex --track --smooth --safety-margin 0.05 --lookahead 0.6 --save-track track.png
python -m experiments.tracking_benchmark --output results/tracking
```

Tham số: `--track` (bật bám đường), `--smooth` (cắt tắt trước khi bám), `--lookahead` (L, m), `--v-max` (m/s),
`--omega-max` (rad/s), `--safety-margin` (lập kế hoạch với bán kính robot + margin, còn va chạm khi bám đường
luôn kiểm tra với bán kính thật), `--save-track` (lưu hình).

- Mô hình: unicycle `x' = v cosθ, y' = v sinθ, θ' = ω`; Pure Pursuit `κ = 2 sinα / d`, `ω = vκ`,
  giới hạn `|v| ≤ v_max`, `|ω| ≤ ω_max`, giảm tốc khi cua gấp và gần đích, quay tại chỗ khi |α| > 90°.
- Chỉ tiêu: tỷ lệ thành công (tới đích, không va chạm), thời gian hoàn thành, sai số bám đường
  (RMS và lớn nhất, so với đường tham chiếu), kiểm tra va chạm liên tục của quỹ đạo thực.
- Quan sát: đường lập kế hoạch với margin = 0 chạm sát biên vật cản nở nên Pure Pursuit cắt góc dễ va chạm,
  nhất là khi L lớn; thêm margin nhỏ hoặc giảm L cải thiện rõ tỷ lệ thành công. Margin quá lớn làm đóng khe hẹp
  (bản đồ `narrow`): với margin 0,05 m, RRT/RRT* chỉ lập được đường ở một phần số seed.

## 7. Kiểm thử

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
