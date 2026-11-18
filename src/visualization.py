from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as CirclePatch, Rectangle as RectanglePatch, FancyBboxPatch
from matplotlib.widgets import Button, TextBox
from matplotlib.ticker import FuncFormatter

from .collision import Circle

COLORS = {"RRT": "#3779ad", "RRT*": "#168b82"}


def decimal(value):
    return f"{value:.3f}".replace(".", ",")


def status_text(iteration, budget, nodes, elapsed, first, length, rewires):
    first = "Chưa có lời giải" if first is None else decimal(first) + " s"
    length = "Chưa tìm được đường" if length is None else decimal(length) + " m"
    return (f"Vòng lặp {iteration}/{budget} | Số nút: {int(nodes)}\n"
            f"Thời gian lập kế hoạch: {decimal(elapsed)} s\n"
            f"Tới lời giải đầu: {first}\nĐộ dài: {length} | Đổi cha: {rewires}")


def draw_environment(ax, env, radius):
    xmin, xmax, ymin, ymax = env.bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.set_facecolor("#f8fafc")
    ax.grid(alpha=0.15, zorder=0)
    ax.add_patch(RectanglePatch((xmin+radius, ymin+radius), xmax-xmin-2*radius,
                               ymax-ymin-2*radius, fill=False, linestyle="--",
                               edgecolor="#b9c3ce", linewidth=0.8))
    for obstacle in env.obstacles:
        if isinstance(obstacle, Circle):
            expanded = CirclePatch((obstacle.x, obstacle.y), obstacle.radius+radius)
            physical = CirclePatch((obstacle.x, obstacle.y), obstacle.radius)
        else:
            expanded = FancyBboxPatch((obstacle.x-radius, obstacle.y-radius), obstacle.width+2*radius,
                                     obstacle.height+2*radius, boxstyle=f"round,pad=0,rounding_size={radius}")
            physical = RectanglePatch((obstacle.x, obstacle.y), obstacle.width, obstacle.height)
        expanded.set(facecolor="#e2aa69", edgecolor="none", alpha=0.35, zorder=1)
        physical.set(facecolor="#394b60", edgecolor="#29394c", zorder=2)
        ax.add_patch(expanded)
        ax.add_patch(physical)
    ax.scatter(*env.start, c="#198754", s=75, marker="o", edgecolor="white", zorder=6, label="Điểm đầu")
    ax.scatter(*env.goal, c="#be3455", s=140, marker="*", edgecolor="white", zorder=6, label="Đích")
    ax.add_patch(CirclePatch(env.start, radius, fill=False, edgecolor="#198754", linewidth=1.4, zorder=6))


class PlannerView:
    def __init__(self, ax, planner):
        self.ax = ax
        self.planner = planner
        draw_environment(ax, planner.environment, planner.config.robot_radius)
        self.edges = LineCollection([], colors=COLORS[planner.name], linewidths=0.45, alpha=0.45, zorder=3)
        ax.add_collection(self.edges)
        self.nodes = ax.scatter([], [], s=2, c=COLORS[planner.name], alpha=0.4, zorder=3)
        self.path_line, = ax.plot([], [], color="#ed7d24", linewidth=2.8, zorder=5, label="Đường đi tốt nhất")
        self.rewired = LineCollection([], colors="#be3455", linewidths=1.1, zorder=4)
        ax.add_collection(self.rewired)
        self.status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                              bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"}, zorder=8)
        ax.legend(loc="lower right", fontsize=8)
        self.update()

    def update(self):
        planner = self.planner
        nodes = planner.tree.nodes
        self.edges.set_segments([(nodes[node.parent].position, node.position) for node in nodes[1:]])
        self.nodes.set_offsets(planner.tree.positions)
        path = planner.path()
        if path is None:
            self.path_line.set_data([], [])
        else:
            self.path_line.set_data(path[:, 0], path[:, 1])
        indices = planner.history[-1].rewired
        self.rewired.set_segments([(nodes[nodes[i].parent].position, nodes[i].position) for i in indices])
        self.ax.set_title(f"{planner.name} | Bản đồ {planner.environment.name.title()} | seed {planner.seed}", fontweight="bold")
        self.status.set_text(status_text(planner.iteration, planner.config.max_iterations, len(nodes),
                                         planner.elapsed, planner.first_solution_time, planner.best_cost,
                                         planner.rewire_count))


def make_figure(planners):
    fig, axes = plt.subplots(1, len(planners), figsize=(7*len(planners), 6.4), squeeze=False)
    fig.suptitle("Lập kế hoạch đường đi hình học bằng RRT và RRT*", fontsize=16, fontweight="bold")
    views = [PlannerView(ax, planner) for ax, planner in zip(axes[0], planners)]
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, views


def save_snapshot(planners, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = make_figure(planners)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tracking(environment, radius, entries, destination):
    """Hình bám đường: đường tham chiếu (nét đứt), quỹ đạo thực (nét liền), điểm va chạm (x đỏ)."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(entries), figsize=(6.5 * len(entries), 6.2), squeeze=False)
    fig.suptitle("Bám đường bằng Pure Pursuit (robot 2 bánh vi sai)", fontsize=16, fontweight="bold")
    for ax, (title, reference, outcome) in zip(axes[0], entries):
        draw_environment(ax, environment, radius)
        ax.plot(*reference.T, "--", color="#6b7280", linewidth=1.6, zorder=4, label="Đường tham chiếu")
        ax.plot(outcome.trajectory[:, 0], outcome.trajectory[:, 1], color="#c2410c", linewidth=2.0,
                zorder=5, label="Quỹ đạo thực")
        if outcome.collided:
            ax.scatter(*outcome.trajectory[-1, :2], marker="x", s=120, c="#dc2626", linewidths=3, zorder=7)
        status = "va chạm" if outcome.collided else ("tới đích" if outcome.reached else "hết thời gian")
        time_text = f"{outcome.time:.1f}".replace(".", ",")
        ax.set_title(f"{title} | {status}\nT = {time_text} s | RMS = {decimal(outcome.rms_cte)} m | "
                     f"max = {decimal(outcome.max_cte)} m", fontweight="bold", fontsize=10)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


class Simulation:
    def __init__(self, factory, seed=0, batch=10, output_dir="results/demo"):
        self.factory = factory
        self.seed = seed
        self.batch = batch
        self.output_dir = Path(output_dir)
        self.running = False
        self.planners = factory(seed)
        self.fig, self.views = make_figure(self.planners)
        self.fig.subplots_adjust(bottom=0.2)
        self.buttons = []
        for x, text, callback in ((0.06, "Chạy / Dừng", self.toggle), (0.22, "Một bước", self.single_step),
                                  (0.34, "Chạy lại", self.reset), (0.46, "Seed kế", self.next_seed),
                                  (0.61, "Lưu PNG", self.save)):
            button = Button(self.fig.add_axes((x, 0.085, 0.12, 0.045)), text)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.seed_box = TextBox(self.fig.add_axes((0.82, 0.085, 0.1, 0.045)), "Seed ", initial=str(seed))
        self.seed_box.on_submit(self.set_seed)
        self.notice = self.fig.text(0.5, 0.155, "Đang dừng. Space: chạy/dừng; →: một bước; R: chạy lại.",
                                    ha="center", fontsize=9)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def refresh(self):
        for view in self.views:
            view.update()
        self.fig.canvas.draw_idle()

    def advance(self, count):
        for planner in self.planners:
            for _ in range(count):
                if planner.done:
                    break
                planner.step()
        if all(planner.done for planner in self.planners):
            self.running = False
            self.notice.set_text("Hoàn tất. Chạy lại hoặc chọn seed khác.")
        self.refresh()

    def tick(self):
        if self.running:
            self.advance(self.batch)
        return True

    def toggle(self, _=None):
        self.running = not self.running
        self.notice.set_text("Đang chạy" if self.running else "Đang dừng")
        self.refresh()

    def single_step(self, _=None):
        self.running = False
        self.advance(1)

    def reset(self, _=None):
        self.running = False
        self.planners = self.factory(self.seed)
        for view, planner in zip(self.views, self.planners):
            view.planner = planner
        self.notice.set_text(f"Chạy lại với seed {self.seed}")
        self.refresh()

    def next_seed(self, _=None):
        self.seed_box.set_val(str(self.seed+1))

    def set_seed(self, text):
        try:
            seed = int(text)
            if seed < 0:
                raise ValueError
        except ValueError:
            self.notice.set_text("Seed phải là số nguyên không âm")
            self.refresh()
            return
        self.seed = seed
        self.reset()

    def save(self, _=None):
        path = self.output_dir / f"{self.planners[0].environment.name}_seed{self.seed}_i{self.planners[0].iteration}.png"
        save_snapshot(self.planners, path)
        self.notice.set_text(f"Đã lưu {path}")
        self.refresh()

    def key_press(self, event):
        # Let the text field receive spaces/arrows without triggering simulation.
        if event.inaxes == self.seed_box.ax:
            return
        if event.key == " ":
            self.toggle()
        elif event.key == "right":
            self.single_step()
        elif event.key == "r":
            self.reset()

    def show(self):
        plt.show()

