from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import pyvista as pv
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
import scipy

from learners import CEMLearner

np.set_printoptions(precision=3, suppress=True)


@dataclass
class Vis:
    mesh: pv.DataSet
    actor: pv.Actor


class RewardFunction:
    def __init__(self, mu: np.ndarray, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pdf_x = scipy.stats.norm.pdf(x, loc=self.mu[0], scale=self.sigma)
        pdf_y = scipy.stats.norm.pdf(y, loc=self.mu[1], scale=self.sigma)
        return pdf_x * pdf_y


class AppState:
    def __init__(self, learner: CEMLearner, reward_vis: Vis, policy_vis: Vis):
        self.show_axes = False
        self.show_reward_landscape = True
        self.show_policy = False
        self.pose_idx = 0
        self.learner = learner
        self.point_cloud = None
        self.point_cloud_actor = None
        self.reward_vis = reward_vis
        self.policy_vis = policy_vis

        self.n_samples = 10
        self.elite_proportion = 0.5
        self.init_sigma = 0.15

        self.first_time = True
        self.stepping = False
        self.n_steps = 1
        self.current_step = 0
        self.step_stage = -1


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    def gui():
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )

        changed, app_state.init_sigma = imgui.slider_float(
            "Initial Sigma", app_state.init_sigma, 0.001, 3
        )

        clicked = imgui.button("Reset")
        if clicked or app_state.first_time:
            app_state.learner.reset(app_state.init_sigma)
            if app_state.point_cloud_actor is not None:
                pl.remove_actor(app_state.point_cloud_actor)

        changed, app_state.n_samples = imgui.slider_int(
            "Number of Samples", app_state.n_samples, 1, 1000
        )
        changed, app_state.elite_proportion = imgui.slider_float(
            "Elite Proportion", app_state.elite_proportion, 0, 1
        )

        clicked = imgui.button("Sample")
        if clicked or (app_state.stepping and app_state.step_stage == 0):
            if app_state.point_cloud_actor is not None:
                pl.remove_actor(app_state.point_cloud_actor)

            print("Sampling")
            app_state.learner.sample(app_state.n_samples)
            samples = np.zeros((app_state.learner.samples.shape[0], 3))
            samples[:, :2] = app_state.learner.samples
            samples[:, -1] = 1
            app_state.point_cloud = pv.PolyData(samples)
            app_state.point_cloud.point_data["weight"] = np.zeros(
                app_state.learner.samples.shape[0]
            )
            app_state.point_cloud_actor = pl.add_pointcloud(
                app_state.point_cloud, scalars="weight", color="blue", point_size=10
            )

        imgui.same_line()

        clicked = imgui.button("Evaluate")
        if clicked or (app_state.stepping and app_state.step_stage == 1):
            print("Evaluating")
            app_state.learner.evaluate()
            app_state.point_cloud.points[:, -1] = app_state.learner.rewards

        imgui.same_line()

        clicked = imgui.button("Weigh")
        if clicked or (app_state.stepping and app_state.step_stage == 2):
            print("Weighing")
            app_state.learner.weigh()
            app_state.point_cloud.point_data["weight"] = app_state.learner.weights

        imgui.same_line()

        clicked = imgui.button("Learn")
        if clicked or (app_state.stepping and app_state.step_stage == 3):
            print("Learning")
            app_state.learner.learn()

        # if clicked or app_state.first_time:
        pdf_ax = scipy.stats.norm.pdf(
            app_state.reward_vis.mesh.points[:, 0],
            loc=app_state.learner.mu[0],
            scale=app_state.learner.sigma[0],
        )
        pdf_ay = scipy.stats.norm.pdf(
            app_state.reward_vis.mesh.points[:, 1],
            loc=app_state.learner.mu[1],
            scale=app_state.learner.sigma[1],
        )
        pdf_az = pdf_ax * pdf_ay
        pdf_az /= np.sum(pdf_az)
        pdf_az /= np.max(pdf_az)
        app_state.policy_vis.mesh.points[:, 2] = pdf_az

        imgui.text(f"Current mu: {app_state.learner.mu}")
        imgui.text(f"Current sigma: {app_state.learner.sigma}")
        imgui.text(f"Current mean reward: {np.mean(app_state.learner.rewards):.3f}")
        imgui.text(
            f"Current weighted mean reward: {np.sum(app_state.learner.weights * app_state.learner.rewards) / np.sum(app_state.learner.weights):.3f}"
        )

        changed, app_state.n_steps = imgui.slider_int(
            "Number of Steps", app_state.n_steps, 1, 100
        )
        clicked = imgui.button("Step")
        if clicked and not app_state.stepping:
            print("Stepping")
            app_state.stepping = True
            app_state.step_stage = -1

        changed, app_state.show_reward_landscape = imgui.checkbox(
            "Show Reward Landscape", app_state.show_reward_landscape
        )

        changed, app_state.show_policy = imgui.checkbox(
            "Show Policy", app_state.show_policy
        )

        imgui.end()

        if app_state.stepping:
            app_state.step_stage += 1

        # Incrementing step number after all stages are passed
        if app_state.stepping and app_state.step_stage > 3:
            app_state.current_step += 1
            app_state.step_stage = 0

        # Resetting step number after reaching the end
        if app_state.current_step >= app_state.n_steps:
            app_state.stepping = False
            app_state.current_step = 0
            app_state.step_stage = -1

        if app_state.show_reward_landscape:
            app_state.reward_vis.actor.SetVisibility(True)
        else:
            app_state.reward_vis.actor.SetVisibility(False)

        if app_state.show_policy:
            app_state.policy_vis.actor.SetVisibility(True)
        else:
            app_state.policy_vis.actor.SetVisibility(False)

        app_state.first_time = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    immapp.run(runner_params=runner_params)


def main(args):
    pl = ImguiPlotter()
    pl.add_axes()

    true_mu = np.array([0.5, 0.5])
    true_sigma = 0.5
    reward_function = RewardFunction(true_mu, true_sigma)
    learner = CEMLearner(reward_function)

    # Unimodal Gaussian hill for reward landscape
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = reward_function.evaluate(X, Y)
    mesh = pv.StructuredGrid(X, Y, Z)
    mesh.point_data["reward"] = Z.reshape(-1)
    actor = pl.add_mesh(mesh, show_edges=True)
    reward_vis = Vis(mesh, actor)

    # Unimodal Gaussian policy landscape
    Z = np.zeros_like(X)
    mesh = pv.StructuredGrid(X, Y, Z)
    actor = pl.add_mesh(mesh, opacity=0.8, color="red")
    actor.SetVisibility(False)
    policy_vis = Vis(mesh, actor)

    # Axis labels
    x_label_mesh = pv.Text3D("Salt", depth=0.01, height=0.1)
    x_label_actor = pl.add_mesh(x_label_mesh, color="black")
    m = np.eye(4)
    m[:3, 3] = [0, -1.1, 0]
    x_label_actor.user_matrix = m

    y_label_mesh = pv.Text3D("Pepper", depth=0.01, height=0.1)
    y_label_actor = pl.add_mesh(y_label_mesh, color="black")
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    m[:3, 3] = [-1.1, 0, 0]
    y_label_actor.user_matrix = m

    # Run the GUI
    app_state = AppState(learner, reward_vis, policy_vis)
    setup_and_run_gui(pl, app_state)

    print(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
