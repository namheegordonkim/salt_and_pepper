import os
from argparse import ArgumentParser

import joblib
import numpy as np
import pyvista as pv
import torch
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation

MJCF_PATH = "assets/my_smpl_humanoid.xml"


class AppState:
    def __init__(self):
        self.show_axes = False
        self.pose_idx = 0


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

        imgui.button("Sample")
        imgui.same_line()
        imgui.button("Evaluate")
        imgui.same_line()
        imgui.button("Weight")
        imgui.same_line()
        imgui.button("Learn")

        imgui.end()

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    immapp.run(runner_params=runner_params)


def main():
    pl = ImguiPlotter()

    pl.add_axes()

    # Run the GUI

    app_state = AppState()
    setup_and_run_gui(pl, app_state)

    print(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    main()
