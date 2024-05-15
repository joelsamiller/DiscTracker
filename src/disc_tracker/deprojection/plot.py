import os

from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import yaml
from disc_tracker.deprojection import settings
from disc_tracker.deprojection.disc_track import DiscTrack


class Plot:
    def __init__(self, directory: str) -> None:
        self.disc_path = DiscTrack(directory).deproject()
        self.pitch_dimensions = self.get_pitch_dimensions(directory)
        self.directory = directory
        self.create_figure()
        self.draw_pitch(**self.pitch_dimensions)
        self.draw_endzones(**self.pitch_dimensions)
        self.plot_disc_path()
        self.apply_settings()

    def create_figure(self):
        pass

    def draw_pitch(self, width: float, length: float, endzone_depth: float):
        pass

    def plot_disc_path(self):
        pass

    def draw_endzones(self, width: float, length: float, endzone_depth: float):
        pass

    def save_figure(self):
        pass

    def show_figure(self):
        pass

    def apply_settings(self):
        pass

    @staticmethod
    def get_pitch_dimensions(directory: str) -> dict:
        pitch_dimensions_path = os.path.join(directory, "pitch_dimensions.yaml")
        if os.path.exists(pitch_dimensions_path):
            with open(pitch_dimensions_path) as file:
                return yaml.safe_load(file)

        print("No pitch dimensions file found. Defaulting to UKU measurements.")
        return settings.UKU_PITCH_DIMENSIONS


class PlotlyPlot(Plot):
    def create_figure(self):
        x, y, z = self.disc_path
        self.fig = go.Figure(
            data=go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="darkblue", width=3),
                name="Disc Path",
            )
        )

    def apply_settings(self):
        self.fig.update_layout(scene=dict(aspectmode="data"), showlegend=False)

    def draw_pitch(self, width: float, length: float, endzone_depth: float):
        # Pitch verts
        px = [-width / 2, width / 2, width / 2, -width / 2]
        py = [length] * 2 + [0] * 2
        # Add pitch mesh
        self.fig.add_trace(
            go.Mesh3d(x=px, y=py, z=[0] * 4, color="limegreen", opacity=0.70)
        )

    def draw_endzones(self, width: float, length: float, endzone_depth: float):
        # Endzone verts
        ezx = [-width / 2, width / 2, width / 2, -width / 2]
        ezy1 = [length] * 2 + [length - endzone_depth] * 2
        ezy2 = [endzone_depth] * 2 + [0] * 2

        # Add upper and lower end zone box outlines
        for y, z in zip(
            ([ezy1 + [ezy1[0]]]) * 2 + ([ezy2 + [ezy2[0]]]) * 2,
            ([[0] * 5] + [[2] * 5]) * 2,
        ):
            self.fig.add_trace(
                go.Scatter3d(
                    x=ezx + [ezx[0]],
                    y=y,
                    z=z,
                    mode="lines",
                    line={"color": "red", "width": 1},
                )
            )
        # Add vertial end zone box outlines
        for x, y1, y2 in zip(ezx, ezy1, ezy2):
            self.fig.add_trace(
                go.Scatter3d(
                    x=[x, x],
                    y=[y1, y1],
                    z=[0, 2],
                    mode="lines",
                    line=dict(color="red", width=1),
                )
            )
            self.fig.add_trace(
                go.Scatter3d(
                    x=[x, x],
                    y=[y2, y2],
                    z=[0, 2],
                    mode="lines",
                    line=dict(color="red", width=1),
                )
            )
        # Fill end zone boxes with transparent meshes
        self.fig.add_trace(
            go.Mesh3d(
                x=ezx * 2,
                y=ezy1 * 2,
                z=[0] * 4 + [2] * 4,
                color="red",
                opacity=0.05,
                flatshading=True,
            )
        )
        self.fig.add_trace(
            go.Mesh3d(
                x=ezx * 2,
                y=ezy2 * 2,
                z=[0] * 4 + [2] * 4,
                color="red",
                opacity=0.05,
                flatshading=True,
            )
        )

    def save_figure(self):
        self.fig.write_html(os.path.join(self.directory, "disc_track.html"))

    def show_figure(self):
        self.fig.show()


class MatplotlibPlot(Plot):
    def create_figure(self):
        self.fig = plt.axes(projection="3d")

    def draw_pitch(self, width: float, length: float, endzone_depth: float):
        # Pitch verts
        px = [-width / 2, width / 2, width / 2, -width / 2]
        py = [length] * 2 + [0] * 2
        pxx, pyy = np.meshgrid(px, py)
        pzz = np.zeros_like(pxx)
        self.fig.plot_surface(pxx, pyy, pzz, color=(0, 1, 0), edgecolor="k", zorder=-1)

    def draw_endzones(self, width: float, length: float, endzone_depth: float):
        # Endzone verts
        ezx = [-width / 2, width / 2, width / 2, -width / 2] * 2
        ezz = [0] * 4 + [endzone_depth] * 4
        ezx, ezy, ezz = self.cuboid_data(
            (0, endzone_depth / 2, 1), (width, endzone_depth, 2)
        )
        self.fig.plot_wireframe(
            np.array(ezx),
            np.array(ezy),
            np.array(ezz),
            facecolor=(0, 0, 0, 0),
            rstride=1,
            cstride=1,
            edgecolor="r",
            zorder=9,
        )
        ezx, ezy, ezz = self.cuboid_data(
            (0, length - endzone_depth / 2, 1), (width, endzone_depth, 2)
        )
        self.fig.plot_wireframe(
            np.array(ezx),
            np.array(ezy),
            np.array(ezz),
            facecolor=(0, 0, 0, 0),
            rstride=1,
            cstride=1,
            edgecolor="r",
            zorder=9,
        )

    @staticmethod
    def cuboid_data(center: tuple[float], size: tuple[float]):
        """
        Create an array for plotting a cuboid

        Args:
            center (tuple[float]): Coordinates of the cuboid centre (x, y, z).
            size (tuple[float]): Size of the cuboid (x, y, z).

        Returns:
            _type_: _description_
        """

        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(center, size)]
        # get the length, width, and height
        l, w, h = size
        x = [
            [
                o[0],
                o[0] + l,
                o[0] + l,
                o[0],
                o[0],
            ],  # x coordinate of points in bottom surface
            # x coordinate of points in upper surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],
            # x coordinate of points in outside surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        ]  # x coordinate of points in inside surface
        y = [
            [
                o[1],
                o[1],
                o[1] + w,
                o[1] + w,
                o[1],
            ],  # y coordinate of points in bottom surface
            # y coordinate of points in upper surface
            [o[1], o[1], o[1] + w, o[1] + w, o[1]],
            # y coordinate of points in outside surface
            [o[1], o[1], o[1], o[1], o[1]],
            [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
        ]  # y coordinate of points in inside surface
        z = [
            [o[2], o[2], o[2], o[2], o[2]],  # z coordinate of points in bottom surface
            # z coordinate of points in upper surface
            [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
            # z coordinate of points in outside surface
            [o[2], o[2], o[2] + h, o[2] + h, o[2]],
            [o[2], o[2], o[2] + h, o[2] + h, o[2]],
        ]  # z coordinate of points in inside surface
        return x, y, z

    def apply_settings(self):
        self.fig.set_box_aspect([1, 1, 1])
        limits = np.array(
            [
                self.fig.get_xlim3d(),
                self.fig.get_ylim3d(),
                self.fig.get_zlim3d(),
            ]
        )
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        x, y, z = origin
        self.fig.set_xlim3d([x - radius, x + radius])
        self.fig.set_ylim3d([y - radius, y + radius])
        self.fig.set_zlim3d([z - radius, z + radius])

    def plot_disc_path(self):
        self.fig.plot3D(*self.disc_path, zorder=10)

    def save_figure(self):
        plt.savefig(os.path.join(self.directory, "disc_track.png"))

    def show_figure(self):
        plt.show()


class MlabPlot(Plot):
    def plot_disc_path(self):
        mlab.plot3d(*self.disc_path)

    def show_figure(self):
        mlab.show()
