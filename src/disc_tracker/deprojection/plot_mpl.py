import argparse
import os

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import yaml

from disc_tracker.deprojection import settings
from disc_tracker.deprojection.disc_track import DiscTrack


def set_axes_equal(ax: plt.Axes):
    """
    Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.

    Args:
        ax (Axes): Axes object to set equal.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax: plt.Axes, origin: npt.NDArray[np.float64], radius: np.float64):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


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


def draw_pitch(ax: plt.Axes, width=15.2, length=30.4, endzone_depth=3) -> None:
    """
    Draw the reference pitch on the axes.

    Args:
        ax (Axes): Axes to update
        width (float, optional): Width of the pitch in meters. Defaults to 15.2.
        length (float, optional): Length of the pitch in meters. Defaults to 30.4.
        endzone_depth (int, optional): Depth of the end zones in meters. Defaults to 3.
    """
    # Pitch verts
    px = [-width / 2, width / 2, width / 2, -width / 2]
    py = [length] * 2 + [0] * 2
    pxx, pyy = np.meshgrid(px, py)
    pzz = np.zeros_like(pxx)
    ax.plot_surface(pxx, pyy, pzz, color=(0, 1, 0), edgecolor="k", zorder=-1)
    # Endzone verts
    ezx = [-width / 2, width / 2, width / 2, -width / 2] * 2
    ezz = [0] * 4 + [endzone_depth] * 4
    ezx, ezy, ezz = cuboid_data((0, endzone_depth / 2, 1), (width, endzone_depth, 2))
    ax.plot_wireframe(
        np.array(ezx),
        np.array(ezy),
        np.array(ezz),
        facecolor=(0, 0, 0, 0),
        rstride=1,
        cstride=1,
        edgecolor="r",
        zorder=9,
    )
    ezx, ezy, ezz = cuboid_data((0, length - endzone_depth / 2, 1), (width, endzone_depth, 2))
    ax.plot_wireframe(
        np.array(ezx),
        np.array(ezy),
        np.array(ezz),
        facecolor=(0, 0, 0, 0),
        rstride=1,
        cstride=1,
        edgecolor="r",
        zorder=9,
    )


def main(directory: str) -> None:
    disc_path = DiscTrack(directory).deproject()
    ax = plt.axes(projection="3d")
    pitch_dimensions_path = os.path.join(directory, "pitch_dimensions.yaml")
    if os.path.exists(pitch_dimensions_path):
        with open(pitch_dimensions_path) as file:
            pitch_dimensions = yaml.safe_load(file)
    else:
        print("No pitch dimensions file found. Defaulting to UKU measurements.")
        pitch_dimensions = settings.UKU_PITCH_DIMENSIONS
    draw_pitch(ax, **pitch_dimensions)
    ax.plot3D(*disc_path, zorder=10)
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.savefig(os.path.join(directory, "disc_track.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Disc Tracker: Plot result with matplotlib")
    parser.add_argument("directory")
    args = parser.parse_args()
    main(args.directory)
