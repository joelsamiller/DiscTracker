import os

import numpy as np
import matplotlib.pyplot as plt

from disc_tracker import DATA_DIRECTORY


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
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


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def cuboid_data(center, size):
    """
    Create a data array for cuboid plotting.


    ============= ================================================
    Argument      Description
    ============= ================================================
    center        center of the cuboid, triple
    size          size of the cuboid, triple, (x_length,y_width,z_height)
    :type size: tuple, numpy.array, list
    :param size: size of the cuboid, triple, (x_length,y_width,z_height)
    :type center: tuple, numpy.array, list
    :param center: center of the cuboid, triple, (x,y,z)


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


# Camera set up info
d = 4  # Horizontal separation of the cameras in meters
f = 0.004 * 1280 / 0.0047  # Focal length converted to pixels
c = 1.8  # Distance between cameras and back of endzone in meters
h = 2.8  # Height of cameras above ground in meters

# Load track data
tracks_directory = os.path.join(DATA_DIRECTORY, "rosie_pull", "tracks")
L = np.load(os.path.join(tracks_directory, "left.npz"))
R = np.load(os.path.join(tracks_directory, "right.npz"))
# Trim right channel to match left
xl = L["x"] - 640
xr = R["x"][3::] - 640
zl = L["y"] - 360
zr = R["y"][3::] - 360

# Deproject
X = 0.5 * d * (xl + xr) / (xr - xl)
Z = -(0.5 * d * (zl + zr) / (xr - xl)) + h
Y = d * f / (xr - xl) - c

def main() -> None:
    ax = plt.axes(projection="3d")

    ax.plot3D(-X, -Y, Z, zorder=10)
    # # Pitch verts
    px = [-7.6, 7.6, 7.6, -7.6]
    py = [30.4, 30.4, 0, 0]
    pz = [0, 0, 0, 0]
    pxx, pyy = np.meshgrid(px, py)
    pzz = np.zeros_like(pxx)
    ax.plot_surface(pxx, pyy, pzz, color=(0, 1, 0), edgecolor="k", zorder=-1)
    # EZ verts
    ezx = [-7.6, 7.6, 7.6, -7.6, -7.6, 7.6, 7.6, -7.6]
    ezz = [0, 0, 0, 0, 2, 2, 2, 2]
    ezx, ezy, ezz = cuboid_data((0, 1.5, 1), (15.2, 3, 2))
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
    ezx, ezy, ezz = cuboid_data((0, 28.9, 1), (15.2, 3, 2))
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


    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()

if __name__ == "__main__":
    main()
