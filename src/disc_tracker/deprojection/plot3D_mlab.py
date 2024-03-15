import os

import numpy as np
from mayavi import mlab

from disc_tracker import DATA_DIRECTORY

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

# # Pitch verts
px = [-7.6, 7.6, 7.6, -7.6]
py = [30.4, 30.4, 0, 0]
pz = [0, 0, 0, 0]

# EZ verts
ezx = [-7.6, 7.6, 7.6, -7.6, -7.6, 7.6, 7.6, -7.6]
ez1y = [30.4, 30.4, 27.4, 27.4, 30.4, 30.4, 27.4, 27.4]
ez2y = [3, 3, 0, 0, 3, 3, 0, 0]
ezz = [0, 0, 0, 0, 2, 2, 2, 2]

def main() -> None:
    mlab.plot3d(-X, -Y, Z)
    mlab.show()

if __name__ == "__main__":
    main()