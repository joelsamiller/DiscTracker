import os

import numpy as np
from mayavi import mlab

# Camera set up info
d = 4  # Horizontal separation of the cameras in meters
f = 0.004 * 1280 / 0.0047  # Focal length converted to pixels
c = 1.8  # Distance between cameras and back of endzone in meters
h = 2.8  # Height of cameras above ground in meters

# Load track data
data_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data")
tracks_directory = os.path.join(data_directory, "rosie_pull", "tracks")
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

mlab.plot3d(-X, -Y, Z)
mlab.show()

# fig = go.Figure(data = go.Scatter3d(x=-X, y=-Y, z=Z, mode='lines', line=dict(color='darkblue', width=2)))

# fig.add_trace(go.Mesh3d(x=px, y=py, z=pz, color='green', opacity=0.50))

# fig.add_trace(go.Mesh3d(x=ezx, y=ez1y, z=ezz, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#                         j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#                         k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color='red', opacity=0.20, flatshading=True))

# fig.add_trace(go.Mesh3d(x=ezx, y=ez2y, z=ezz, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
#                         j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
#                         k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], color='red', opacity=0.20, flatshading=True))

# fig.update_layout(scene=dict(aspectmode='data'))
# fig.show()
