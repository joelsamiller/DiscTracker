import os

import numpy as np
import plotly.graph_objects as go

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

# Verticies of pitch
px = [-7.6, 7.6, 7.6, -7.6]
py = [30.4, 30.4, 0, 0]
pz = [0, 0, 0, 0]
# Verticies of end zone boxes for mesh
ezx = [-7.6, 7.6, 7.6, -7.6, -7.6, 7.6, 7.6, -7.6]
ez1y = [30.4, 30.4, 27.4, 27.4, 30.4, 30.4, 27.4, 27.4]
ez2y = [3, 3, 0, 0, 3, 3, 0, 0]
ezz = [0, 0, 0, 0, 2, 2, 2, 2]
# Verticies of end zone boxes for outline
EZx = [-7.6, 7.6, 7.6, -7.6, -7.6]
EZ1y = [30.4, 30.4, 27.4, 27.4, 30.4]
EZ2y = [3, 3, 0, 0, 3]

def main() -> None:
    # Create figure plotting the disc path
    fig = go.Figure(
        data=go.Scatter3d(
            x=-X,
            y=-Y,
            z=Z,
            mode="lines",
            line=dict(color="darkblue", width=3),
            name="Disc Path",
        )
    )
    # Add pitch mesh
    fig.add_trace(go.Mesh3d(x=px, y=py, z=pz, color="limegreen", opacity=0.70))
    # Add upper and lower end zone box outlines
    for y, z in zip(
        [EZ1y, EZ1y, EZ2y, EZ2y],
        [[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]],
    ):
        fig.add_trace(
            go.Scatter3d(x=EZx, y=y, z=z, mode="lines", line=dict(color="red", width=1))
        )
    # Add vertial end zone box outlines
    for i in range(4):
        fig.add_trace(
            go.Scatter3d(
                x=[EZx[i], EZx[i]],
                y=[EZ1y[i], EZ1y[i]],
                z=[0, 2],
                mode="lines",
                line=dict(color="red", width=1),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[EZx[i], EZx[i]],
                y=[EZ2y[i], EZ2y[i]],
                z=[0, 2],
                mode="lines",
                line=dict(color="red", width=1),
            )
        )
    # Fill end zone boxes with transparent meshes
    fig.add_trace(
        go.Mesh3d(
            x=ezx,
            y=ez1y,
            z=ezz,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color="red",
            opacity=0.05,
            flatshading=True,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=ezx,
            y=ez2y,
            z=ezz,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color="red",
            opacity=0.05,
            flatshading=True,
        )
    )

    # Plot setttings
    fig.update_layout(scene=dict(aspectmode="data"), showlegend=False)
    # Save plot to file
    fig.write_html("DiscTrack.html")
    fig.show()

if __name__ == "__main__":
    main()
