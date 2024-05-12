import argparse
import os

import plotly.graph_objects as go

from disc_tracker.deprojection.disc_track import DiscTrack


def draw_pitch(fig: go.Figure, width=15.2, length=30.4) -> None:
    """
    Draw the reference pitch on the figure.

    Args:
        fig (Figure): Plotly figure to update.
        width (float, optional): Width of the pitch in meters. Defaults to 15.2.
        length (float, optional): Length of the pitch in meters. Defaults to 30.4.
    """
    # Pitch verts
    px = [-width / 2, width / 2, width / 2, -width / 2]
    py = [length] * 2 + [0] * 2
    # Add pitch mesh
    fig.add_trace(go.Mesh3d(x=px, y=py, z=[0] * 4, color="limegreen", opacity=0.70))


def draw_endzones(fig: go.Figure, width=15.2, length=30.4, endzone_depth=3) -> None:
    """
    Draw the end zones and their associated 2m high box.

    Args:
        fig (Figure): Plotly figure to update
        width (float, optional): Width of the pitch in meters. Defaults to 15.2.
        length (float, optional): Length of the pitch in meters. Defaults to 30.4.
        endzone_depth (int, optional): Depth of the end zones in meters. Defaults to 3.
    """
    # Endzone verts
    ezx = [-width / 2, width / 2, width / 2, -width / 2]
    ezy1 = [length] * 2 + [length - endzone_depth] * 2
    ezy2 = [endzone_depth] * 2 + [0] * 2

    # Add upper and lower end zone box outlines
    for y, z in zip(
        ([ezy1 + [ezy1[0]]]) * 2 + ([ezy2 + [ezy2[0]]]) * 2, ([[0] * 5] + [[2] * 5]) * 2
    ):
        fig.add_trace(
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
        fig.add_trace(
            go.Scatter3d(
                x=[x, x],
                y=[y1, y1],
                z=[0, 2],
                mode="lines",
                line=dict(color="red", width=1),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x, x],
                y=[y2, y2],
                z=[0, 2],
                mode="lines",
                line=dict(color="red", width=1),
            )
        )
    # Fill end zone boxes with transparent meshes
    fig.add_trace(
        go.Mesh3d(
            x=ezx * 2,
            y=ezy1 * 2,
            z=[0] * 4 + [2] * 4,
            color="red",
            opacity=0.05,
            flatshading=True,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=ezx * 2,
            y=ezy2 * 2,
            z=[0] * 4 + [2] * 4,
            color="red",
            opacity=0.05,
            flatshading=True,
        )
    )


def main(directory: str) -> None:
    # Create figure plotting the disc path
    x, y, z = DiscTrack(directory).deproject()
    fig = go.Figure(
        data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="darkblue", width=3),
            name="Disc Path",
        )
    )
    draw_pitch(fig)
    draw_endzones(fig)
    # Plot setttings
    fig.update_layout(scene=dict(aspectmode="data"), showlegend=False)
    # Save plot to file
    fig.write_html(os.path.join(directory, "disc_track.html"))
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Disc Tracker: Plot result with plotly")
    parser.add_argument("directory")
    args = parser.parse_args()
    main(args.directory)
