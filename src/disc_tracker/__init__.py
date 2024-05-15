import argparse
import os

from disc_tracker.video_processing import gg6
from disc_tracker.deprojection import plot_mlab, plot_mpl, plot_plotly


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Disc Tracker",
        description="Using video from two parallel cameras, reconstruct the 3D path of a disc.",
    )
    parser.add_argument("directory")
    parser.add_argument(
        "-p", "--plot-method", default="plotly", choices=["plotly", "mpl", "mlab"]
    )
    parser.add_argument("-plot_only", action="store_true")

    args = parser.parse_args()

    print("*" * 64)
    print("Disc Tracker".center(64))
    print("*" * 64)
    print(f"Directory: {args.directory}")
    print(f"Plotting method: {args.plot_method}")
    print(f"Plot only: {args.plot_only}")

    if not args.plot_only:
        tracks_directory = os.path.join(args.directory, "tracks")
        # Create tracks directory if one doesn't exist
        os.makedirs(tracks_directory, exist_ok=True)
        for chanel in ["left", "right"]:
            video = gg6.load_video(args.directory, chanel)
            print(f"\nTracking objects for {chanel} chanel...")
            tracks = gg6.track_objects(video, chanel)
            disc_id = int(input(f"Enter id of disc in {chanel} chanel: "))
            gg6.save_disc_track(
                os.path.join(tracks_directory, f"{chanel}.npz"), tracks, disc_id
            )

    print("\nPlotting results...")
    eval(f"plot_{args.plot_method}.main(args.directory)")
    print("Done!")
