import argparse

from mayavi import mlab

from disc_tracker.deprojection.disc_track import DiscTrack

def main(directory: str) -> None:
    disc_path = DiscTrack(directory).deproject()
    mlab.plot3d(disc_path[0], disc_path[1], disc_path[2])
    mlab.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Disc Tracker: Plot result with Mayavi"
    )
    parser.add_argument("directory")
    args = parser.parse_args()
    main(args.directory)