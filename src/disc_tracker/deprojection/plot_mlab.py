from mayavi import mlab

from disc_tracker.deprojection.disc_track import DiscTrack

def main() -> None:
    disc_path = DiscTrack("rosie_pull").deproject()
    mlab.plot3d(-disc_path[0], -disc_path[1], disc_path[2])
    mlab.show()

if __name__ == "__main__":
    main()