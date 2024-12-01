import os
from collections import OrderedDict

import numpy as np
import numpy.testing as npt

from disc_tracker.deprojection.disc_track import DiscTrack
from disc_tracker.video_processing.gg6 import save_disc_track
from disc_tracker.video_processing.tracker import Object


def test_complete_tracks():
    test_object = Object(0, np.array([1.0, 2.0]))
    test_object.update_position(2, np.array([3.0, 4.0]))
    try:
        os.mkdir("tmp")
        save_disc_track(
            os.path.join("tmp", "test_track.npz"), OrderedDict({0: test_object}), 0
        )
        with np.load(os.path.join("tmp", "test_track.npz")) as test_file:
            test_track = DiscTrack.complete_tracks(test_file, np.array([0, 1, 2]))
    finally:
        # Clean up even if there is an error above
        os.remove(os.path.join("tmp", "test_track.npz"))
        os.rmdir("tmp")

    npt.assert_array_equal(test_track["x"], np.array([1.0, 2.0, 3.0]))
    npt.assert_array_equal(test_track["y"], np.array([2.0, 3.0, 4.0]))


def test_get_time_index():
    test_index = DiscTrack.get_time_index(
        np.array([2, 3, 4, 6, 7]), np.array([1, 2, 3, 5, 8])
    )

    npt.assert_array_equal(test_index, np.array([2, 3, 4, 5, 6, 7]))
