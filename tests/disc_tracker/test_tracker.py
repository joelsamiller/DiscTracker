import numpy as np
import numpy.testing as npt

from disc_tracker.video_processing.tracker import Tracker, Object


class TestObject:
    def test_init(self):
        test_object = Object(0, np.array([1.0, 2.0]))

        npt.assert_array_equal(test_object.position, np.array([1.0, 2.0, 0]))
        npt.assert_array_equal(test_object.track, np.array([[1.0, 2.0, 0]]))

    def test_update_position(self):
        test_object = Object(0, np.array([1.0, 2.0]))
        test_object.update_position(1, np.array([3.0, 4.0]))

        npt.assert_array_equal(test_object.position, np.array([3.0, 4.0, 1]))
        npt.assert_array_equal(
            test_object.track, np.array([[1.0, 2.0, 0], [3.0, 4.0, 1]])
        )
