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


class TestTracker:
    def test_register(self):
        test_tracker = Tracker()
        test_tracker.register(np.array([1.0, 2.0]))

        assert len(test_tracker.objects) == 1
        assert type(test_tracker.objects[0]) is Object
        assert test_tracker.disappeared[0] == 0
        assert test_tracker.next_id == 1

    def test_deregister(self):
        test_tracker = Tracker()
        test_tracker.register(np.array([1.0, 2.0]))
        test_tracker.deregister(0)

        assert len(test_tracker.objects) == 0
        assert len(test_tracker.disappeared) == 0

    def test_update_empty_tracker_no_new_positions(self):
        test_tracker = Tracker()
        test_tracker.update(np.array([[]]))

        assert len(test_tracker.objects) == 0
        assert len(test_tracker.disappeared) == 0
        assert test_tracker.current_time == 1

    def test_update_no_new_positions(self):
        test_tracker = Tracker()
        test_tracker.register(np.array([1.0, 2.0]))
        test_tracker.update(np.array([[]]))

        assert len(test_tracker.objects) == 1
        assert test_tracker.disappeared[0] == 1
        assert test_tracker.current_time == 1

    def test_update_empty_tracker(self):
        test_tracker = Tracker()
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))

        assert len(test_tracker.objects) == 2

    def test_update_equal_objects(self):
        test_tracker = Tracker()
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        test_tracker.update(np.array([[3.1, 4.1], [1.1, 2.1]]))

        assert len(test_tracker.objects) == 2
        npt.assert_array_equal(
            test_tracker.objects[0].track, np.array([[1.0, 2.0, 0], [1.1, 2.1, 1]])
        )
        npt.assert_array_equal(
            test_tracker.objects[1].track, np.array([[3.0, 4.0, 0], [3.1, 4.1, 1]])
        )

    def test_update_extra_objects(self):
        test_tracker = Tracker()
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        test_tracker.update(np.array([[5.0, 6.0], [1.1, 2.1], [3.1, 4.1]]))

        assert len(test_tracker.objects) == 3
        assert test_tracker.objects[0].track.shape[0] == 2
        assert test_tracker.objects[1].track.shape[0] == 2
        assert test_tracker.objects[2].track.shape[0] == 1
        npt.assert_array_equal(
            test_tracker.objects[2].position, np.array([5.0, 6.0, 1])
        )

    def test_update_missing_objects(self):
        test_tracker = Tracker()
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        test_tracker.update(np.array([[1.1, 2.1]]))

        assert len(test_tracker.objects) == 2
        assert test_tracker.objects[0].track.shape[0] == 2
        assert test_tracker.disappeared[0] == 0
        assert test_tracker.disappeared[1] == 1

    def test_update_reappeard_objects(self):
        test_tracker = Tracker()
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        test_tracker.update(np.array([[1.1, 2.1]]))
        test_tracker.update(np.array([[3.1, 4.1], [1.2, 2.2]]))

        assert len(test_tracker.objects) == 2
        assert test_tracker.objects[0].track.shape[0] == 3
        assert test_tracker.objects[1].track.shape[0] == 2
        assert test_tracker.disappeared[0] == 0
        assert test_tracker.disappeared[1] == 0

    def test_update_deregister_objects(self):
        test_tracker = Tracker(max_disappeared=1)
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        test_tracker.update(np.array([[1.1, 2.1]]))
        test_tracker.update(np.array([[1.2, 2.2]]))

        assert len(test_tracker.objects) == 1
        npt.assert_array_equal(
            test_tracker.objects[0].track,
            np.array([[1.0, 2.0, 0], [1.1, 2.1, 1], [1.2, 2.2, 2]]),
        )

        test_tracker = Tracker(max_disappeared=1)
        test_tracker.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        test_tracker.update(np.array([[]]))
        test_tracker.update(np.array([[]]))
        assert len(test_tracker.objects) == 0
