import numpy as np
import numpy.typing as npt

from collections import OrderedDict
from scipy.spatial import distance
from scipy import optimize


class Object:
    def __init__(
        self, creation_time: np.int64, position: npt.NDArray[np.int64]
    ) -> None:
        """
        Initialise class.

        Args:
            creation_time (int64): Time object id first used (frame number).
            position (NDArray[int64]): Coordinates of object.
        """
        self.position = np.append(position, creation_time)

        self.track = self.position[None, :]

    def update_position(self, time: np.int64, position: npt.NDArray[np.int64]) -> None:
        """
        Update the position and track of the object with new coordinates.

        Args:
            time (int64): Time the update occures (frame number).
            position (npt.NDArray[np.int64]): Coordinates of the object.
        """
        self.position = np.append(position, time)
        self.track = np.vstack([self.track, self.position])


class Tracker:
    def __init__(self, max_disappeared=5000):
        """
        Initialise class.

        Args:
            max_disappeared (int, optional): Delete object after this many frames with no detection. Defaults to 5000.
        """
        self.next_id = 0
        self.current_time = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, position: npt.NDArray[np.int64]):
        """
        Register a new object.

        Args:
            position (NDArray[int64]): Coordinates of the object.
        """
        self.objects[self.next_id] = Object(
            creation_time=self.current_time, position=position
        )
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, id: np.int64):
        """
        Remove an object.

        Args:
            id (int64): ID of object to remove.
        """
        del self.objects[id]
        del self.disappeared[id]

    def update(
        self, new_position: npt.NDArray[np.int64]
    ) -> OrderedDict[np.int64, Object]:
        """
        Update the positions of existing objects and register any new objects.

        Args:
            new_position (NDArray[int64]): Coordinates of all objects detected in the frame.

        Returns:
            OrderedDict[int64, Object]: Dictionary of all tracked objects.
        """
        if new_position.size == 0:
            for id in self.disappeared:
                self.disappeared[id] += 1

                if self.disappeared[id] > self.max_disappeared:
                    self.deregister(id)
            self.current_time += 1
            return self.objects

        if len(self.objects) == 0:
            for p in new_position:
                self.register(p)
        else:
            ids = np.array([*self.objects])
            old_position = np.vstack([o.position[0:2] for o in self.objects.values()])
            # Jonker-Volgenant assignment using distance from old position as cost matrix
            row, col = optimize.linear_sum_assignment(
                distance.cdist(old_position, new_position)
            )

            for r, c in zip(row, col):
                id = ids[r]
                self.objects[id].update_position(
                    time=self.current_time, position=new_position[c]
                )
                self.disappeared[id] = 0

            ids = np.delete(ids, row)
            if ids.size > 0:
                for id in ids:
                    self.disappeared[id] += 1
                    if self.disappeared[id] > self.max_disappeared:
                        self.deregister(id)
            else:
                unassigned_positions = np.delete(new_position, col, axis=0)
                for position in unassigned_positions:
                    self.register(position)

        self.current_time += 1
        return self.objects
