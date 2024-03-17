import numpy as np

from scipy.spatial import distance
from scipy import optimize
from collections import OrderedDict


class Tracker:
    def __init__(self, max_disappeared=5000):
        self.next_id = 0
        self.current_time = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, position: np.ndarray[float]):
        self.objects[self.next_id] = {
            "position": position[None, :],
            "time": self.current_time,
        }  # Ensure array is 2D
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, id):
        del self.objects[id]
        del self.disappeared[id]

    def update(self, new_position: np.ndarray[float]):
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
            old_position = np.vstack([p["position"][-1] for p in self.objects.values()])
            # Jonker-Volgenant assignment using distance from last position as cost matrix
            row, col = optimize.linear_sum_assignment(
                distance.cdist(old_position, new_position)
            )

            for r, c in zip(row, col):
                id = ids[r]
                self.objects[id]["position"] = np.vstack(
                    [self.objects[id]["position"], new_position[c]]
                )
                self.objects[id]["time"] = np.append(
                    self.objects[id]["time"], self.current_time
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
