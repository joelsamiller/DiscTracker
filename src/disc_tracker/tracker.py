from scipy.spatial import distance
from collections import OrderedDict
import numpy as np


class Tracker:
    def __init__(self, max_disappeared=5000):
        self.nextID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, position: np.ndarray[float]):
        self.objects[self.nextID] = [position]
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, ID):
        del self.objects[ID]
        del self.disappeared[ID]

    def update(self, new_position: np.ndarray[float]):
        if len(new_position) == 0:
            for id in self.disappeared:
                self.disappeared[id] += 1

                if self.disappeared[id] > self.max_disappeared:
                    self.deregister(id)

            return self.objects

        if len(self.objects) == 0:
            for p in new_position:
                self.register(p)
        else:
            ids = [*self.objects]
            old_position = np.array([p[-1] for p in self.objects.values()])

            r = distance.cdist(old_position, new_position)

            rows = r.min(axis=1).argsort()
            cols = r.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                id = ids[row]
                self.objects[id].append(new_position[col])
                self.disappeared[id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, r.shape[0])).difference(used_rows)
            unused_cols = set(range(0, r.shape[1])).difference(used_cols)

            if r.shape[0] >= r.shape[1]:
                for row in unused_rows:
                    id = ids[row]
                    self.disappeared[id] += 1

                    if self.disappeared[id] > self.max_disappeared:
                        self.deregister(id)
            else:
                for col in unused_cols:
                    self.register(new_position[col])

        return self.objects
