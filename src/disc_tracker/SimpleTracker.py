from re import S
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class SimpleDistanceTracker:
    def __init__(self, maxDisappeared=5000):
        self.nextID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, pos):
        self.objects[self.nextID] = [pos]
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, ID):
        del self.objects[ID]
        del self.disappeared[ID]

    def update(self, newPos):
        if len(newPos) == 0:
            for ID in list(self.disappeared.keys()):
                self.disappeared[ID] += 1

                if self.disappeared[ID] > self.maxDisappeared:
                    self.deregister(ID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(newPos)):
                self.register(newPos[i])
        else:
            IDs = list(self.objects.keys())
            oldPos = [positions[-1] for positions in list(self.objects.values())]

            r = dist.cdist(np.array(oldPos), newPos)

            rows = r.min(axis=1).argsort()
            cols = r.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                ID = IDs[row]
                self.objects[ID].append(newPos[col])
                self.disappeared[ID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, r.shape[0])).difference(usedRows)
            unusedCols = set(range(0, r.shape[1])).difference(usedCols)

            if r.shape[0] >= r.shape[1]:
                for row in unusedRows:
                    ID = IDs[row]
                    self.disappeared[ID] += 1

                    if self.disappeared[ID] > self.maxDisappeared:
                        self.deregister(ID)
            else:
                for col in unusedCols:
                    self.register(newPos[col])
        return self.objects
