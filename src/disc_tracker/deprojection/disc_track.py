import os

import numpy as np
import yaml
from numpy.lib.npyio import NpzFile


class DiscTrack:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.read_camera_settings()
        self.read_tracks()

    def deproject(self) -> tuple:
        x = 0.5 * self.camera_settings["d"] * (self.xl + self.xr) / (self.xl - self.xr)
        z = (
            -0.5 * self.camera_settings["d"] * (self.zl + self.zr) / (self.xl - self.xr)
            + self.camera_settings["h"]
        )
        y = (
            self.camera_settings["d"] * self.camera_settings["f"] / (self.xl - self.xr)
            - self.camera_settings["c"]
        )
        return (x, y, z)

    def read_tracks(self) -> None:
        tracks_directory = os.path.join(self.directory, "tracks")
        L = np.load(os.path.join(tracks_directory, "left.npz"))
        R = np.load(os.path.join(tracks_directory, "right.npz"))
        time_index = self.get_time_index(L["t"], R["t"])
        right = self.complete_tracks(R, time_index)
        left = self.complete_tracks(L, time_index)
        # Centre coordinates on 0
        for track in [left, right]:
            for i, axis in enumerate("xy"):
                track[axis] -= int(self.camera_settings["resolution"][i] / 2)

        self.xl = left["x"]
        self.xr = right["x"]
        self.zl = left["y"]
        self.zr = right["y"]

    def read_camera_settings(self) -> None:
        with open(os.path.join(self.directory, "camera_settings.yaml")) as file:
            self.camera_settings = yaml.safe_load(file)
        self.camera_settings["f"] = (
            self.camera_settings["focal_length"]
            * self.camera_settings["resolution"][0]
            / self.camera_settings["sensor_width"]
        )

    @staticmethod
    def complete_tracks(data: NpzFile, time_index: np.ndarray[int]) -> dict:
        return {
            axis: np.interp(time_index, data["t"], data[axis]).astype(int)
            for axis in "xy"
        }

    @staticmethod
    def get_time_index(
        left: np.ndarray[int], right: np.ndarray[int]
    ) -> np.ndarray[int]:
        t_min = max(min(left), min(right))
        t_max = min(max(left), max(right))
        return np.arange(t_min, t_max + 1)
