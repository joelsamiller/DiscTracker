import os

import numpy as np
import numpy.typing as npt
import yaml
from numpy.lib.npyio import NpzFile


class DiscTrack:
    """
    Class for storing the disc track coordinates and deprojecting them to 3D coordinates.
    """    
    def __init__(self, directory: str) -> None:
        """
        Initialise class.

        Args:
            directory (str): Path to directory containing `tracks` sub-directory.
        """        
        self.directory = directory
        self.read_camera_settings()
        self.read_tracks()

    def deproject(self) -> tuple[npt.NDArray[np.float64]]:
        """
        Deproject the 2D coordinates from each camera into 3D space.

        Returns:
            tuple[NDArray[float64]]: X, Y and Z coordinates of the disc in 3D space.
        """        
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
        """
        Load the tracks data from file.
        """        
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
        """
        Read in the camera settings from file.
        """        
        with open(os.path.join(self.directory, "camera_settings.yaml")) as file:
            self.camera_settings = yaml.safe_load(file)
        self.camera_settings["f"] = (
            self.camera_settings["focal_length"]
            * self.camera_settings["resolution"][0]
            / self.camera_settings["sensor_width"]
        )

    @staticmethod
    def complete_tracks(data: NpzFile, time_index: npt.NDArray[np.int64]) -> dict[str, npt.NDArray[np.int64]]:
        """
        Interpolate coordinates for missing frames and trim to coeval time period.

        Args:
            data (NpzFile): Input file data.
            time_index (NDArray[int64]): Time values to interpolate for.

        Returns:
            dict[str, NDArray[int64]]: Cleaned coordinate tracks for each axis.
        """        
        return {
            axis: np.interp(time_index, data["t"], data[axis]).astype(np.int64)
            for axis in "xy"
        }

    @staticmethod
    def get_time_index(
        left: npt.NDArray[np.int64], right: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """
        Generate an array giving the coeval time coordinates for both tracks.

        Args:
            left (NDArray[int64]): Time values for the left camera.
            right (NDArray[int64]): Time values for the right camera.

        Returns:
            NDArray[int64]: Coveal time values.
        
        Examples:
        >>> DiscTrack.get_time_index(np.array[2, 3, 4, 6, 7], np.array[1, 2, 3, 5, 8])
        array([2, 3, 4, 5, 6, 7])
        """        
        t_min = max(min(left), min(right))
        t_max = min(max(left), max(right))
        return np.arange(t_min, t_max + 1)
