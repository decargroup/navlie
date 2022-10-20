from typing import List
import numpy as np
from pynav.lib.imu import IMUState, IMU
from pynav.utils import randvec


def generate_landmark_positions(
    cylinder_radius: float,
    max_height: float,
    n_levels: int,
    n_landmarks_per_level: int,
) -> List[np.ndarray]:
    """Generates landmarks arranged in a cylinder.

    Parameters
    ----------
    cylinder_radius : float
        Radius of the cylinder that the landmarks are arranged in.
    max_height : float
        Top of cylinder.
    n_levels : int
        Number of discrete levels to place landmarks at vertically.
    n_landmarks_per_level : int
        Number of landmarks per level

    Returns
    -------
    List[np.ndarray]
        List of landmarks.
    """

    z = np.linspace(0, max_height, n_levels)

    angles = np.linspace(0, 2 * np.pi, n_landmarks_per_level + 1)
    angles = angles[0:-1]
    x = cylinder_radius * np.cos(angles)
    y = cylinder_radius * np.sin(angles)

    # Generate landmarks
    landmarks = []
    for level_idx in range(n_levels):
        for landmark_idx in range(n_landmarks_per_level):
            cur_landmark = np.array(
                [x[landmark_idx], y[landmark_idx], z[level_idx]]
            ).reshape((3, -1))
            landmarks.append(cur_landmark)

    return landmarks
