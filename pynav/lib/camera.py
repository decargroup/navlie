"""Module containing a basic pinhole camera model."""

import numpy as np
from pynav.lib import SE3State


class CameraModel:
    """Class for a pinhole camera model.

    This class contains utilities for generating measurements
    and predicting measurements, and converting pixel coordinates to
    normalized image coordinates using the camera intrinsic parameters.
    """

    def __init__(
        self,
        fu: float,
        fv: float,
        cu: float,
        cv: float,
        image_width: int,
        image_height: int,
        sigma: float,
        T_bc: SE3State = None,
        freq: int = None,
        id: int = None,
    ):
        """Instantiate a CameraModel

        Parameters
        ----------
        fu : float
            Focal length, horizontal pixels.
        fv : float
            Focal length, vertical pixels.
        cu : float
            Optical axis intersection, horizontal pixels.
        cv : float
            Optical axis intersection, vertical pixels.
        sigma : float
            _description_
        image_width : int
            _description_
        image_height : int
            _description_
        T_bc : SE3State, optional
            Transformation between the body frame
            and the camera frame, by default None.
            If None, transformation will be set to
            identity.
        freq : int, optional
            _description_, by default None
        id : int, optional
            _description_, by default None
        """
        # Camera intrinsic parameters
        self.fu = float(fu)
        self.fv = float(fv)
        self.cu = float(cu)
        self.cv = float(cv)

        # Camera extrinsic parameters, an element of SE(3)
        if T_bc is None:
            # If not specified, T_bc is the identity element.
            T_bc = SE3State(np.identity(4))

        self.T_bc = T_bc

        # Image resolution
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        # Noise parameter
        self.sigma = sigma

        # Camera frequency
        self.freq = freq
        self.id = id

    @staticmethod
    def get_enu_to_cam() -> np.ndarray:
        """Returns a DCM that relates the "ENU frame to the camera frame,
        where the camera z-axis points forward.
        """
        return np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    def get_camera_intrinsics(self) -> np.ndarray:
        """Returns the intrinsic matrix K."""
        return np.array(
            [
                [self.fu, 0, self.cu],
                [0, self.fv, self.cv],
                [0, 0, 1],
            ]
        )

    def copy(self) -> "CameraModel":
        """Returns a copy of the camera model."""
        return (
            CameraModel(
                self.fu,
                self.fv,
                self.cu,
                self.cv,
                self.sigma,
                self.image_width,
                self.image_height,
                self.T_bc.copy(),
                self.freq,
                self.id,
            ),
        )

    def is_measurement_valid(self, uv: np.ndarray) -> bool:
        """Checks if a measurement is valid.

        A valid measurement should be positive and within the image plane.

        Parameters
        ----------
        uv : np.ndarray
            Measurement to check.

        Returns
        -------
        bool
            Is valid.
        """
        uv = uv.ravel()
        return (
            (uv[1] > 0.0)
            and (uv[1] < self.image_height)
            and (uv[0] > 0.0)
            and (uv[0] < self.image_width)
        )

    def is_landmark_in_front_of_cam(
        self, pose: SE3State, r_pw_a: np.ndarray
    ) -> bool:
        """Checks if a given landmark is in front of the camera."""
        r_pc_c: np.ndarray = self.resolve_landmark_in_cam_frame(pose, r_pw_a)
        r_pc_c = r_pc_c.ravel()

        return r_pc_c[2] > 0.0

    def evaluate(self, pose: SE3State, r_pw_a: np.ndarray) -> np.ndarray:
        """Predicts a noise-free measurement.

        Parameters
        ----------
            X : IMUState
                IMU state to generate measurement from
            r_pw_a : np.ndarray
                3D landmark position, resolved in the world frame.

        Returns
        -------
        np.ndarray
            noise-free pixel measurement
        """

        r_pc_c = self.resolve_landmark_in_cam_frame(pose, r_pw_a)
        y_check = self.project(r_pc_c)

        return y_check

    def resolve_landmark_in_cam_frame(
        self,
        pose: SE3State,
        r_pw_a: np.ndarray,
    ):
        """Resolves a landmark with position r_pw_a in the camera frame."""

        r_pw_a = r_pw_a.reshape((-1, 1))
        C_bc = self.T_bc.attitude
        C_ab = pose.attitude
        r_zw_a = pose.position.reshape((-1, 1))
        r_cz_b = self.T_bc.position.reshape((-1, 1))

        r_pc_c = C_bc.T @ (C_ab.T @ (r_pw_a - r_zw_a) - r_cz_b)
        return r_pc_c

    def project(self, r_pc_c: np.ndarray) -> np.ndarray:
        """Pinhole projection model.

        Parameters
        ----------
        r_pc_c : np.ndarray
            Landmark relative to camera, resolved in camera frame.

        Returns
        -------
        np.ndarray
            (u, v) pixel coordinates.
        """
        x, y, z = r_pc_c.ravel()

        u = self.fu * x / z + self.cu
        v = self.fv * y / z + self.cv

        return np.array([u, v])

    def to_normalized_coords(self, uv: np.ndarray) -> np.ndarray:
        """Converts (u, v) pixel coordinates to
        normalized image coordinates.

        Parameters
        ----------
        uv : np.ndarray
            Pixel coordinates

        Returns
        -------
        np.ndarray
            Normalized image coordinates, computed using intrinsics.
        """

        u, v = uv.ravel()
        x_n = (u - self.cu) / self.fu
        y_n = (v - self.cv) / self.fv

        return np.array([x_n, y_n, 1])
