from navlie.lib.camera import Camera, PoseMatrix
import numpy as np
from pymlg import SE3


def test_valid_measurements():
    # Create camera with its z-axis pointing forward
    C_bc = Camera.get_cam_to_enu()
    T_bc = PoseMatrix(SE3.from_components(C_bc, np.array([0, 0, 0])))
    camera = Camera(385, 385, 323, 236, 480, 640, 0.1, T_bc)

    # Create some invalid measurements
    invalid_meas = [
        np.array([-50, 50]),
        np.array([50, -50]),
        np.array([500, 50]),
        np.array([50, 700]),
    ]

    # Check if measurements are valid
    valid = []
    for meas in invalid_meas:
        valid.append(camera.is_measurement_valid(meas))

    assert not any(valid)

    # Create some valid measurements of landmarks from one pose
    landmarks = [
        np.array([1.0, 0.25, 0.25]),
        np.array([1.0, 0.1, 0.1]),
        np.array([1.0, 0.4, 0.4]),
    ]

    pose = PoseMatrix(np.identity(4))

    # Check if landmarks are in front of camera
    in_front = []
    for landmark in landmarks:
        in_front.append(camera.is_landmark_in_front_of_cam(pose, landmark))

    assert all(in_front)

    # Create measurements
    valid_meas = [camera.evaluate(pose, landmark) for landmark in landmarks]

    is_valid = []
    for meas in valid_meas:
        is_valid.append(camera.is_measurement_valid(meas))

    assert all(is_valid)


def test_project_function():
    camera = Camera(385, 385, 323, 236, 480, 640, 0.1)
    # Define a landmark resolved in the camera frame
    r_pc_c = np.random.randn(3, 1)
    # Project onto image plane
    uv = camera.project(r_pc_c)
    # Compute normalized image coordinates and compare to camera function
    x, y, z = r_pc_c.ravel()
    normalized_image_coords = np.array([x / z, y / z, 1])
    p = camera.to_normalized_coords(uv)

    assert np.allclose(normalized_image_coords, p)


def test_camera_intrinsics():
    fu = 385
    fv = 385
    cu = 323
    cv = 236
    camera = Camera(fu, fv, cu, cv, 480, 640, 0.1)
    K = camera.intrinsics

    assert K[0, 0] == fu
    assert K[0, 2] == cu
    assert K[1, 1] == fv
    assert K[1, 2] == cv


if __name__ == "__main__":
    test_camera_intrinsics()
