import navlie as nav
from navlie.lib import VectorState, VectorInput
import numpy as np

# Define the initial state
x0 = VectorState([0, 0, 0], stamp=0.0)
Q = np.eye(2) * 0.1**2


# Define the process model
class BicycleModel(nav.ProcessModel):
    def evaluate(self, x: VectorState, u: nav.VectorInput, dt: float) -> VectorState:
        x_next = x.copy()
        x_next.value[0] += u.value[0] * dt * np.cos(x.value[2])
        x_next.value[1] += u.value[0] * dt * np.sin(x.value[2])
        x_next.value[2] += u.value[1] * dt
        return x_next

    def input_covariance(self, x: VectorState, u: VectorInput, dt: float) -> np.ndarray:
        return Q


# Define the measurement model
class RangeToLandmark(nav.MeasurementModel):
    def __init__(self, landmark_position: np.ndarray):
        self.landmark_position = landmark_position

    def evaluate(self, x: VectorState) -> np.ndarray:
        return np.linalg.norm(x.value[:2] - self.landmark_position)

    def covariance(self, x: VectorState) -> np.ndarray:
        return 0.1**2


# Generate some simulated data
landmarks = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
meas_models = [RangeToLandmark(landmark) for landmark in landmarks]
process_model = BicycleModel()
dg = nav.DataGenerator(
    process_model=process_model,
    input_func=lambda t, x: np.array([0.5, 0.3]),
    input_covariance=lambda t: Q,
    input_freq=50,
    meas_model_list=meas_models,
    meas_freq_list=[10, 10, 10, 10],
)

state_data, input_data, meas_data = dg.generate(x0, start=0, stop=10, noise=True)


# Now lets run a filter!
# First, define the filter
kalman_filter = nav.ExtendedKalmanFilter(process_model)
P0 = np.diag([1, 1, 0.1**2])  # Initial covariance
x = nav.StateWithCovariance(x0, P0)  # Estimate and covariance in one container
meas_idx = 0
y = meas_data[meas_idx]
estimates = []
for k in range(len(input_data) - 1):
    u = input_data[k]

    # Fuse any measurements that have occurred.
    while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):
        x = kalman_filter.correct(x, y, u)

        # Load the next measurement
        meas_idx += 1
        if meas_idx < len(meas_data):
            y = meas_data[meas_idx]

    # Predict until the next input is available
    dt = input_data[k + 1].stamp - x.state.stamp
    x = kalman_filter.predict(x, u, dt)

    estimates.append(x.copy())


# Plot the results
import matplotlib.pyplot as plt

pos = np.array([state.value[:2] for state in state_data])
plt.plot(pos[:, 0], pos[:, 1])
plt.scatter(landmarks[:, 0], landmarks[:, 1])
# add labels
for i, landmark in enumerate(landmarks):
    plt.annotate(f"Landmark {i}", landmark)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simulated Trajectory")
plt.axis("equal")

plt.show()
