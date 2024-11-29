""" 
Collection of miscellaneous plotting functions.
"""

from typing import List, Tuple, Dict
from navlie.types import State, Measurement, StateWithCovariance
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from navlie.utils.common import GaussianResultList

def plot_error(
    results: GaussianResultList,
    axs: List[plt.Axes] = None,
    label: str = None,
    sharey: bool = False,
    color=None,
    bounds=True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    A generic three-sigma bound plotter.

    Parameters
    ----------
    results : GaussianResultList
        Contains the data to plot
    axs : List[plt.Axes], optional
        Axes to draw on, by default None. If None, new axes will be created.
    label : str, optional
        Text label to add, by default None
    sharey : bool, optional
        Whether to have a common y axis or not, by default False
    color : color, optional
        specify the color of the error/bounds.

    Returns
    -------
    plt.Figure
        Handle to figure.
    List[plt.Axes]
        Handle to axes that were drawn on.
    """

    dim = results.error.shape[1]

    if dim < 3:
        n_rows = dim
    else:
        n_rows = 3

    n_cols = int(np.ceil(dim / 3))

    if axs is None:
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=sharey)
    else:
        fig: plt.Figure = axs.ravel("F")[0].get_figure()

    axs_og = axs
    kwargs = {}
    if color is not None:
        kwargs["color"] = color

    axs: List[plt.Axes] = axs.ravel("F")
    for i in range(results.three_sigma.shape[1]):
        if bounds:
            axs[i].fill_between(
                results.stamp,
                results.three_sigma[:, i],
                -results.three_sigma[:, i],
                alpha=0.5,
                **kwargs,
            )
        axs[i].plot(results.stamp, results.error[:, i], label=label, **kwargs)

    fig: plt.Figure = fig  # For type hinting
    return fig, axs_og


def plot_nees(
    results: GaussianResultList,
    ax: plt.Axes = None,
    label: str = None,
    color=None,
    confidence_interval: float = 0.95,
    normalize: bool = False,
    expected_nees_color="r",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Makes a plot of the NEES, showing the actual NEES values, the expected NEES,
    and the bounds of the specified confidence interval.

    Parameters
    ----------
    results : GaussianResultList or MonteCarloResult
        Results to plot
    ax : plt.Axes, optional
        Axes on which to draw, by default None. If None, new axes will be
        created.
    label : str, optional
        Label to assign to the NEES line, by default None
    color : optional
        Fed directly to the ``plot(..., color=color)`` function, by default None
    confidence_interval : float or None, optional
        Desired probability confidence region, by default 0.95. Must lie between
        0 and 1. If None, no confidence interval will be plotted.
    normalize : bool, optional
        Whether to normalize the NEES by the degrees of freedom, by default False

    Returns
    -------
    plt.Figure
        Figure on which the plot was drawn
    plt.Axes
        Axes on which the plot was drawn
    """

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
        )
    else:
        fig = ax.get_figure()

    axs_og = ax
    kwargs = {}
    if color is not None:
        kwargs["color"] = color

    if normalize:
        s = results.dof
    else:
        s = 1

    expected_nees_label = "Expected NEES"
    _, exisiting_labels = ax.get_legend_handles_labels()

    if expected_nees_label in exisiting_labels:
        expected_nees_label = None

    # fmt:off
    ax.plot(results.stamp, results.nees/s, label=label, **kwargs)
    if confidence_interval:
        ci_label = f"${int(confidence_interval*100)}\%$ conf. bounds"
        if ci_label in exisiting_labels:
            ci_label = None
        ax.plot(results.stamp, results.dof/s, label=expected_nees_label, color=expected_nees_color)
        ax.plot(results.stamp, results.nees_upper_bound(confidence_interval)/s, "--", color="k", label=ci_label)
        ax.plot(results.stamp, results.nees_lower_bound(confidence_interval)/s, "--", color="k")
    # fmt:on

    ax.legend()

    return fig, axs_og


def plot_meas(
    meas_list: List[Measurement],
    state_list: List[State],
    axs: List[plt.Axes] = None,
    sharey=False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Given measurement data, make time-domain plots of the measurement values
    and their ground-truth model-based values.

    Parameters
    ----------
    meas_list : List[Measurement]
        Measurement data to be plotted.
    state_list : List[State]
        A list of true State objects with similar timestamp domain. Will be
        interpolated if timestamps do not line up perfectly.
    axs : List[plt.Axes], optional
        Axes to draw on, by default None. If None, new axes will be created.
    sharey : bool, optional
        Whether to have a common y axis or not, by default False

    Returns
    -------
    plt.Figure
        Handle to figure.
    List[plt.Axes]
        Handle to axes that were drawn on.
    """

    # Convert everything to numpy arrays for plotting, and compute the
    # ground-truth model-based measurement value.

    meas_list.sort(key=lambda y: y.stamp)

    # Find the state of the nearest timestamp to the measurement
    y_stamps = np.array([y.stamp for y in meas_list])
    x_stamps = np.array([x.stamp for x in state_list])
    indexes = np.array(range(len(state_list)))
    nearest_state = interp1d(
        x_stamps,
        indexes,
        "nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    state_idx = nearest_state(y_stamps)
    y_meas = []
    y_true = []
    three_sigma = []
    for i in range(len(meas_list)):
        data = np.ravel(meas_list[i].value)
        y_meas.append(data)
        x = state_list[int(state_idx[i])]
        y = meas_list[i].model.evaluate(x)
        if y is None:
            y_true.append(np.zeros_like(data) * np.nan)
            three_sigma.append(np.zeros_like(data) * np.nan)
        else:
            y_true.append(np.ravel(y))
            R = np.atleast_2d(meas_list[i].model.covariance(x))
            three_sigma.append(3 * np.sqrt(np.diag(R)))

    y_meas = np.atleast_2d(np.array(y_meas))
    y_true = np.atleast_2d(np.array(y_true))
    three_sigma = np.atleast_2d(np.array(three_sigma))
    y_stamps = np.array(y_stamps)  # why is this necessary?
    x_stamps = np.array(x_stamps)

    # Plot
    size_y = np.size(meas_list[0].value)
    if axs is None:
        fig, axs = plt.subplots(size_y, 1, sharex=True, sharey=sharey)
    else:
        if isinstance(axs, plt.Axes):
            axs = np.array([axs])

        fig = axs.ravel("F")[0].get_figure()
    axs = np.atleast_1d(axs)
    for i in range(size_y):
        axs[i].scatter(
            y_stamps, y_meas[:, i], color="b", alpha=0.7, s=2, label="Measured"
        )
        axs[i].plot(
            y_stamps, y_true[:, i], color="r", alpha=1, label="Modelled"
        )
        axs[i].fill_between(
            y_stamps,
            y_true[:, i] + three_sigma[:, i],
            y_true[:, i] - three_sigma[:, i],
            alpha=0.5,
            color="r",
        )
    return fig, axs


def plot_meas_by_model(
    meas_list: List[Measurement],
    state_list: List[State],
    axs: List[plt.Axes] = None,
    sharey=False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Given measurement data, make time-domain plots of the measurement values
    and their ground-truth model-based values.

    Parameters
    ----------
    meas_list : List[Measurement]
        Measurement data to be plotted.
    state_list : List[State]
        A list of true State objects with similar timestamp domain. Will be
        interpolated if timestamps do not line up perfectly.
    axs : List[plt.Axes], optional
        Axes to draw on, by default None. If None, new axes will be created.
    sharey : bool, optional
        Whether to have a common y axis or not, by default False

    Returns
    -------
    plt.Figure
        Handle to figure.
    List[plt.Axes]
        Handle to axes that were drawn on.
    """

    # Create sub-lists for every model ID
    meas_by_model: Dict[int, List[Measurement]] = {}
    for meas in meas_list:
        model_id = id(meas.model)
        if model_id not in meas_by_model:
            meas_by_model[model_id] = []
        meas_by_model[model_id].append(meas)

    if axs is None:
        axs = np.array([None] * len(meas_by_model))

    axs = axs.ravel("F")

    figs = [None] * len(meas_by_model)
    for i, temp in enumerate(meas_by_model.items()):
        model_id, meas_list = temp
        fig, ax = plot_meas(meas_list, state_list, axs[i], sharey=sharey)
        figs[i] = fig
        axs[i] = ax
        ax[0].set_title(f"{meas_list[0].model} {hex(model_id)}", fontsize=12)
        ax[0].tick_params(axis="both", which="major", labelsize=10)
        ax[0].tick_params(axis="both", which="minor", labelsize=8)

    return fig, axs


def plot_camera_poses(
    poses,
    ax: plt.Axes = None,
    color: str = "tab:blue",
    line_thickness: float = 1,
    step: int = 1,
    scale: float = 0.25,
):
    """
    Plots camera poses along a 3D plot.

    The camera poses should be elements of SE(3), with the z-axis pointing
    forward through the optical axis.

    Parameters
    ----------
    poses : List[SE3State]
        A list objects containing a ``position`` property and an attitude
        property, representing the rotation matrix :math:``\mathbf{C}_{ab}``.
    ax : plt.Axes, optional
        Axes to plot on, if none, 3D axes are created.
    color : str, optional
        Color of the plotted camera, by default "tab:blue"
    line_thickness : float, optional
        Thickness of the camera line, by default 1
    step : int, optional
        Step size in number of poses to plot, by default 1 which plots all poses
    scale : float, optional
        Scale of the camera, by default 0.25
    Returns
    -------
    plt.Figure
        Handle to figure.
    List[plt.Axes]
        Handle to axes that were drawn on.
    """
    if isinstance(poses, GaussianResultList):
        poses = poses.state

    if isinstance(poses, StateWithCovariance):
        poses = [poses.state]

    if isinstance(poses, np.ndarray):
        poses = poses.tolist()

    if not isinstance(poses, list):
        poses = [poses]

    # Check if provided axes are in 3D
    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
    else:
        fig = ax.get_figure()

    # Plot the individual camera poses
    cam_pose_viz = CameraPoseVisualizer(
        line_thickness=line_thickness,
        scale=scale,
        color=color,
    )
    for i in range(0, len(poses), step):
        fig, ax = cam_pose_viz.plot_pose(
            poses[i].attitude, poses[i].position, ax=ax
        )

    set_axes_equal(ax)
    return fig, ax


class CameraPoseVisualizer:
    """A class to plot camera poses in 3D using matplotlib."""
    def __init__(
        self,
        line_thickness: float = 1,
        scale: float = 0.25,
        color: str = "tab:blue",
    ):
        self.line_thickness = line_thickness
        self.scale = scale
        self.color = color

        # Define points resolved in the camera frame
        cam_points: List[np.ndarray] = []
        # Image plane corners
        cam_points.append(np.array([-1.0, -1.0, 1.0]))  # left top
        cam_points.append(np.array([1.0, -1.0, 1.0]))  # right top
        cam_points.append(np.array([-1.0, 1.0, 1.0]))  # left bottom
        cam_points.append(np.array([1.0, 1.0, 1.0]))  # right bottom
        # Optical center
        cam_points.append(np.array([0.0, 0.0, 0.0]))
        for point in cam_points:
            point *= scale
        self.cam_points = cam_points

    def plot_pose(self, C: np.ndarray, r: np.ndarray, ax: plt.Axes = None):
        """Plots a camera pose in 3D.

        Plots lines representing connection between the optical center and the
        camera corners.

        Parameters
        ----------
        C : np.ndarray
            Rotation matrix representing the camera attitude :math:``\mathbf{C}_{ab}``.
        r : np.ndarray
            Position of the camera in the inertial frame.

        Returns
        -------
        plt.Figure
            Handle to figure.
        plt.Axes
            Handle to axes that were drawn on.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        r = r.ravel()
        # Resolve the points in the inertial frame
        cam_points = self.cam_points
        inertial_points: List[np.ndarray] = []
        for point in cam_points:
            inertial_points.append(C @ point + r)

        # Define the connections between the points
        connections = [
            (0, 1),
            (2, 3),
            (0, 2),
            (1, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ]

        # Plot lines between each point defined in the connections list
        for connection in connections:
            p1 = inertial_points[connection[0]]
            p2 = inertial_points[connection[1]]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                linewidth=self.line_thickness,
                color=self.color,
            )
        return fig, ax


def plot_poses(
    poses,
    ax: plt.Axes = None,
    line_color: str = None,
    triad_color: str = None,
    arrow_length: float = 1,
    step: int = 5,
    label: str = None,
    linewidth=None,
    plot_2d: bool =False,
):
    """
    Plots a pose trajectory, representing the attitudes by triads
    plotted along the trajectory.

    The poses may be either elements of SE(2), 
    representing planar 2D poses, or elements of SE(3), representing 3D poses.

    Parameters
    ----------
    poses : List[Union[SE2State, SE3State]]
        A list objects containing a ``position`` property and an attitude
        property, representing the rotation matrix :math:``\mathbf{C}_{ab}``.
        Can either be 2D or 3D poses.
    ax : plt.Axes, optional
        Axes to plot on, if none, 3D axes are created.
    line_color : str, optional
        Color of the position trajectory.
    triad_color : str, optional
        Triad color. If none are specified, defaults to RGB.
    arrow_length : int, optional
        Triad arrow length, by default 1.
    step : int or None, optional
        Step size in list of poses, by default 5. If None, no triads are plotted.
    label : str, optional
        Optional label for the triad
    plot_2d: bool, optional
        Flag to plot a 3D pose trajectory in 2D bird's eye view.
    """
    if isinstance(poses, GaussianResultList):
        poses = poses.state

    if isinstance(poses, StateWithCovariance):
        poses = [poses.state]

    if isinstance(poses, np.ndarray):
        poses = poses.tolist()

    if not isinstance(poses, list):
        poses = [poses]

    # Check if poses are in 2D or 3D
    if poses[0].position.size == 2:
        plot_2d = True

    # Check if provided axes are in 3D
    if ax is not None:
        if ax.name == "3d":
            plot_2d = False

    if ax is None:
        fig = plt.figure()
        if plot_2d:
            ax = plt.axes()
        else:
            ax = plt.axes(projection="3d")
    else:
        fig = ax.get_figure()

    if triad_color is None:
        colors = ["tab:red", "tab:green", "tab:blue"]  # Default to RGB
    else:
        colors = [triad_color] * 3

    # Plot a line for the positions
    r = np.array([pose.position for pose in poses])
    if plot_2d:
        ax.plot(r[:, 0], r[:, 1], color=line_color, label=label)
    else:
        ax.plot3D(r[:, 0], r[:, 1], r[:, 2], color=line_color, label=label)

    # Plot triads using quiver
    if step is not None:
        C = np.array([poses[i].attitude.T for i in range(0, len(poses), step)])
        r = np.array([poses[i].position for i in range(0, len(poses), step)])
        if plot_2d:
            x, y = r[:, 0], r[:, 1]
            ax.quiver(
                x, y,
                C[:, 0, 0],
                C[:, 0, 1],
                color=colors[0],
                scale=20.0,
                headwidth=2,
            )

            ax.quiver(
                x, y,
                C[:, 1, 0],
                C[:, 1, 1],
                color=colors[1],
                scale=20.0,
                headwidth=2,
            )
        else: 
            x, y, z = r[:, 0], r[:, 1], r[:, 2]
            ax.quiver(
                x,
                y,
                z,
                C[:, 0, 0],
                C[:, 0, 1],
                C[:, 0, 2],
                color=colors[0],
                length=arrow_length,
                arrow_length_ratio=0.1,
                linewidths=linewidth,
            )
            ax.quiver(
                x,
                y,
                z,
                C[:, 1, 0],
                C[:, 1, 1],
                C[:, 1, 2],
                color=colors[1],
                length=arrow_length,
                arrow_length_ratio=0.1,
                linewidths=linewidth,
            )
            ax.quiver(
                x,
                y,
                z,
                C[:, 2, 0],
                C[:, 2, 1],
                C[:, 2, 2],
                color=colors[2],
                length=arrow_length,
                arrow_length_ratio=0.1,
                linewidths=linewidth,
            )

    if plot_2d:
        ax.axis("equal")
    else:
        set_axes_equal(ax)
    return fig, ax


def set_axes_equal(ax: plt.Axes):
    """
    Sets the axes of a 3D plot to have equal scale.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    length = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - length, x_middle + length])
    ax.set_ylim3d([y_middle - length, y_middle + length])
    ax.set_zlim3d([z_middle - length, z_middle + length])
    