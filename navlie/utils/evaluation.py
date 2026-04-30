import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from navlie.lib.states import SE3State
from navlie.utils.alignment import state_list_to_evo_traj

from evo.main_ape import ape
from evo.main_rpe import rpe
from evo.core import sync
from evo.core.metrics import PoseRelation, logger
from evo.core.units import Unit


def compute_ate(
    gt_states: typing.List[SE3State],
    est_states: typing.List[SE3State],
    max_diff: float = 0.02,
    align: bool = True,
) -> typing.Tuple[float, float]:
    """Compute Absolute Trajectory Error (ATE) between two trajectories.
    This computes the RMSE of the attitude and position errors over the entire 
    trajectory.

    Parameters
    ----------
    gt_states: List[SE3State]
        Groundtruth poses
    est_states: List[SE3State]
        Estimated poses
    max_diff: float
        Max timestamp difference for trajectory association (seconds).
    align: bool
        If True, align trajectories before computing ATE.
    Returns
    -------
    typing.Tuple[float, float]
        (orientation_ate, position_ate)
    """
    traj_ref = state_list_to_evo_traj(gt_states)
    traj_est = state_list_to_evo_traj(est_states)

    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_diff,
    )

    attitude_ape_result = ape(
        traj_ref_sync,
        traj_est_sync,
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
    )

    position_ape_result = ape(
        traj_ref_sync,
        traj_est_sync,
        pose_relation=PoseRelation.translation_part,
        align=align,
    )

    att_ate = attitude_ape_result.stats["rmse"]
    pos_ate = position_ape_result.stats["rmse"]
    return att_ate, pos_ate


def compute_rpe_over_segment_lengths(
    gt_states: typing.List[SE3State],
    est_states: typing.List[SE3State],
    segment_lengths: typing.List[float] = None,
    delta_unit: Unit = Unit.meters,
    pose_relation: PoseRelation = PoseRelation.translation_part,
    max_diff: float = 0.02,
    all_pairs: bool = True,
    align: bool = True,
) -> typing.Dict[float, np.ndarray]:
    """Compute RPE error arrays for a list of trajectory segment lengths.

    Parameters
    ----------
    gt_states: List[SE3State]
        Groundtruth poses
    est_states: List[SE3State]
        Estimated poses
    segment_lengths: List[float]
        Segment lengths, with units `delta_unit`, to compute RPE over.
        If not specified, defaults to [8, 16, 24, 32, 40, 48] meters.
    delta_unit: Unit
        Unit for `segment_lengths`, default is meters.
    pose_relation: PoseRelation
        Which state to compute RPE over, default is translation only.
    max_diff: float
        Max timestamp difference for trajectory association (seconds).
    all_pairs: bool
        If True, use all pairs of states separated by `delta`; otherwise consecutive only.
    align: bool
        If True, align trajectories before computing RPE.
    Returns
    -------
    Dict[float, np.ndarray]:
        Dictionary mapping segment length to RPE error array.
    """

    if segment_lengths is None:
        segment_lengths = [8, 16, 24, 32, 40, 48]
    traj_ref = state_list_to_evo_traj(gt_states)
    traj_est = state_list_to_evo_traj(est_states)

    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_diff,
    )

    results: typing.Dict[float, np.ndarray] = {}
    for delta in segment_lengths:
        try:
            rpe_result = rpe(
                traj_ref_sync,
                traj_est_sync,
                pose_relation=pose_relation,
                delta=delta,
                delta_unit=delta_unit,
                all_pairs=all_pairs,
                align=align,
                support_loop=True,
            )
            results[delta] = rpe_result.np_arrays["error_array"]
        except Exception as e:
            logger.warning("RPE failed for delta=%.1f: %s", delta, e)
            results[delta] = np.array([])

    return results


def plot_rpe_boxplot(
    rpe_results: typing.Dict[str, typing.Dict[float, np.ndarray]],
    xlabel: str = "Segment length (m)",
    ylabel: str = "Translation RPE (m)",
    figsize: typing.Tuple[int, int] = (10, 5),
    ax: plt.Axes = None,
    **boxplot_kwargs,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """Generates a boxplot of RPE error distribution over segment lengths for one or more estimators.

    Parameters
    ----------
    rpe_results: Dict[str, Dict[float, np.ndarray]]
        Dictionary mapping estimator labels to dict of (segment length -> RPE array).
    xlabel: str
        x-axis label of plot
    ylabel: str
        y-axis label of plot.
    title: str
        Plot title.
    figsize: Tuple[int, int]
        Figure size when a new figure is created.
    ax: plt.Axes
        Optional axes to plot on.
    **boxplot_kwargs:
        Additional keyword arguments to pass to seaborn boxplot.
    Returns
    -------
    fig, ax
    """

    # Build a dataframe for seaborn boxplot
    # each row has the following columns:
    # Estimator, segment_length, error
    rows = []
    for estimator, length_errors in rpe_results.items():
        for delta, errors in length_errors.items():
            for val in errors:
                rows.append(
                    {"Estimator": estimator, "segment_length": delta, "error": val}
                )
    df = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    segment_lengths = sorted(df["segment_length"].unique())
    x_labels = [f"{dl:g}" for dl in segment_lengths]

    # Optional boxplot kwargs
    kwargs = dict(
        fill=False,
        width=0.8,
        gap=0.15,
        showfliers=False,
        linewidth=1.5,
    )
    kwargs.update(boxplot_kwargs)
    sns.boxplot(
        data=df,
        x="segment_length",
        y="error",
        hue="Estimator",
        order=segment_lengths,
        ax=ax,
        **kwargs,
    )

    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    return fig, ax
