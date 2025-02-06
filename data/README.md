## navlie data

This folder contains groundtruth data useful for generating simulated
trajectories in 3D space to test estimators on, using the B-Spline simulator
included in navlie. For an example of how to load in the data and use it to
generate simulated trajectories, see the example `ex_simulated_kitti.py`.

To generate your own simulated dataset from a custom trajectory, ensure that the
trajectory file is in TUM format, where each line is formatted as follows:
```
timestamp tx ty tz qx qy qz qw
```

where `timestamp` is the time in seconds, `tx ty tz` are the translation