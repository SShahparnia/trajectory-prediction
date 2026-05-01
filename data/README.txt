Expanded Waymo processed infos splits
=====================================

Built from ALL entries in:
  - waymo_processed_data_v0_5_0_infos_train.pkl
  - waymo_processed_data_v0_5_0_infos_val.pkl

Total frames combined: 4761
  train: 3176 frames
  val:   595 frames
  test:  990 frames

Split is by lidar_sequence (whole segments) to avoid leakage.
Point cloud paths inside each info still reference /scratch/... originals.

Point training code at these pickles when you are ready (not wired yet).
