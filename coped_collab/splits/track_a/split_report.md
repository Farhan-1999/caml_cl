# Split report: NYUARPL, robots ptah, race1

Window: 206.96 s while the drone is airborne (6206 complete 30 Hz frames, 975 in buffers).
Dropped frames: {'outside_window': 9582, 'ptah.rgb': 0, 'ptah.depth_zoe': 0, 'ptah.ir': 0, 'race1.rgb': 0, 'race1.depth_zoe': 0, 'race1.ir': 0}

| Split | Samples | Rate | Seconds | Blocks (s from window start) | Unlabeled |
|---|---|---|---|---|---|
| train | 1345 | 10 | 134.2 | 3-67, 103-167, 203-207 | 20.7% |
| val | 300 | all frames | 9.9 | 73-77, 173-178 | 7.6% |
| test | 899 | all frames | 29.9 | 83-97, 183-197 | 33.6% |

Class pixel share (% of all pixels):

| Class | train | val | test |
|---|---|---|---|
| 0 ceiling | 7.04 | 4.87 | 7.17 |
| 1 floor | 42.87 | 39.49 | 38.33 |
| 2 wall | 13.02 | 28.43 | 11.77 |
| 3 chair | 0.77 | 2.61 | 0.27 |
| 4 door | 6.26 | 2.48 | 1.78 |
| 5 window | 8.67 | 11.87 | 6.54 |
| 6 light | 0.05 | 0.00 | 0.04 |
| 7 person | 0.53 | 0.75 | 0.34 |
| 8 table/furniture | 0.09 | 1.93 | 0.19 |

Pairing offset to the ego frame (ms):

| Stream | Median | p95 | Max |
|---|---|---|---|
| ptah.rgb | 0.0 | 0.0 | 0.0 |
| ptah.depth_zoe | 0.0 | 0.0 | 0.0 |
| ptah.ir | 0.0 | 0.0 | 0.0 |
| race1.rgb | 8.8 | 16.37 | 33.37 |
| race1.depth_zoe | 8.8 | 16.37 | 33.37 |
| race1.ir | 12.3 | 24.55 | 45.73 |
