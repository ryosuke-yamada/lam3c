# RoomTours Dataset Provenance

## Path prefixes

- `DATA=<dataset root>`
- `DATASET_GEN_ROOT=<repo>/dataset_gen`

## Common flow

1. Raw videos are collected under `DATA/RoomTours/raw_videos/*` or `DATA/HouseTours/data/files/official-housetour-dataset/videos`.
2. Scene segmentation writes `inside_only.avi` and `scenes/scene-*.mp4` under `DATA/RoomTours/processed_label_segments_*`.
3. Pi3 consumes `scenes/scene-*.mp4` and writes `pi3.ply` under each scene directory.
4. `roomtours_vggt_v2_200` additionally re-samples frames from `processed_label_segments_v2`, using existing `roomtours_pi3_v2` coverage as a prerequisite, then runs VGGT / COLMAP export.

## Dataset matrix

| Final dataset | Raw input | Segmentation output | Stage1 submit | Stage2 submit | Frame / image cap | Observed outputs | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `DATA/roomtours_pi3_v2` | `DATA/RoomTours/raw_videos/1st_download` | `DATA/RoomTours/processed_label_segments_v2` | `scripts/segmentation/submit_roomtour_segmentation.sh` | `scripts/pi3/submit_pi3_roomtours_v2.sh` | `400` | `2267 pi3.ply` | Historical `v2` dataset. Current source repo no longer had an explicit wrapper for this output root. |
| `DATA/roomtours_pi3_v2_200` | `DATA/RoomTours/raw_videos/1st_download` | `DATA/RoomTours/processed_label_segments_v2` | `scripts/segmentation/submit_roomtour_segmentation.sh` | `scripts/pi3/submit_pi3_roomtours.sh` | `200` | `1594 pi3.ply` | Current `submit_pi3_roomtours.sh` lineage. `lam3c` version freezes `ROOMTOURS_TARGET_FRAMES=200` explicitly. |
| `DATA/roomtours_pi3_2nd` | `DATA/RoomTours/raw_videos/2nd_download` | `DATA/RoomTours/processed_label_segments_2nd_download` | `scripts/segmentation/submit_roomtour_segmentation_2nd.sh` | `scripts/pi3/submit_pi3_roomtours_2nd.sh` | `400` | `8940 pi3.ply` | Source repo script existed, but current code default became `200`; `lam3c` version fixes the cap to `400`. |
| `DATA/roomtours_pi3_3rd` | `DATA/RoomTours/raw_videos/3rd_download` | `DATA/RoomTours/processed_label_segments_3rd_download` | `scripts/segmentation/submit_roomtour_segmentation_3rd.sh` | `scripts/pi3/submit_pi3_roomtours_3rd.sh` | `400` | `4767 pi3.ply` | Stage1 wrapper reconstructed from archived logs. Stage2 output root normalized to the observed dataset dir `roomtours_pi3_3rd`. |
| `DATA/roomtours_pi3_house24` | `DATA/RoomTours/raw_videos/house24_download` | `DATA/RoomTours/processed_label_segments_house24_download` | `scripts/segmentation/submit_roomtour_segmentation_house24.sh` | `scripts/pi3/submit_pi3_roomtours_house24.sh` | `400` | `15655 pi3.ply` | Stage1 wrapper reconstructed from archived logs. Stage2 output root normalized to the observed dataset dir `roomtours_pi3_house24`. |
| `DATA/roomtours_pi3_HouseTours` | `DATA/HouseTours/data/files/official-housetour-dataset/videos` | `DATA/RoomTours/processed_label_segments_HouseTours` | `scripts/segmentation/submit_roomtour_segmentation_housetours.sh` | `scripts/pi3/submit_pi3_roomtours_housetours.sh` | `400` | `10071 pi3.ply` | Output layout is `video_id/scene/pi3.ply` rather than `channel/video/scene/pi3.ply`. |
| `DATA/roomtours_pi3_v4_relaxed` | `DATA/RoomTours/raw_videos/v4_relaxed_download` | `DATA/RoomTours/processed_label_segments_v4_relaxed` | `scripts/segmentation/submit_roomtour_segmentation_v4_relaxed.sh` | `scripts/pi3/submit_pi3_roomtours_v4_relaxed.sh` | `400` | `8599 pi3.ply` | Existing source scripts matched the dataset. |
| `DATA/roomtours_pi3_batch_v6` | `DATA/RoomTours/raw_videos/batch_v6_download` | `DATA/RoomTours/processed_label_segments_batch_v6` | `scripts/segmentation/submit_roomtour_segmentation_batch_v6.sh` | `scripts/pi3/submit_pi3_roomtours_batch_v6_rt_HG.sh` | `400` | `11705 pi3.ply` | Stage1 wrapper reconstructed from archived logs. |
| `DATA/roomtours_pi3_batch_v7` | `DATA/RoomTours/raw_videos/batch_v7_download` | `DATA/RoomTours/processed_label_segments_batch_v7` | `scripts/segmentation/submit_roomtour_segmentation_batch_v7.sh` | `scripts/pi3/submit_pi3_roomtours_batch_v7_rt_HG.sh` | `400` | `8302 pi3.ply` | Stage1 wrapper reconstructed from archived logs. |
| `DATA/roomtours_pi3_batch_v8` | `DATA/RoomTours/raw_videos/batch_v8_download` | `DATA/RoomTours/processed_label_segments_batch_v8` | `scripts/segmentation/submit_roomtour_segmentation_batch_v8.sh` | `scripts/pi3/submit_pi3_roomtours_batch_v8_rt_HG.sh` | `400` | `19871 pi3.ply` | Stage1 wrapper reconstructed from archived logs. |
| `DATA/roomtours_vggt_v2_200` | `DATA/RoomTours/raw_videos/1st_download` | `DATA/RoomTours/processed_label_segments_v2` | `scripts/segmentation/submit_roomtour_segmentation.sh` | `scripts/vggt/submit_vggt_roomtours_v2.sh` | `200` images | `1773 points3D.bin`, `355321 jpg` | Requires existing `DATA/roomtours_pi3_v2` outputs to decide which scenes to process. |

## Key evidence used during reconstruction

- Archived Pi3 and segmentation job logs from the original internal workspace were used to recover
  - the historical `roomtours_pi3_v2` wrapper,
  - the `400`-frame cap for `roomtours_pi3_2nd`, and
  - the missing batch-v6 / v7 / v8 stage1 wrappers.

## Files added or adjusted inside `dataset_gen`

- Added stage1 wrappers reconstructed from logs:
  - `scripts/segmentation/submit_roomtour_segmentation_3rd.sh`
  - `scripts/segmentation/submit_roomtour_segmentation_house24.sh`
  - `scripts/segmentation/submit_roomtour_segmentation_batch_v6.sh`
  - `scripts/segmentation/submit_roomtour_segmentation_batch_v7.sh`
  - `scripts/segmentation/submit_roomtour_segmentation_batch_v8.sh`
- Added an explicit historical `v2` Pi3 wrapper:
  - `scripts/pi3/submit_pi3_roomtours_v2.sh`
- Adjusted copied Pi3 wrappers so they match the observed dataset directories and frame caps:
  - `scripts/pi3/submit_pi3_roomtours.sh`
  - `scripts/pi3/submit_pi3_roomtours_2nd.sh`
  - `scripts/pi3/submit_pi3_roomtours_3rd.sh`
  - `scripts/pi3/submit_pi3_roomtours_house24.sh`

## Release status

This directory now vendors the required Pi3 / VGGT runtime code and uses repo-relative paths. The remaining environment-specific pieces are the PBS resource directives and the dataset-root defaults embedded in each submit wrapper.
