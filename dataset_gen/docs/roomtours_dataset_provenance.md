# RoomTours Dataset Provenance

## Canonical public pipeline

The public `dataset_gen` entrypoint no longer exposes the historical internal batch names such as `v2`, `v6`, `v7`, or `v8`.

The canonical public input is:

- manifest: `../video_lists.csv`
- row format: `video_id,url`
- observed rows in the current manifest: `3461`

The public workflow is:

1. download each listed video to `data/roomtours/videos/<video_id>.<ext>`
2. run CLIP-based filtering and label segmentation into `data/roomtours/segmentation/<video_id>/`
3. run Pi3 on `scenes/*.mp4` and write `data/roomtours/pi3/<video_id>/<scene>/pi3.ply`

## Public commands

| Stage | Command | Default output |
| --- | --- | --- |
| Download | `python download.py` | `data/roomtours/videos/` |
| Segmentation | `python segmentation.py` | `data/roomtours/segmentation/` |
| Pi3 | `python pi3.py` | `data/roomtours/pi3/` |

## Historical note

The previously used names `roomtours_v2`, `roomtours_2nd`, `roomtours_batch_v6`, `roomtours_batch_v7`, `roomtours_batch_v8`, and related variants were internal partitions of the raw-video collection and cluster-submission setup.

They are intentionally not part of the public interface anymore. For reproduction, the maintained path is the single manifest-driven pipeline described above.
