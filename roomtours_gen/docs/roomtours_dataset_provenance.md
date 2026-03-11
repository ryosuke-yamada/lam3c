# RoomTours Dataset Provenance

## Canonical public pipeline

The public `roomtours_gen` interface is manifest-driven and scheduler-agnostic.

The canonical public input is:

- manifest: `../video_lists.csv`
- row format: `video_id,url`

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

Earlier internal experiments used multiple batch-specific partitions such as `v2`, `v6`, `v7`, and `v8`. Those partitions are intentionally not part of the public interface.

For the public release, the maintained path is the single manifest-driven pipeline described above.
