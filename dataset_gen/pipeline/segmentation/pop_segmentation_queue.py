#!/usr/bin/env python3
import os
import sys
import fcntl
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: pop_segmentation_queue.py <queue_file>", file=sys.stderr)
    sys.exit(1)

queue_file = Path(sys.argv[1]).resolve()
if not queue_file.exists():
    sys.exit(1)

with queue_file.open('r+') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    lines = f.readlines()
    if not lines:
        f.truncate(0)
        fcntl.flock(f, fcntl.LOCK_UN)
        sys.exit(1)
    first = lines[0].rstrip('\n')
    rest = lines[1:]
    f.seek(0)
    f.truncate(0)
    f.writelines(rest)
    fcntl.flock(f, fcntl.LOCK_UN)

print(first)
