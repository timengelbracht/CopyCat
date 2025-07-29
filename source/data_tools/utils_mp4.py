import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import pandas as pd
import telemetry_parser
import av

def get_frames_from_mp4(
    mp4_path: str | Path,
    outdir: str | Path | None = None,) -> Tuple[List[av.video.frame.VideoFrame], List[int]]:
    """
    Decode every video frame in an MP4 and return
        • a list of PyAV VideoFrame objects  (empty if `outdir` is set)
        • a parallel list of timestamps in **nanoseconds**
    
    When `outdir` is provided, frames are **written straight to disk**
    as JPEGs named   <timestamp_ns>.jpg   and not kept in RAM.
    This keeps memory usage low for long clips.
    """
    mp4_path = Path(mp4_path)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # check if the file exists
    if not mp4_path.is_file():
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")

    container = av.open(mp4_path)
    vstream   = container.streams.video[0]

    ns_per_tick = float(vstream.time_base) * 1e9     # scalar once
    total       = vstream.frames or 0                # may be 0 if unknown

    frame_ts: list[int] = []
    frame_objs: list[av.video.frame.VideoFrame] = []

    for frame in tqdm(container.decode(video=0), total=total,
                      unit="frame", desc=f"Extracting {mp4_path.name}"):
        if frame.pts is None:      # should not happen, but be safe
            continue

        ts_ns = int(frame.pts * ns_per_tick)
        frame_ts.append(ts_ns)

        if outdir is None:
            frame_objs.append(frame)                 # keep in RAM
        else:
            # write JPEG without extra conversions; PIL handles RGB24
            frame.to_image().save(outdir / f"{ts_ns}.jpg", format="JPEG")

    return frame_objs, frame_ts


def get_imu_from_mp4(mp4_file: str | Path) -> pd.DataFrame:
    """ Extract IMU data from a GoPro MP4 file and return it as a DataFrame (timestamps in nanosecs).
    Args:
        mp4_file (str | Path): Path to the MP4 file.
    Returns:
        pd.DataFrame: DataFrame containing the IMU data with columns for timestamps, gyro, and accelerometer.
    """

    if isinstance(mp4_file, Path):
        mp4_file = str(mp4_file)

    # Extract telemetry data
    tp = telemetry_parser.Parser(mp4_file)

    telemetry = tp.telemetry()
    imu = tp.normalized_imu()

    # parse to DataFrame
    timestamps_ms = np.fromiter((p["timestamp_ms"] for p in imu), dtype=np.float64)
    timestamps_ns = timestamps_ms * 1e6 

    # gyro and accel come as tuples
    gyro_arr = np.stack([p['gyro'] for p in imu])     
    acc_arr  = np.stack([p['accl'] for p in imu])      

    # ── 2. build the DataFrame column‑wise (no Python loop) ───────────────
    df = pd.DataFrame({
        'timestamp_ns'  : timestamps_ns.astype(np.int64),
        'angular_vel_x'  : gyro_arr[:,0],
        'angular_vel_y'  : gyro_arr[:,1],
        'angular_vel_z'  : gyro_arr[:,2],
        'linear_accel_x' : acc_arr[:,0],
        'linear_accel_y' : acc_arr[:,1],
        'linear_accel_z' : acc_arr[:,2],
    })

    return df