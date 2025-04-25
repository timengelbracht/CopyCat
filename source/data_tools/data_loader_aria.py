from pathlib import Path
from typing import Optional
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np

class AriaData:
    def __init__(self, base_path: Path, rec_name: str, sensor_module_name: str):
        self.rec_name = rec_name
        self.base_path = base_path
        self.sensor_module_name = sensor_module_name

        self.mps_path = self.base_path / "raw" / self.rec_name / self.sensor_module_name / f"mps_gripper_recording_sync_{rec_name}_vrs"
        # self.eye_gaze = AriaEyeGazeData(mps_path / "eye_gaze")
        # self.hand_tracking = AriaHandTrackingData(mps_path / "hand_tracking")
        # self.slam = AriaSlamData(mps_path / "slam")
        self.vrs_file = self.base_path / "raw" / self.rec_name / self.sensor_module_name / f"gripper_recording_sync_{rec_name}.vrs"

        self.device_calib = None
        self.provider = None
        self.load_provider()

        a = self.device_calib.get_camera_calib("camera-rgb");
        a = 2

    def load_provider(self):
        if not self.vrs_file.exists():
            raise FileNotFoundError(f"VRS file not found: {self.vrs_file}")
        
        self.provider = data_provider.create_vrs_data_provider(str(self.vrs_file))
        if not self.provider:
            raise RuntimeError(f"Failed to create data provider for {self.vrs_file}")

        self.device_calib = self.provider.get_device_calibration()

    def extract_vrs(self, undistort: bool = False):

        if not self.vrs_file:
            raise FileNotFoundError(f"No vrs file found for {self.vrs_file}")

        label = f"/camera_rgb"
        out_dir = self.base_path / "extracted" / self.rec_name / self.sensor_module_name / label.strip("/")
        out_dir.mkdir(parents=True, exist_ok=True)

        calib = self.device_calib.get_camera_calib("camera-rgb")
        pinhole = calibration.get_linear_camera_calibration(512, 512, 150)

        print(f"[INFO] Data provider created successfully")
        stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        for i in tqdm(range(0, self.provider.get_num_data(stream_id)), total=self.provider.get_num_data(stream_id)):
            image_data =  self.provider.get_image_data_by_index(stream_id, i)
            sensor_data = self.provider.get_sensor_data_by_index(stream_id, i)
            ts = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
            image_array = image_data[0].to_numpy_array()
            if undistort:
                image_array = calibration.distort_by_calibration(image_array, pinhole, calib)
            out_file = out_dir / f"{ts}.png"
            image_array= cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)            
            cv2.imwrite(str(out_file), image_array)

    def extract_mps(self):
        # Path to closed loop SLAM trajectory
        closed_loop_trajectory_file = self.mps_path / "slam" / "closed_loop_trajectory.csv"  # fixed typo in filename

        try:
            df = pd.read_csv(closed_loop_trajectory_file)
        except Exception as e:
            print(f"[!] Failed to read CSV {closed_loop_trajectory_file}: {e}")
            return

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df.columns:
            df["tracking_timestamp_us"] = (df["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save to extracted location
        label = f"/slam"
        csv_dir = self.base_path / "extracted" / self.rec_name / self.sensor_module_name / label.strip("/")
        csv_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_dir / "data.csv", index=False)

        print(f"[✓] Saved SLAM CSV: {csv_dir}/data.csv")

        # TODO - add more mps data extraction as needed
        # TODO - UTC timestamp is NOT changed at the moment, only device timestamp!!!!
        

if __name__ == "__main__":

    # Example usage
    rec_name = "bottle_6"
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    sensor_module_name = "aria_human_ego"

    aria_data = AriaData(base_path, rec_name, sensor_module_name)


    aria_data.extract_vrs()
    aria_data.extract_mps()

