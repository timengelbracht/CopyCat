import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from pathlib import Path
import json
from typing import Optional
import numpy as np
import cv2
from tqdm import tqdm


class IPhoneData:
    def __init__(self, base_path: Path, rec_name: str, sensor_module_name: str):

        self.rec_name = rec_name
        self.base_path = base_path
        self.sensor_module_name = sensor_module_name
    
        rgbd_path = self.base_path / "raw" / self.rec_name / self.sensor_module_name / f"gripper_recording_sync_{self.rec_name}" / "EXR_RGBD"
        self.rgb = rgbd_path / "rgb"
        self.depth = rgbd_path / "depth"
        self.meta_data = rgbd_path / "metadata.json"

        self.K = None
        self.fps = None
        self.timestamps = None
        
        self.load_metadata()

        self.t_ns_init = 0

    def load_metadata(self):
        if not self.meta_data.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_data}")

        with open(self.meta_data, 'r') as f:
            metadata = json.load(f)

        self.K = metadata.get("K")
        self.fps = metadata.get("fps")
        self.timestamps = metadata.get("frameTimestamps")

        # TODO - add more metadata fields as needed

    def extract_data(self):
        
        label_rgb = f"/camera_rgb"
        label_depth = f"/camera_depth"

        timestamps_ns = (np.array(self.timestamps) * 1e9).astype(np.int64).tolist()

        out_dir_rgb = self.base_path / "extracted" / self.rec_name / self.sensor_module_name / label_rgb.strip("/")
        out_dir_depth = self.base_path / "extracted" / self.rec_name / self.sensor_module_name / label_depth.strip("/")

        out_dir_rgb.mkdir(parents=True, exist_ok=True)
        out_dir_depth.mkdir(parents=True, exist_ok=True)

        for i, timestamp_ns in tqdm(enumerate(timestamps_ns), total=len(timestamps_ns)):
            rgb_img = cv2.imread(str(self.rgb / f"{i}.jpg"))
            depth_img = cv2.imread(str(self.depth / f"{i}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if rgb_img is None or depth_img is None:
                print(f"Error loading image at index {i}")
                continue

            out_file_rgb = out_dir_rgb / f"{timestamp_ns}.jpg"
            out_file_depth = out_dir_depth / f"{timestamp_ns}.exr"

            cv2.imwrite(str(out_file_rgb), rgb_img)
            cv2.imwrite(str(out_file_depth), depth_img)


if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    rec_name = "bottle_6"
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    
    sensor_module_name = "iphone_left"

    iphone_data = IPhoneData(base_path, rec_name, sensor_module_name)

    iphone_data.extract_data()

