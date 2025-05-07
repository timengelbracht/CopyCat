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

        self.extraction_path = self.base_path / "extracted" / self.rec_name / self.sensor_module_name
        self.extracted_rgbd = (self.extraction_path / "camera_rgb").exists()

        self.label_rgb = f"/camera_rgb"
        self.label_depth = f"/camera_depth"
        self.label_keyframes = f"/keyframes/rgb"

        self.visual_registration_output_path = self.extraction_path / "visual_registration"

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

        K = metadata.get("K")

        self.calibration = {}

        self.calibration["K"] = np.array(K).reshape(3, 3).T if K is not None else None
        self.calibration["h"] = metadata.get("h")
        self.calibration["w"] = metadata.get("w")

        self.fps = metadata.get("fps")
        self.timestamps = metadata.get("frameTimestamps")

        # TODO - add more metadata fields as needed

    def extract_data(self):
        if self.extracted_rgbd:
            print(f"[!] iPhone RGB-D data already extracted to {self.extraction_path}")
            return

        label_rgb = f"/camera_rgb"
        label_depth = f"/camera_depth"

        timestamps_ns = (np.array(self.timestamps) * 1e9).astype(np.int64).tolist()

        out_dir_rgb = self.extraction_path / self.label_rgb.strip("/")
        out_dir_depth = self.extraction_path / self.label_depth.strip("/")

        out_dir_rgb.mkdir(parents=True, exist_ok=True)
        out_dir_depth.mkdir(parents=True, exist_ok=True)

        for i, timestamp_ns in tqdm(enumerate(timestamps_ns), total=len(timestamps_ns)):
            rgb_img = cv2.imread(str(self.rgb / f"{i}.jpg"))
            depth_img = cv2.imread(str(self.depth / f"{i}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if rgb_img is None or depth_img is None:
                print(f"[!] Error loading image at index {i}")
                continue

            out_file_rgb = out_dir_rgb / f"{timestamp_ns}.jpg"
            out_file_depth = out_dir_depth / f"{timestamp_ns}.exr"

            cv2.imwrite(str(out_file_rgb), rgb_img)
            cv2.imwrite(str(out_file_depth), depth_img)

        self.extracted_rgbd = True

    def extract_keyframes(self):
        """
        Returns the keyframes as a numpy array.
        Reads every `stride`-th image from the directory.
        """

        if not self.extracted_rgbd:
            raise FileNotFoundError(f"VRS data not extracted to {self.extraction_path}")
        
        in_dir = self.extraction_path / self.label_rgb.strip("/")
        out_dir = self.visual_registration_output_path / self.label_keyframes.strip("/")

        out_dir.mkdir(parents=True, exist_ok=True)

        # Get sorted list of image files (assumes naming like 0.png, 1.png, ...)
        image_files = sorted(
            Path(in_dir).glob("*.jpg"),
            key=lambda x: int(x.stem)
        )

        img = cv2.imread(str(image_files[0]))
        if img is not None:
            timestamp = image_files[0].stem  
            out_path = out_dir / f"{timestamp}.png"
            cv2.imwrite(str(out_path), img)

    def extract_video(self, out_dir: Optional[str | Path] = None) -> None:
        """
        Extracts the video from the RGB images in the specified directory.
        """

        if out_dir is None:
            label_rgb = f"/camera_rgb"
            out_dir = self.extraction_path / label_rgb.strip("/")

        video_name = out_dir / 'data.mp4'

        # Read and sort images
        images = sorted(
            [img for img in os.listdir(out_dir) if img.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]))

        # Estimate average fps
        timestamps = [int(os.path.splitext(img)[0]) for img in images]
        time_diffs = np.diff(timestamps)  # nanoseconds
        avg_dt = np.mean(time_diffs)  # average nanosecond difference
        fps = 1e9 / avg_dt  # frames per second

        print(f"Estimated fps: {fps:.2f}")

        # Initialize video writer
        frame = cv2.imread(os.path.join(out_dir, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        # Write frames
        for image in tqdm(images, desc="Creating video", total=len(images)):
            img = cv2.imread(os.path.join(out_dir, image))
            video.write(img)

        video.release()

        print(f"[✓] Saved video to {out_dir}")

if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    rec_name = "door_6"
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    
    sensor_module_name = "iphone_left"

    iphone_data = IPhoneData(base_path, rec_name, sensor_module_name)

    # iphone_data.extract_data()
    iphone_data.extract_keyframes()

