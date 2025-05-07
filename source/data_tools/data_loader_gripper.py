from pathlib import Path
from typing import Optional
from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr
import cv2
from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr
from rosbags.image import message_to_cvimage
from tqdm import tqdm
import pandas as pd
import os
import numpy as np


class GripperData:

    TOPICS = {
        "/digit/left/image_raw": "/digit/left/image_raw",
        "/digit/right/image_raw": "/digit/right/image_raw",
        "/gripper_force_trigger": "/gripper_force_trigger",
        "/joint_states": "/joint_states",
        "/rosout": "/rosout",
        "/rosout_agg": "/rosout_agg",
        "/tf": "/tf",
        "/tf_static": "/tf_static",
        "/zedm/zed_node/depth/depth_registered": "/zedm/zed_node/depth/depth_registered",
        "/zedm/zed_node/imu/data": "/zedm/zed_node/imu/data",
        "/zedm/zed_node/imu/data_raw": "/zedm/zed_node/imu/data_raw",
        "/zedm/zed_node/odom": "/zedm/zed_node/odom",
        "/zedm/zed_node/pose": "/zedm/zed_node/pose",
        "/zedm/zed_node/pose_with_covariance": "/zedm/zed_node/pose_with_covariance",
        "/zedm/zed_node/rgb/camera_info": "/zedm/zed_node/rgb/camera_info",
        "/zedm/zed_node/rgb/image_rect_color": "/zedm/zed_node/rgb/image_rect_color",
        "/zedm/zed_node/left/image_rect_color": "/zedm/zed_node/left/image_rect_color",
        "/zedm/zed_node/right/image_rect_color": "/zedm/zed_node/right/image_rect_color",
        "/zedm/zed_node/left_raw/image_raw_color": "/zedm/zed_node/left_raw/image_raw_color",
        "/zedm/zed_node/right_raw/image_raw_color": "/zedm/zed_node/right_raw/image_raw_color",
    }

    IMAGE_TOPICS = {
        "/digit/left/image_raw",
        "/digit/right/image_raw",
        "/zedm/zed_node/depth/depth_registered",
        "/zedm/zed_node/rgb/image_rect_color",
        "/zedm/zed_node/left/image_rect_color",
        "/zedm/zed_node/right/image_rect_color",
        "/zedm/zed_node/left_raw/image_raw_color",
        "/zedm/zed_node/right_raw/image_raw_color",

    }

    def __init__(self, base_path: Path, rec_name: str, sensor_module_name: str):
        self.rec_name = rec_name
        self.base_path = base_path
        self.sensor_module_name = sensor_module_name
        self.bag = next(base_path.glob(f"raw/{self.rec_name}/{self.sensor_module_name}/gripper_recording_sync_{rec_name}_*.bag"), None)
        self.svo = next(base_path.glob(f"raw/{self.rec_name}/{self.sensor_module_name}/gripper_recording_sync_{rec_name}_*.svo"), None)

        self.extraction_path = self.base_path / "extracted" / self.rec_name / self.sensor_module_name
        self.extracted_bag = (self.extraction_path / "digit/left/image_raw").exists()

        self.visual_registration_output_path = self.extraction_path / "visual_registration"

    def extract_bag(self):
        if self.extracted_bag:
            print(f"[!] Bag data already extracted to {self.extraction_path}")
            return

        if not self.bag:
            raise FileNotFoundError(f"No bag file found for {self.bag}")

        print(f"[INFO] Reading from: {self.bag}")
        print(f"[INFO] File exists: {Path(self.bag).exists()}")

        csv_data = {v: [] for k, v in self.TOPICS.items() if k not in self.IMAGE_TOPICS}

        with AnyReader([Path(self.bag)]) as reader:

            for conn, timestamp, rawdata in tqdm(reader.messages(), total=reader.message_count):
                topic = conn.topic
                if topic not in self.TOPICS:
                    continue

                label = self.TOPICS[topic]
                ts = timestamp.to_nsec() if hasattr(timestamp, "to_nsec") else int(timestamp)
                ts_str = f"{ts}" 
                msg = reader.deserialize(rawdata, conn.msgtype)

                if topic in self.IMAGE_TOPICS:
                    try:
                        img = message_to_cvimage(msg)
                        out_dir = self.extraction_path / label.strip("/")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_file = out_dir / f"{ts_str}.png"
                        cv2.imwrite(str(out_file), img)
                    except Exception as e:
                        print(f"[!] Failed to decode image @ {ts:.9f} on {topic}: {e}")
                else:
                    try:
                        row = {"timestamp": ts}
                        for field in dir(msg):
                            if field.startswith('_') or callable(getattr(msg, field)):
                                continue
                            val = getattr(msg, field)
                            if isinstance(val, list):
                                for idx, item in enumerate(val):
                                    row[f"{field}.{idx}"] = item
                            elif hasattr(val, '__dict__'):
                                for subkey, subval in val.__dict__.items():
                                    row[f"{field}.{subkey}"] = subval
                            else:
                                row[field] = val
                            csv_data[label].append(row)
                    except Exception as e:
                        print(f"[!] Failed to extract data from {topic}: {e}")

        # Save CSVs
        for label, records in csv_data.items():
            df = pd.DataFrame(records)
            csv_dir = self.extraction_path / label.strip("/")
            csv_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_dir / "data.csv", index=False)
            print(f"[✓] Saved CSV: {csv_dir}/data.csv")

        self.extracted_bag = True

    def extract_video(self) -> None:
        """
        Converts extracted image topics into videos inside their respective folders.
        """

        if not self.extraction_path.exists():
            raise FileNotFoundError(f"Extraction path {self.extraction_path} does not exist.")

        for topic in self.IMAGE_TOPICS:
            label = self.TOPICS.get(topic)
            if label is None:
                print(f"[!] Skipping unknown topic: {topic}")
                continue

            out_dir = self.extraction_path / label.strip("/")
            if not out_dir.exists():
                print(f"[!] No images found for {topic} at {out_dir}")
                continue

            # Find all image files
            images = sorted(
                [img for img in os.listdir(out_dir) if img.endswith(".png")],
                key=lambda x: int(os.path.splitext(x)[0])
            )

            if len(images) < 2:
                print(f"[!] Not enough images to create a video for {topic}")
                continue

            # Estimate average fps
            timestamps = [int(os.path.splitext(img)[0]) for img in images]
            time_diffs = np.diff(timestamps)
            avg_dt = np.mean(time_diffs)
            fps = 1e9 / avg_dt  # frames per second

            print(f"[INFO] Topic {topic} -> Estimated fps: {fps:.2f}")

            # Prepare video writer
            first_frame = cv2.imread(str(out_dir / images[0]))
            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = out_dir / "data.mp4"
            video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # Write frames
            for image_file in tqdm(images, desc=f"Creating video for {label}", total=len(images)):
                img = cv2.imread(str(out_dir / image_file))
                video.write(img)

            video.release()
            print(f"[✓] Saved video for {topic} at {video_path}")

if __name__ == "__main__":

    # Example usage
    rec_name = "door_6"
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    sensor_module_name = "gripper_right"


    gripper_data = GripperData(base_path, rec_name, sensor_module_name)
    print(gripper_data.bag)
    
    # gripper_data.extract_bag()
    gripper_data.extract_video()
