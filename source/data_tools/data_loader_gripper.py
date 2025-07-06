import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Any
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
import shutil

import open3d as o3d
import time
import json

from utils import parse_str_ros_geoemtry_msgs_pose
from utils_parsing import flatten_dict, ros_to_dict, ROS_MESSAGE_PARSING_CONFIG, ros_message_to_dict_recursive

from data_indexer import RecordingIndex

class GripperData:

    NON_IMAGE_TOPICS = {
        "/gripper_force_trigger": "std_msgs/Float32",
        "/dynamixel_workbench/joint_states": "sensor_msgs/JointState",
        "/tf_static": "tf2_msgs/TFMessage",
        "/zedm/zed_node/imu/data": "sensor_msgs/Imu",
        "/force_torque/ft_sensor0/ft_sensor_readings/imu": "sensor_msgs/Imu",
        "/force_torque/ft_sensor0/ft_sensor_readings/temperature": "sensor_msgs/Temperature",
        "/force_torque/ft_sensor0/ft_sensor_readings/wrench": "geometry_msgs/WrenchStamped"
    }

    IMAGE_TOPICS = [
        "/digit/left/image_raw",
        "/digit/right/image_raw",
        "/zedm/zed_node/depth/depth_registered",
        "/zedm/zed_node/left_raw/image_raw_color",
        "/zedm/zed_node/right_raw/image_raw_color",
    ]

    def __init__(self, 
                 base_path: Path, 
                 rec_loc: str, 
                 rec_type: str,
                 rec_module: str,
                 interaction_indices: Optional[str] = None,
                 data_indexer: Optional[RecordingIndex] = None):

        self.rec_loc = rec_loc
        self.base_path = base_path
        self.rec_module = rec_module
        self.rec_type = rec_type
        self.interaction_indices = interaction_indices

        self.bag = next(base_path.glob(f"raw/{self.rec_loc}/{self.rec_type}/{self.rec_type}/{self.rec_loc}_{self.interaction_indices}*.bag"), None)

        self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}_bag"

        self.extracted_bag = (self.extraction_path / "digit/left/image_raw").exists()

        self.label_rgb = "/zedm/zed_node/left_raw/image_raw_color"
        self.rgb_extension = ".png"  # Assuming RGB images are in PNG format

        # self.visual_registration_output_path = self.extraction_path / "visual_registration"
        # self.svo_output_path = self.extraction_path / "svo"
        # self.points_ply_path = self.extraction_path / "points" / "data.ply"
        # self.traj_ply_path = self.extraction_path / "points" / "traj_bag.ply"
        # self.traj_ply_path_svo = self.extraction_path / "points" / "traj_svo.ply"

        # self.zed = None

        self.rgb_extension = ".png"
        self.logging_tag = f"{self.rec_loc}_{self.rec_type}_{self.rec_module}".upper()

    def extract_bag_full(self):

        if self.extracted_bag:
            print(f"[{self.logging_tag}] Bag data already extracted to {self.extraction_path}")
            return

        if not self.bag or not self.bag.is_file():
            raise FileNotFoundError(f"Bag file not found: {self.bag}")

        print(f"[{self.logging_tag}] Reading from: {self.bag}")

        # This will now store lists of *nested dictionaries* for each topic
        parsed_non_image_data: Dict[str, List[Dict[str, Any]]] = \
            {topic: [] for topic in self.NON_IMAGE_TOPICS.keys()}

        with AnyReader([self.bag]) as reader:
            for conn, bag_time, raw in tqdm(reader.messages(),
                                            total=getattr(reader, "message_count", None)):
                topic = conn.topic
                
                # Filter for topics we explicitly listed
                if topic not in self.IMAGE_TOPICS and topic not in self.NON_IMAGE_TOPICS.keys():
                    continue

                msg = reader.deserialize(raw, conn.msgtype)


                # --- Universal Timestamp Extraction ---
                ts: int
                if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    # ROS 2 message timestamps usually directly accessible
                    if hasattr(msg.header.stamp, 'sec') and hasattr(msg.header.stamp, 'nanosec'):
                        ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    else: # Fallback for ROS 1 or other stamp types with to_nsec()
                         try:
                             ts = msg.header.stamp.to_nsec()
                         except AttributeError:
                             print(f"Warning: Could not get nsec from {topic} header.stamp. Falling back to bag_time.")
                             ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
                else: # No header, use bag message time
                    ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
                
                ts_str = str(ts)

                # --- IMAGE TOPICS Handling ---
                if topic in self.IMAGE_TOPICS:
                    try:
                        img = message_to_cvimage(msg)
                        img_dir = self.extraction_path / topic.strip("/")
                        img_dir.mkdir(parents=True, exist_ok=True)

                        if hasattr(msg, 'encoding') and msg.encoding == "32FC1": # Depth images
                            np.save(img_dir / f"{ts_str}.npy", img)

                            # Visualization for depth
                            vis_dir = self.extraction_path / f"{topic.strip('/')}_visualization"
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            
                            min_d, max_d = 0.0, 5.0
                            norm = np.clip(np.nan_to_num(img), min_d, max_d)
                            norm = ((norm - min_d) / (max_d - min_d) * 255).astype(np.uint8)
                            heat = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
                            cv2.imwrite(vis_dir / f"{ts_str}.png", heat)
                        elif hasattr(msg, 'encoding') and msg.encoding == "bgra8": # BGRA images
                            cv2.imwrite(img_dir / f"{ts_str}.png", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
                        else: # Default to BGR8 (common for color images)
                            cv2.imwrite(img_dir / f"{ts_str}.png", img)
                    except Exception as e:
                        print(f"[!] Failed to decode image @ {ts} on {topic}: {e}")
                    continue # Image topics are handled, move to the next message

                # --- NON-IMAGE TOPICS - Direct Parsing and Storage ---
                try:
                    # Convert the ROS message object directly to a nested Python dictionary
                    parsed_msg_dict = ros_message_to_dict_recursive(msg)
                    
                    # Add the universal timestamp at the top level of the dictionary
                    # This ensures all extracted data has a consistent timestamp key.
                    parsed_msg_dict['timestamp'] = ts

                    parsed_non_image_data[topic].append(parsed_msg_dict)
                except Exception as e:
                    print(f"[!] Failed to parse and extract data from {topic}: {e}")

        # --- Dump Parsed Data to Structured Files ---
        # Instead of CSV, we'll save to JSON Lines or Parquet.
        
        for topic_name, messages_list in parsed_non_image_data.items():
            if not messages_list:
                print(f"[INFO] No data extracted for topic: {topic_name}. Skipping file dump.")
                continue

            output_dir = self.extraction_path / topic_name.strip("/")
            output_dir.mkdir(parents=True, exist_ok=True)

            # === Save as JSONL ===
            # jsonl_path = output_dir / "data.jsonl"
            # with open(jsonl_path, 'w') as f:
            #     for msg_dict in messages_list:
            #         json.dump(msg_dict, f)
            #         f.write('\n')
            # print(f"[✓] Saved JSON Lines: {jsonl_path}")

            # === Save as Parquet ===
            # try:
            #     import pyarrow as pa
            #     import pyarrow.parquet as pq
            #     table = pa.Table.from_pylist(messages_list)
            #     parquet_path = output_dir / "data.parquet"
            #     pq.write_table(table, parquet_path)
            #     print(f"[✓] Saved Parquet: {parquet_path}")
            # except ImportError:
            #     print(f"[!] pyarrow not installed. Skipping Parquet export for {topic_name}.")
            # except Exception as e:
            #     print(f"[!] Failed to save Parquet for {topic_name}: {e}")

            # === Save as CSV ===
            try:
                # Flatten each dict in the list
                flat_messages = [flatten_dict(msg_dict) for msg_dict in messages_list]
                df = pd.DataFrame(flat_messages)
                csv_path = output_dir / "data.csv"
                df.to_csv(csv_path, index=False)
                print(f"[✓] Saved CSV: {csv_path}")
            except Exception as e:
                print(f"[!] Failed to save CSV for {topic_name}: {e}")

                
        self.extracted_bag = True # Update the instance state

    def extract_bag(self):
        if self.extracted_bag:
            print(f"[!] Bag data already extracted to {self.extraction_path}")
            return                     # ← early-exit restored

        if not Path(self.bag).is_file():
            raise FileNotFoundError(self.bag)

        print(f"[INFO] Reading from: {self.bag}")

        csv_data = {topic: [] for topic in list(self.NON_IMAGE_TOPICS.keys())}

        with AnyReader([self.bag]) as reader:
            for conn, bag_time, raw in tqdm(reader.messages(),
                                            total=getattr(reader, "message_count", None)):
                topic = conn.topic
                if topic not in self.IMAGE_TOPICS and topic not in list(self.NON_IMAGE_TOPICS.keys()):
                    #print(f"[!] Skipping unknown topic: {topic}")
                    continue

                msg = reader.deserialize(raw, conn.msgtype)

                # ---------- universal timestamp ----------
                if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                else:
                    ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
                ts_str = str(ts)

                # ---------- IMAGE TOPICS ----------
                if topic in self.IMAGE_TOPICS:
                    
                    try:
                        img = message_to_cvimage(msg)
                        img_dir = self.extraction_path / topic.strip("/")
                        img_dir.mkdir(parents=True, exist_ok=True)

                        if msg.encoding == "32FC1":
                            np.save(img_dir / f"{ts_str}.npy", img)

                            # visualisation
                            min_d, max_d = 0.0, 5.0
                            norm = np.clip(np.nan_to_num(img), min_d, max_d)
                            norm = ((norm - min_d) / (max_d - min_d) * 255).astype(np.uint8)
                            heat = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

                            vis_dir = self.extraction_path / f"{topic.strip('/')}_visualization"
                            vis_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(vis_dir / f"{ts_str}.png", heat)

                        elif msg.encoding == "bgra8":
                            cv2.imwrite(img_dir / f"{ts_str}.png",
                                        cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
                        else:  # bgr8
                            cv2.imwrite(img_dir / f"{ts_str}.png", img)
                    except Exception as e:
                        print(f"[!] Failed to decode image @ {ts} on {topic}: {e}")
                    continue  # image topics handled, next message

                # ---------- NON-IMAGE TOPICS ----------
                try:
                    row = {"timestamp": ts}
                    for name in dir(msg):
                        if name.startswith('_') or callable(getattr(msg, name)):
                            continue
                        val = getattr(msg, name)
                        if isinstance(val, list):
                            for i, item in enumerate(val):
                                row[f"{name}.{i}"] = item
                        elif hasattr(val, "__dict__"):
                            for k, v in val.__dict__.items():
                                row[f"{name}.{k}"] = v
                        else:
                            row[name] = val
                    csv_data[topic].append(row)
                except Exception as e:
                    print(f"[!] Failed to extract data from {topic}: {e}")

        # ---------- dump CSV files ----------
        for label, rows in csv_data.items():
            csv_dir = self.extraction_path / label.strip("/")
            csv_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_dir / "data.csv", index=False)
            print(f"[✓] Saved CSV: {csv_dir}/data.csv")

        self.extracted_bag = True

    # def extract_svo(self):
    #     """
    #     Copy SVO data into the extraction path.
    #     """
    #     if not self.svo:
    #         raise FileNotFoundError(f"No SVO file found for {self.svo}")

    #     svo_path = Path(self.svo)
    #     if not svo_path.exists():
    #         raise FileNotFoundError(f"SVO file does not exist: {svo_path}")
        
    #     #ensure path 
    #     out_dir = self.svo_output_path
    #     out_dir.mkdir(parents=True, exist_ok=True)

    #     # Copy the SVO file to the extraction path
    #     svo_file = out_dir / "data.svo"
    #     if not svo_file.exists():
    #         shutil.copy(svo_path, svo_file)
    #         print(f"[✓] Copied SVO file to {svo_file}")
    #     else:
    #         print(f"[!] SVO file already exists at {svo_file}")

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

            if "depth" in topic:
                for image_file in tqdm(images, desc=f"Creating video for {label}", total=len(images)):
                    depth = cv2.imread(os.path.join(out_dir, image_file))

                    depth_norm = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    img8 = (depth_norm * 255).astype(np.uint8)

                    heat = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
                    video.write(heat)
            else:
                for image_file in tqdm(images, desc=f"Creating video for {label}", total=len(images)):
                    img = cv2.imread(str(out_dir / image_file))
                    video.write(img)

            video.release()
            print(f"[✓] Saved video for {topic} at {video_path}")

    def get_trajectory(self) -> pd.DataFrame:
        """Return the trajectory as a pandas DataFrame"""
        # TODO add covariance extraction
        csv_dir = self.extraction_path / self.TOPICS["/zedm/zed_node/pose_with_covariance"].strip("/") / "data.csv"

        if not Path(csv_dir).exists():
            raise FileNotFoundError(f"Trajectory CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        df[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]] = df["pose.pose"].apply(parse_str_ros_geoemtry_msgs_pose)
        cleaned_df = df[["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]

        return cleaned_df
    
    # def _get_zed_from_svo(self, svo_path: str | None = None) -> sl.Camera:
    #     """Get ZED camera from SVO file."""

    #     if svo_path is None:
    #         svo_path = self.svo_output_path / "data.svo"
    #     if not svo_path:
    #         raise FileNotFoundError("[GRIPPER] No SVO file found. Please extract the SVO first.")
        
    #     zed = sl.Camera()
    #     input_type = sl.InputType()
    #     input_type.set_from_svo_file(str(svo_path))
        
    #     init_params = sl.InitParameters(input_t=input_type)
    #     init_params.coordinate_units = sl.UNIT.METER
    #     init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    #     init_params.camera_resolution = sl.RESOLUTION.HD720
        
    #     status = zed.open(init_params)
    #     if status != sl.ERROR_CODE.SUCCESS:
    #         raise RuntimeError(f"Failed to open SVO file: {status}")
        
    #     return zed
    
    # def _close_zed(self) -> None:
    #     """Close ZED camera."""
    #     if self.zed is not None:
    #         self.zed.close()
    #         self.zed = None
    #     else:
    #         print("[!] ZED camera is not open.")
        
    # def show_svo_recording(self):

    #     if self.zed is None:
    #         self.zed = self._get_zed_from_svo()

    #     runtime_params = sl.RuntimeParameters()

    #     frame_count = 0
    #     while True:
    #         if self.zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
    #             break
    #         image = sl.Mat()
    #         depth_map = sl.Mat()
    #         self.zed.retrieve_image(depth_map, sl.VIEW.DEPTH)
    #         self.zed.retrieve_image(image, sl.VIEW.LEFT)
    #         image_np = image.get_data()
    #         timestamp_ns = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
    #         cv2.imshow("ZED SVO", image_np)
    #         frame_count += 1
    #         print(f"Timestamp: {timestamp_ns}")
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     cv2.destroyAllWindows()
    #     self._close_zed()

    # def save_trajectory_as_ply(self, color: tuple = (1.0, 0.0, 0.0)):
    #     """
    #     Save trajectory positions as a point cloud PLY file.
        
    #     Args:
    #         df: DataFrame with columns ['tx', 'ty', 'tz']
    #         ply_path: Output path for the PLY file
    #         color: RGB color tuple for all points (0.0–1.0)
    #     """
    #     df = self.get_trajectory()
    #     # df = self.get_trajectory_from_svo()

    #     ply_path = self.traj_ply_path

    #     if not {"tx", "ty", "tz"}.issubset(df.columns):
    #         raise ValueError("DataFrame must contain 'tx', 'ty', 'tz' columns")

    #     positions = df[["tx", "ty", "tz"]].to_numpy()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(positions)

    #     color_np = np.array([color], dtype=np.float32).repeat(len(positions), axis=0)
    #     pcd.colors = o3d.utility.Vector3dVector(color_np)

    #     ply_path = Path(ply_path)
    #     ply_path.parent.mkdir(parents=True, exist_ok=True)
    #     o3d.io.write_point_cloud(str(ply_path), pcd)
    #     print(f"[GRIPPER] ✅ Saved trajectory as PLY → {ply_path}")

    # def get_trajectory_from_svo(self, svo_path: str | None = None) -> pd.DataFrame:
    #     """Get the trajectory from SVO file."""

    #     if self.zed is None:
    #         self.zed = self._get_zed_from_svo(svo_path)

    #     if svo_path is None:
    #         svo_path = self.svo_output_path / "data.svo"

    #     tracking_params = sl.PositionalTrackingParameters()
    #     self.zed.enable_positional_tracking(tracking_params)
    #     runtime_params = sl.RuntimeParameters()
    #     pose = sl.Pose()
    #     timestamps, poses = [], []

    #     while True:
    #         if self.zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
    #             break

    #         # Get timestamp and pose
    #         timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

    #         if self.zed.get_position(pose) == sl.POSITIONAL_TRACKING_STATE.OK:
    #             t = pose.get_translation()
    #             r = pose.get_orientation()

    #             poses.append([
    #                 t.get()[0], t.get()[1], t.get()[2],  # tx, ty, tz
    #                 r.get()[0], r.get()[1], r.get()[2], r.get()[3],  # qx, qy, qz, qw
    #             ])
    #             timestamps.append(timestamp)

    #     self.zed.disable_positional_tracking()
    #     self._close_zed()

    #     df = pd.DataFrame(poses, columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
    #     df["timestamp"] = timestamps
    #     df_svo = df[["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]
    #     df_ros = self.get_trajectory()

    #     # df.to_csv(out_csv, index=False)
    #     # print(f"✅ Saved trajectory with {len(df)} poses to {out_csv}")

    #     return df_svo
    
    # def visualize_svo_trajectory(self):

    #     import matplotlib.pyplot as plt

    #     if self.zed is None:
    #         self.zed = self._get_zed_from_svo()

    #     tracking_params = sl.PositionalTrackingParameters()
    #     if self.zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
    #         print("Failed to enable tracking")
    #         exit(1)

    #     # === Grab trajectory ===
    #     runtime_params = sl.RuntimeParameters()
    #     pose = sl.Pose()
    #     positions = []

    #     while True:
    #         if self.zed.grab(runtime_params) == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    #             break
    #         if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    #             if self.zed.get_position(pose, sl.REFERENCE_FRAME.WORLD) == sl.POSITIONAL_TRACKING_STATE.OK:
    #                 t = pose.get_translation(sl.Translation()).get()
    #                 positions.append(t)

    #     self.zed.disable_positional_tracking()
    #     self._close_zed()

    #     positions = np.array(positions)
    #     print(f"Read {len(positions)} poses")

    #     # === Plot the 3D trajectory ===
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory")
    #     ax.set_xlabel("X [m]")
    #     ax.set_ylabel("Y [m]")
    #     ax.set_zlabel("Z [m]")
    #     ax.set_title("Camera Trajectory from SVO")
    #     ax.legend()
    #     plt.show()

    # def _points_raw_to_ply(self, svo_path: str | None = None, output_ply_path: str | None = None) -> None:
    #     """Extract fused point cloud from a ZED SVO and save as PLY."""
        
    #     if self.zed is None:
    #         self.zed = self._get_zed_from_svo()

    #     if svo_path is None:
    #         svo_path = self.svo_output_path / "data.svo"

    #     if output_ply_path is None:
    #         output_ply_path = self.points_ply_path

    #     # --- Enable positional tracking ---
    #     tracking_params = sl.PositionalTrackingParameters()
    #     if self.zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
    #         self._close_zed()
    #         raise RuntimeError("[GRIPPER] Failed to enable positional tracking")

    #     # --- Enable spatial mapping (fused point cloud only) ---
    #     map_params = sl.SpatialMappingParameters(
    #         resolution=sl.MAPPING_RESOLUTION.HIGH,
    #         mapping_range=sl.MAPPING_RANGE.AUTO,
    #         save_texture=False,
    #         use_chunk_only=True,
    #         map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    #     )
    #     if self.zed.enable_spatial_mapping(map_params) != sl.ERROR_CODE.SUCCESS:
    #         self._close_zed()
    #         raise RuntimeError("[GRIPPER] Failed to enable spatial mapping")

    #     runtime_params = sl.RuntimeParameters()
    #     fused_map = sl.FusedPointCloud()
        
    #     # --- Process SVO until the end ---
    #     while True:
    #         if self.zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
    #             break

    #         self.zed.request_spatial_map_async()



    #     # --- Extract and save fused map ---

    #     if self.zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
    #         self.zed.extract_whole_spatial_map(fused_map)

    #     output_path = Path(output_ply_path)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     fused_map.save(str(output_path), sl.MESH_FILE_FORMAT.PLY)

    #     print(f"[GRIPPER] Saved fused point cloud to {output_path}")

    #     # --- Cleanup ---
    #     self.zed.disable_spatial_mapping()
    #     self.zed.disable_positional_tracking()
    #     self._close_zed()

    

        
if __name__ == "__main__":

    # Example usage

    rec_location = "bedroom_1"
    base_path = Path(f"/data/ikea_recordings")

    
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    gripper_queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=None, 
        recorder="gripper"
    )

    
    for loc, inter, rec, ii, path in gripper_queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        gripper_data = GripperData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)

        gripper_data.extract_bag_full()