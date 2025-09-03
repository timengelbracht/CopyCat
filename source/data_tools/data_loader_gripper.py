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
from scipy.signal import butter, filtfilt

import open3d as o3d
import time
import json
from utils_bag import get_topics_from_bag

from utils import parse_str_ros_geoemtry_msgs_pose
from utils_parsing import flatten_dict, ros_to_dict, ROS_MESSAGE_PARSING_CONFIG, ros_message_to_dict_recursive

from data_indexer import RecordingIndex
from utils_yaml import load_camchain, load_imucam, load_imu
import json

from data_loader_aria import AriaData
from utils_calibration import _slerp_pose_series_to_targets, compensate_wrench_batch, find_contact_free_segments, _estimate_tool_params_ls

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
                 data_indexer: Optional[RecordingIndex] = None,
                 color: str = "yellow"
):

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

        self.logging_tag = f"{self.rec_loc}_{self.rec_type}_{self.rec_module}".upper()
        self.color = color
        self.calibration = self.get_calibration()

        self.loader_aria_gripper = AriaData(
            base_path=self.base_path,
            rec_loc=self.rec_loc,
            rec_type=self.rec_type,
            rec_module="aria_gripper",
            interaction_indices=self.interaction_indices,
            data_indexer=data_indexer
        )

        # self.visual_registration_output_path = self.extraction_path / "visual_registration"
        # self.svo_output_path = self.extraction_path / "svo"
        # self.points_ply_path = self.extraction_path / "points" / "data.ply"
        # self.traj_ply_path = self.extraction_path / "points" / "traj_bag.ply"
        # self.traj_ply_path_svo = self.extraction_path / "points" / "traj_svo.ply"

        # self.zed = None

        self.rgb_extension = ".png"
       
        

    def get_calibration(self):
        """
        load metadata from camera calibration file
        """
        # types of calibration models taht we calibrated beforehand
        # pinhole-equi for pycolmap/ hloc
        # omni-radtan for openvins
        calibration = {}

        calib_path = self.extraction_path / "calib"
        calib_path_raw = self.base_path / "raw" / "calib" / f"gripper_{self.color}"

        if not calib_path.exists():
            print(f"[{self.logging_tag}] No calibration file found at {calib_path}")
            # copy all contents in raw to calib_path
            shutil.copytree(calib_path_raw, calib_path)
            print(f"[{self.logging_tag}] Copied calibration files from {calib_path_raw} to {calib_path}")


        calib_path_cam_imu = calib_path
        # calib_path_gravity_compensation = calib_path / "gravity_comp"

        file = calib_path_cam_imu.glob("*camchain.yaml")
        file_imucam = calib_path_cam_imu.glob("*imucam.yaml")
        file_imu = calib_path_cam_imu.glob("*imu.yaml")

        calib_file = next(file, None)
        if calib_file is None:
            print(f"[{self.logging_tag}] No calibration file found at {calib_path}")
            raise FileNotFoundError(f"No calibration file found at {calib_path}")

        calib_file_imucam = next(file_imucam, None)
        if calib_file_imucam is None:
            print(f"[{self.logging_tag}] No IMU calibration file found at {calib_path}")
            raise FileNotFoundError(f"No IMU calibration file found at {calib_path}")
        
        calib_file_imu = next(file_imu, None)
        if calib_file_imu is None:
            print(f"[{self.logging_tag}] No IMU calibration file found at {calib_path}")
            raise FileNotFoundError(f"No IMU calibration file found at {calib_path}")

        print(f"[{self.logging_tag}] Loading calibration file {calib_file}")
        print(f"[{self.logging_tag}] Loading IMU calibration file {calib_file_imucam}")

        calibration = {}
        cams = ["cam0", "cam1", "cam2"]
        for cam_name in cams:
            calib_data = load_camchain(calib_file, cam_name=cam_name)
            calib_data_imu = load_imucam(calib_file_imucam, imu_name=cam_name)

            clb = {}
            h = calib_data.resolution[1]
            w = calib_data.resolution[0]
            f_x, f_y = calib_data.intrinsics[0], calib_data.intrinsics[1]
            c_x, c_y = calib_data.intrinsics[2], calib_data.intrinsics[3]
            disortion = calib_data.distortion
            timeshift_cam_imu = calib_data_imu.timeshift_cam_imu
            T_cam_imu = calib_data_imu.T_cam_imu

            K = np.array([
                [f_x, 0, c_x],
                [0, f_y, c_y],
                [0, 0, 1]
            ], dtype=np.float32)

            # convert to dictionary
            if calib_data.model == "pinhole" and calib_data.distortion_model == "equidistant":
                model = "OPENCV_FISHEYE"
                type = "PINHOLE"
                colmap_camera_cfg = {
                "model":  model,
                "width":   w,
                "height":  h,
                "params": [f_x, f_y, c_x, c_y] + disortion,
                }

            elif calib_data.model == "pinhole" and calib_data.distortion_model == "radtan":
                model = "PINHOLE"
                colmap_camera_cfg = {
                    "model":  model,
                    "width":   w,
                    "height":  h,
                    "params": [f_x, f_y, c_x, c_y] + disortion,
                }

            clb["K"] = K
            clb["model"] = model
            clb["w"] = w
            clb["h"] = h
            clb["focal_length"] = np.array([f_x, f_y], dtype=np.float32)
            clb["principal_point"] = np.array([c_x, c_y], dtype=np.float32)
            clb["distortion"] = np.array(disortion, dtype=np.float32)
            clb["colmap_camera_cfg"] = colmap_camera_cfg
            clb["T_cam_imu"] = np.array(T_cam_imu, dtype=np.float32) 
            clb["timeshift_cam_imu"] = timeshift_cam_imu

            calibration[cam_name] = clb

        calib_data_imu = load_imu(calib_file_imu, imu_name="imu0")
        T_imu_body = calib_data_imu.T_imu_body
        T_imu_tool = calib_data_imu.T_imu_tool
        T_imu_sensor = calib_data_imu.T_imu_sensor
        calibration["imu0"] = {}
        calibration["imu0"]["T_imu_body"] = np.array(T_imu_body, dtype=np.float32)
        calibration["imu0"]["T_imu_tool"] = np.array(T_imu_tool, dtype=np.float32)
        calibration["imu0"]["T_imu_sensor"] = np.array(T_imu_sensor, dtype=np.float32)

        return calibration

    def extract_bag_full(self):

        if self.extracted_bag:
            print(f"[{self.logging_tag}] Bag data already extracted to {self.extraction_path}")
            return

        if not self.bag or not self.bag.is_file():
            raise FileNotFoundError(f"Bag file not found: {self.bag}")

        print(f"[{self.logging_tag}] Reading from: {self.bag}")

        # TODO test
        get_topics_from_bag(self.IMAGE_TOPICS, self.NON_IMAGE_TOPICS, self.bag, self.extraction_path)

        # This will now store lists of *nested dictionaries* for each topic
        # parsed_non_image_data: Dict[str, List[Dict[str, Any]]] = \
        #     {topic: [] for topic in self.NON_IMAGE_TOPICS.keys()}

        # with AnyReader([self.bag]) as reader:
        #     for conn, bag_time, raw in tqdm(reader.messages(),
        #                                     total=getattr(reader, "message_count", None)):
        #         topic = conn.topic
                
        #         # Filter for topics we explicitly listed
        #         if topic not in self.IMAGE_TOPICS and topic not in self.NON_IMAGE_TOPICS.keys():
        #             continue

        #         msg = reader.deserialize(raw, conn.msgtype)

        #         # --- Universal Timestamp Extraction ---
        #         ts: int
        #         if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        #             # ROS 2 message timestamps usually directly accessible
        #             if hasattr(msg.header.stamp, 'sec') and hasattr(msg.header.stamp, 'nanosec'):
        #                 ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        #             else: # Fallback for ROS 1 or other stamp types with to_nsec()
        #                  try:
        #                      ts = msg.header.stamp.to_nsec()
        #                  except AttributeError:
        #                      print(f"Warning: Could not get nsec from {topic} header.stamp. Falling back to bag_time.")
        #                      ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
        #         else: # No header, use bag message time
        #             ts = bag_time.to_nsec() if hasattr(bag_time, "to_nsec") else int(bag_time)
                
        #         ts_str = str(ts)

        #         # --- IMAGE TOPICS Handling ---
        #         if topic in self.IMAGE_TOPICS:
        #             try:
        #                 img = message_to_cvimage(msg)
        #                 img_dir = self.extraction_path / topic.strip("/")
        #                 img_dir.mkdir(parents=True, exist_ok=True)

        #                 if hasattr(msg, 'encoding') and msg.encoding == "32FC1": # Depth images
        #                     np.save(img_dir / f"{ts_str}.npy", img)

        #                     # Visualization for depth
        #                     vis_dir = self.extraction_path / f"{topic.strip('/')}_visualization"
        #                     vis_dir.mkdir(parents=True, exist_ok=True)
                            
        #                     min_d, max_d = 0.0, 5.0
        #                     norm = np.clip(np.nan_to_num(img), min_d, max_d)
        #                     norm = ((norm - min_d) / (max_d - min_d) * 255).astype(np.uint8)
        #                     heat = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        #                     cv2.imwrite(vis_dir / f"{ts_str}.png", heat)
        #                 elif hasattr(msg, 'encoding') and msg.encoding == "bgra8": # BGRA images
        #                     cv2.imwrite(img_dir / f"{ts_str}.png", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
        #                 else: # Default to BGR8 (common for color images)
        #                     cv2.imwrite(img_dir / f"{ts_str}.png", img)
        #             except Exception as e:
        #                 print(f"[!] Failed to decode image @ {ts} on {topic}: {e}")
        #             continue # Image topics are handled, move to the next message

        #         # --- NON-IMAGE TOPICS - Direct Parsing and Storage ---
        #         try:
        #             # Convert the ROS message object directly to a nested Python dictionary
        #             parsed_msg_dict = ros_message_to_dict_recursive(msg)
                    
        #             # Add the universal timestamp at the top level of the dictionary
        #             # This ensures all extracted data has a consistent timestamp key.
        #             parsed_msg_dict['timestamp'] = ts

        #             parsed_non_image_data[topic].append(parsed_msg_dict)
        #         except Exception as e:
        #             print(f"[!] Failed to parse and extract data from {topic}: {e}")

        # # --- Dump Parsed Data to Structured Files ---
        # # Instead of CSV, we'll save to JSON Lines or Parquet.
        
        # for topic_name, messages_list in parsed_non_image_data.items():
        #     if not messages_list:
        #         print(f"[INFO] No data extracted for topic: {topic_name}. Skipping file dump.")
        #         continue

        #     output_dir = self.extraction_path / topic_name.strip("/")
        #     output_dir.mkdir(parents=True, exist_ok=True)

        #     try:
        #         # Flatten each dict in the list
        #         flat_messages = [flatten_dict(msg_dict) for msg_dict in messages_list]
        #         df = pd.DataFrame(flat_messages)
        #         csv_path = output_dir / "data.csv"
        #         df.to_csv(csv_path, index=False)
        #         print(f"[✓] Saved CSV: {csv_path}")
        #     except Exception as e:
        #         print(f"[!] Failed to save CSV for {topic_name}: {e}")

                
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
    
    # def get_closed_loop_trajectory(self) -> pd.DataFrame:
    #     """
    #     Extracts the closed-loop trajectory from gripper aria data.
    #     """
        
    #     # get the unaligned mps trajectory from aria gripper data
    #     df = self.loader_aria_gripper.get_closed_loop_trajectory()

    #     return df
    
    def get_force_torque_measurements(self) -> pd.DataFrame:
        """
        Extracts force-torque measurements from the gripper data.
        """
        
        # get the unaligned force-torque measurements from aria gripper data
        file = self.extraction_path / ("/force_torque/ft_sensor0/ft_sensor_readings/wrench").strip("/") / "data.csv"

        if not file.exists():
            raise FileNotFoundError(f"Force-torque data file not found: {file}")
        
        df = pd.read_csv(file)

        return df


    def apply_force_torque_gravity_compensation(self, visualize: bool = False) -> pd.DataFrame:
        
        df_ft = self.get_force_torque_measurements()
        # doesnt need to be aligned closed loop traj since z axis is along gravity for both aria and leica system
        df_poses = self.loader_aria_gripper.get_closed_loop_trajectory()

        # wrench in ft frame (sensor)
        wrench_ft = df_ft[["timestamp", "wrench.force.x", "wrench.force.y", "wrench.force.z",
                                        "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]]
        wrench_timestamps_ns = wrench_ft["timestamp"].to_numpy(dtype=np.int64)

        f_meas_S = wrench_ft[["wrench.force.x", "wrench.force.y", "wrench.force.z"]].to_numpy(dtype=np.float64)  # (N, 3)
        tau_meas_S = wrench_ft[["wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]].to_numpy(dtype=np.float64)

        # aria slam poses in aria world frame
        poses_aria = df_poses[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device",
                                            "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]]

        R_ariaworld_ariadevice_list, t_ariaworld_ariadevice_list = _slerp_pose_series_to_targets(poses_aria, wrench_timestamps_ns)
        R_ariaworld_ariadevice = np.array(R_ariaworld_ariadevice_list, dtype=np.float64)  # (N, 3, 3)

        # TODO check if already applied
        # TODO on the fly compensation with known mass and CoG using static parts off the measuremets

        T_ariadevice_ariacam = self.loader_aria_gripper.calibration["NON_PINHOLE"]["T_device_camera"]
        R_ariadevice_ariacam = T_ariadevice_ariacam[:3, :3]

        T_ariacam_imuft = self.calibration["cam2"]["T_cam_imu"]
        R_ariacam_imuft = T_ariacam_imuft[:3, :3]

        # no rot between imu and imuft
        R_ariacam_ft = R_ariacam_imuft

        # rot from ft to ariadevice
        R_ariadevice_ft = R_ariadevice_ariacam @ R_ariacam_ft
        R_ariaworld_ft = R_ariaworld_ariadevice @ R_ariadevice_ft  # (N, 3, 3)
        R_ft_ariaworld = np.transpose(R_ariaworld_ft, axes=(0, 2, 1))  # (N, 3, 3)

        # compute contact-free segments for static estimation
        windows, mask, dbg = find_contact_free_segments(
            timestamps_ns=wrench_ft["timestamp"].to_numpy(np.int64),
            F_meas_S=f_meas_S,
            tau_meas_S=tau_meas_S,
            R_S_W=R_ft_ariaworld,
            m=0.490,                                  # your known/estimated mass
            c_S=np.array([0.021, 0.020, 0.059]),      # your CoG in sensor frame
            use_torque=False,                           # start with forces only
            smooth_len=15,                              # ~150 ms if 100 Hz
            k_thresh=1.0,
            min_free_sec=5,
            erode_sec=1.0,
        )

        # apply time window mask to measurements
        f_meas_S_contact_free = f_meas_S[mask]
        tau_meas_S_contact_free = tau_meas_S[mask]
        R_ft_ariaworld_contact_free = R_ft_ariaworld[mask]

        params = _estimate_tool_params_ls(
            F_meas_S=f_meas_S_contact_free,
            tau_meas_S=tau_meas_S_contact_free,
            R_S_W_list=R_ft_ariaworld_contact_free,
            m_known=0.490,
            c_S_known=np.array([0.021, 0.020, 0.059]),
            g=9.81
        )

        f_ext, tau_ext =compensate_wrench_batch(F_meas_S=f_meas_S,
                                                tau_meas_S=tau_meas_S,
                                                R_S_W= R_ft_ariaworld,
                                                params=params,)
        
        # add froces and torques to the DataFrame
        df_ft["wrench_ext.force.x"] = f_ext[:, 0]
        df_ft["wrench_ext.force.y"] = f_ext[:, 1]
        df_ft["wrench_ext.force.z"] = f_ext[:, 2]
        df_ft["wrench_ext.torque.x"] = tau_ext[:, 0]
        df_ft["wrench_ext.torque.y"] = tau_ext[:, 1]
        df_ft["wrench_ext.torque.z"] = tau_ext[:, 2]

        # filtering 
        CUTOFF_HZ = 15.0   # tune: ~10–25 Hz works well for F/T
        ORDER = 4
        # --- sampling rate from ns timestamps ---
        t = (df_ft["timestamp"].to_numpy() - df_ft["timestamp"].iloc[0]) * 1e-9
        dt = np.median(np.diff(t))
        fs = 1.0 / dt
        b, a = butter(ORDER, CUTOFF_HZ / (0.5 * fs), btype="low")

        for col in ["wrench_ext.force.x", "wrench_ext.force.y", "wrench_ext.force.z",
                    "wrench_ext.torque.x", "wrench_ext.torque.y", "wrench_ext.torque.z"]:
            df_ft[col + "_filt"] = filtfilt(b, a, df_ft[col].to_numpy())
            
        # save the compensated force-torque measurements
        file = self.extraction_path / ("/force_torque/ft_sensor0/ft_sensor_readings/wrench").strip("/") / "data.csv"

        df_ft.to_csv(file, index=False)

        if visualize:
            fig = plt.figure(figsize=(18, 8))

            # 1) Non-compensated forces
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(df_ft["timestamp"], df_ft["wrench.force.x"], label="Fx")
            ax1.plot(df_ft["timestamp"], df_ft["wrench.force.y"], label="Fy")
            ax1.plot(df_ft["timestamp"], df_ft["wrench.force.z"], label="Fz")
            ax1.set_title("Non-Compensated Forces"); ax1.set_xlabel("Timestamp (ns)"); ax1.set_ylabel("Force (N)"); ax1.legend()

            # 2) Compensated forces
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(df_ft["timestamp"], df_ft["wrench_ext.force.x"], label="Fx")
            ax2.plot(df_ft["timestamp"], df_ft["wrench_ext.force.y"], label="Fy")
            ax2.plot(df_ft["timestamp"], df_ft["wrench_ext.force.z"], label="Fz")
            ax2.set_title("Compensated Forces"); ax2.set_xlabel("Timestamp (ns)"); ax2.set_ylabel("Force (N)"); ax2.legend()

            # 3) Filtered forces
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(df_ft["timestamp"], df_ft["wrench_ext.force.x_filt"], label="Fx (filt)")
            ax3.plot(df_ft["timestamp"], df_ft["wrench_ext.force.y_filt"], label="Fy (filt)")
            ax3.plot(df_ft["timestamp"], df_ft["wrench_ext.force.z_filt"], label="Fz (filt)")
            ax3.set_title(f"Filtered Forces (Butter {ORDER}, {CUTOFF_HZ:g} Hz)"); ax3.set_xlabel("Timestamp (ns)"); ax3.set_ylabel("Force (N)"); ax3.legend()

            # 4) Non-compensated torques
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(df_ft["timestamp"], df_ft["wrench.torque.x"], label="Tx")
            ax4.plot(df_ft["timestamp"], df_ft["wrench.torque.y"], label="Ty")
            ax4.plot(df_ft["timestamp"], df_ft["wrench.torque.z"], label="Tz")
            ax4.set_title("Non-Compensated Torques"); ax4.set_xlabel("Timestamp (ns)"); ax4.set_ylabel("Torque (N·m)"); ax4.legend()

            # 5) Compensated torques
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(df_ft["timestamp"], df_ft["wrench_ext.torque.x"], label="Tx")
            ax5.plot(df_ft["timestamp"], df_ft["wrench_ext.torque.y"], label="Ty")
            ax5.plot(df_ft["timestamp"], df_ft["wrench_ext.torque.z"], label="Tz")
            ax5.set_title("Compensated Torques"); ax5.set_xlabel("Timestamp (ns)"); ax5.set_ylabel("Torque (N·m)"); ax5.legend()

            # 6) Filtered torques
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(df_ft["timestamp"], df_ft["wrench_ext.torque.x_filt"], label="Tx (filt)")
            ax6.plot(df_ft["timestamp"], df_ft["wrench_ext.torque.y_filt"], label="Ty (filt)")
            ax6.plot(df_ft["timestamp"], df_ft["wrench_ext.torque.z_filt"], label="Tz (filt)")
            ax6.set_title(f"Filtered Torques (Butter {ORDER}, {CUTOFF_HZ:g} Hz)"); ax6.set_xlabel("Timestamp (ns)"); ax6.set_ylabel("Torque (N·m)"); ax6.legend()

            # shade contact-free windows everywhere (uses your existing `windows`)
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                for (t0, t1) in windows:
                    ax.axvspan(t0, t1, alpha=0.15, linewidth=0)

            plt.tight_layout()
            plt.show()

        a = 2

    def transform_wrench_to_tool_and_express_in_world(self):
        
        # trajectory of aria on gripper in world frame
        trajectory_query = self.loader_aria_gripper.get_closed_loop_trajectory_aligned()
        trajectory_query = trajectory_query[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device", "qw_world_device", "qx_world_device", "qy_world_device", "qz_world_device"]]
    
        # forces and torques in sensor frame
        df_ft = self.get_force_torque_measurements()

        # get all necessary transforms
        T_imu_tool = self.calibration["imu0"]["T_imu_tool"]
        T_imu_sensor = self.calibration["imu0"]["T_imu_sensor"]
        T_device_camera = self.loader_aria_gripper.calibration["PINHOLE"]["T_device_camera"]
        T_cameraRaw_cameraRect = self.loader_aria_gripper.calibration["PINHOLE"]["pinhole_T_device_camera"]
        T_camera_imu = self.calibration["cam2"]["T_cam_imu"]
        T_tool_sensor = np.linalg.inv(T_imu_tool) @ T_imu_sensor
        T_sensor_tool = np.linalg.inv(T_tool_sensor)

        # get force torque origin in sensor frame
        T_device_sensor = T_device_camera @ T_camera_imu @ T_imu_sensor
        T_device_tool = T_device_camera @ T_camera_imu @ T_imu_tool
        
        # get clostest pose of aria in world frame at force torque timestamps
        # aria poses at 1000 Hz, force torque at 100 Hz
        timestamps_ns = df_ft["timestamp"].to_numpy(dtype=np.int64)
        R_world_device_list, t_world_device_list = _slerp_pose_series_to_targets(trajectory_query, timestamps_ns)
        R_world_device = np.array(R_world_device_list, dtype=np.float64)  # (N, 3, 3)
        t_world_device = np.array(t_world_device_list, dtype=np.float64)  # (N, 3)

        # T_world_tool = T_world_device @ T_device_tool

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