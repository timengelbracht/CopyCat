import cv2
from pathlib import Path
import telemetry_parser
from utils_mp4 import get_frames_from_mp4, get_imu_from_mp4
from utils_vrs import VRSUtils
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import numpy as np
from utils_bag import get_topics_from_bag
from qrcode_detector_decoder import QRCodeDetectorDecoder
from time_aligner import TimeAligner
from typing import List, Tuple
import cv2
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
from typing import Union, List, Tuple, Optional, Dict, Any
import sys
from rosbags.typesys.base import TypesysError
import pandas as pd
from utils_yaml import load_imucam, load_camchain
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def mp4_to_rosbag(mp4_path: Path | str, 
                bag_output_path: Path | str,
                cam_topic: str = "/cam0/image_raw",
                imu_topic: str = "/imu0",
                cam_frame_id: str = "cam0",
                imu_frame_id: str = "imu0",) -> None:
    """
    Convert MP4 files to ROS bag format (for Kalibr)
    """

    imu_df = get_imu_from_mp4(mp4_path)
    frames_av, timestamps_ns = get_frames_from_mp4(mp4_path)
    
    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header    = typestore.types['std_msgs/msg/Header']
    ImageMsg  = typestore.types['sensor_msgs/msg/Image']
    ImuMsg    = typestore.types['sensor_msgs/msg/Imu']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']

    bag_path = Path(bag_output_path)
    with Writer(bag_path) as writer:

        # Register both connections once
        con_cam = writer.add_connection(
            cam_topic, ImageMsg.__msgtype__, typestore=typestore)
        con_imu = writer.add_connection(
            imu_topic, ImuMsg.__msgtype__, typestore=typestore)

        events = []

        # ---------- IMU ----------------------------------------------------
        default_covariance = np.zeros(9, dtype=np.float64)                         # list, not np.array
        default_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        for seq, row in tqdm(enumerate(imu_df.itertuples(index=False)),
                            total=len(imu_df), desc='IMU', unit='sample'):
            t_ns = int(row.timestamp)
            stamp = Time(sec=t_ns // 1_000_000_000,
                        nanosec=t_ns % 1_000_000_000)
            header = Header(seq=seq, stamp=stamp, frame_id=imu_frame_id)
            imu_msg = ImuMsg(
                header=header,
                orientation=default_orientation,
                orientation_covariance=default_covariance,
                angular_velocity=Vector3(x=row.angular_vel_x, y=row.angular_vel_y, z=row.angular_vel_z),                
                angular_velocity_covariance=default_covariance,
                linear_acceleration=Vector3(x=row.linear_accel_x, y=row.linear_accel_y, z=row.linear_accel_z),
                linear_acceleration_covariance=default_covariance,
            )

            raw = typestore.serialize_ros1(imu_msg, ImuMsg.__msgtype__)
            events.append((t_ns, con_imu, raw))

        # ---------- Images -------------------------------------------------
        for seq, (frame, t_ns) in enumerate(
                tqdm(zip(frames_av, timestamps_ns),
                    total=len(timestamps_ns), desc='Images', unit='frame')):

            bgr = frame.to_ndarray(format='bgr24')
            h, w = bgr.shape[:2]
            stamp = Time(sec=t_ns // 1_000_000_000,
                        nanosec=t_ns % 1_000_000_000)
            header = Header(seq=seq, stamp=stamp, frame_id=cam_frame_id)
            img_msg = ImageMsg(
                header=header, height=h, width=w,
                encoding='bgr8', is_bigendian=0, step=3 * w,
                data=bgr.reshape(-1),            # ndarray view, not bytes
            )

            raw = typestore.serialize_ros1(img_msg, ImageMsg.__msgtype__)
            events.append((t_ns, con_cam, raw))

        events.sort(key=lambda e: e[0])   # sort by timestamp

        for t_ns, con, raw in tqdm(events, desc="Writing events", unit="event"):
            writer.write(con, t_ns, raw)

    print(f"[rosbags] wrote {len(timestamps_ns)} images and {len(imu_df)} IMU "
          f"samples → {bag_path}")
    
def merge_calibration_vrs_and_calibration_bag(vrs_path: Path | str, 
                                            rosbag_path: Path | str,
                                            temp_path: Path | str) -> None:
    """
    Merges the calibration data from VRS and the ROS bag.
    Background: Aria vrs and gripper back are jointly recorded, with the aria mounted on the gripper.
    First tiemstamps for the aria need to be adjusted to match the gripper timestamps by detecting
    the timestamped qr code. then aria is added to rosbag for later Kalibr calibration.
    1. Extract frames abnd timestamps from vrs into dummy directory
    2. Extract frames and timestamps from rosbag
    3. Detect the timestamped qr code in the aria frames and gripper frames, compoute offset
    4. Adjust the aria timestamps by the offset
    5. Write the adjusted aria frames and timestamps to the rosbag
    """
    
    if isinstance(vrs_path, str):
        vrs_path = Path(vrs_path)
    if isinstance(rosbag_path, str):
        rosbag_path = Path(rosbag_path)
    if isinstance(temp_path, str):
        temp_path = Path(temp_path)

    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")
    
    if not rosbag_path.exists():
        raise FileNotFoundError(f"ROS bag file not found: {rosbag_path}")
    
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    temp_path_rosbag = temp_path / "rosbag"
    temp_path_vrs = temp_path / "vrs"
    temp_path_rosbag.mkdir(parents=True, exist_ok=True)
    temp_path_vrs.mkdir(parents=True, exist_ok=True)

    new_cam_topic = "/aria/camera_rgb/image_raw"
    new_cam_frame_id = "aria_camera_rgb"
    
    # Extract frames and timestamps from VRS into a temporary directory
    if not any(temp_path_vrs.glob("*")):
        vrs_utils = VRSUtils(vrs_path, undistort=False)
        _, _ = vrs_utils.get_frames_from_vrs(out_dir=temp_path_vrs)

    # Extract frames and timestamps from ROS bag
    if not any(temp_path_rosbag.glob("*")):
        get_topics_from_bag(
            image_topics=["/zedm/zed_node/left_raw/image_raw_color"],
            non_image_topics={},
            bag_path=rosbag_path,
            out_dir=temp_path_rosbag
        )

    # Detect the timestamped QR code in the VRS frames
    qr = QRCodeDetectorDecoder(frame_dir=temp_path_vrs, ext=".png")
    time_pair_aria = qr.find_first_valid_qr()

    # Detect the timestamped QR code in the ROS bag frames
    frame_dir = temp_path_rosbag / "zedm/zed_node/left_raw/image_raw_color"
    qr = QRCodeDetectorDecoder(frame_dir=frame_dir, ext=".png")
    time_pair_gripper = qr.find_first_valid_qr()

    # get the offset between the two timestamps
    if time_pair_aria is None or time_pair_gripper is None:
        raise ValueError("Could not find valid QR codes in either VRS or ROS bag frames.")
    
    # flip time pairs, so we get aria delta to gripper 
    # (unlike in data extraction, where we had gripper delta to aria)
    timealigner = TimeAligner(
        aria_pair=time_pair_gripper,
        sensor_pair=time_pair_aria,
    )
    delta = timealigner.get_delta()

    # Adjust the timestamps of the VRS frames by the delta
    for frame in temp_path_vrs.glob("*.png"):
        ts = int(frame.stem)
        adjusted_ts = ts + delta
        new_frame_name = temp_path_vrs / f"{adjusted_ts}.png"
        frame.rename(new_frame_name)

    # Write the adjusted VRS frames and timestamps to the ROS bag
    # First read the old rosbag into all_events, adding the adjusted VRS frames and sort
    output_bag_path = temp_path / "merged_calibration.bag"

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types['std_msgs/msg/Header']
    ImageMsg = typestore.types['sensor_msgs/msg/Image']
    Time = typestore.types['builtin_interfaces/msg/Time']


    all_events: List[Tuple[int, object, bytes]] = []

    # --- 1. Read existing messages from the original bag ---
    print(f"Reading existing messages from {rosbag_path}...")
    with Reader(rosbag_path) as reader:
        for connection, timestamp_ns, rawdata in tqdm(reader.messages(), desc="Reading existing bag"):
            all_events.append((timestamp_ns, connection, rawdata))
    
    # --- 2. Add new images from the VRS frames ---
    vrs_files = sorted(list(temp_path_vrs.glob("*.png")))
    for seq, img_file in enumerate(tqdm(vrs_files, desc="Processing new images")):
        try:
            # Extract timestamp from filename (assuming integer timestamp)
            t_ns = int(img_file.stem) # .stem gets the filename without suffix

            # Load image using OpenCV
            bgr_image = cv2.imread(str(img_file))
            if bgr_image is None:
                print(f"Warning: Could not read image {img_file}. Skipping.", file=sys.stderr)
                continue

            h, w = bgr_image.shape[:2]
            
            # Create ROS 1 Time object
            stamp = Time(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000)
            
            # Create ROS 1 Header object
            header = Header(seq=seq, stamp=stamp, frame_id=new_cam_frame_id)
            
            # Create ROS 1 Image message
            img_msg = ImageMsg(
                header=header,
                height=h,
                width=w,
                encoding='bgr8', # OpenCV reads as BGR, so 'bgr8' is appropriate
                is_bigendian=0,
                step=3 * w, # 3 bytes per pixel (BGR) * width
                data=bgr_image.reshape(-1) # Flatten the numpy array to bytes
            )

            # Serialize message to raw bytes
            raw = typestore.serialize_ros1(img_msg, ImageMsg.__msgtype__)
            all_events.append((t_ns, new_cam_topic, raw)) # Store topic name for new connection

        except ValueError:
            print(f"Warning: Skipping {img_file} as its name is not a valid integer timestamp.", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {img_file}: {e}. Skipping.", file=sys.stderr)

    # --- 3. Sort all events by timstamp ---
    all_events.sort(key=lambda e: e[0])

    # --- 4. Write all events to the new bag file ---
    print(f"Writing all messages to new bag file: {output_bag_path}...")
    with Writer(output_bag_path) as writer:
        # Keep track of connections for existing topics
        existing_connections = {}
        new_cam_connection = None

        for t_ns, con_or_topic, raw in tqdm(all_events, desc="Writing events to new bag"):
            if isinstance(con_or_topic, str): # This is a new camera topic
                if new_cam_connection is None:
                    new_cam_connection = writer.add_connection(
                        new_cam_topic, ImageMsg.__msgtype__, typestore=typestore
                    )
                writer.write(new_cam_connection, t_ns, raw)
            else: # This is an existing connection from the original bag
                # Add connection if not already added (e.g., first message of a topic)
                if con_or_topic.topic not in existing_connections:
                    try:
                        existing_connections[con_or_topic.topic] = writer.add_connection(
                            con_or_topic.topic, con_or_topic.msgtype, typestore=typestore
                        )
                    except TypesysError as e:
                        print(f"Warning: Skipping topic '{con_or_topic.topic}' due to unknown type '{con_or_topic.msgtype}': {e}", file=sys.stderr)
                        # Mark this connection as None to indicate it was skipped
                        existing_connections[con_or_topic.topic] = None
                        continue # Skip writing this specific message

                # Only write if the connection was successfully added (not None)
                if existing_connections[con_or_topic.topic] is not None:
                    writer.write(existing_connections[con_or_topic.topic], t_ns, raw)


def estimate_tool_params(vrs_path: Path | str,
                        mps_file: Path | str, 
                        rosbag_path: Path | str,
                        temp_path: Path | str,
                        camchain_imucam: Path,
                        forceless_time_intervall: List[int]) -> None:
    """
    Estimate tool mass parameters to later compensate the tool from the force/torque readings.
    Background: Aria vrs and gripper back are jointly recorded, with the aria mounted on the gripper.
    First tiemstamps for the aria need to be adjusted to match the gripper timestamps by detecting
    the timestamped qr code. mps is extracted and timestamps are again adjusted by the offset.
    1. Extract frames abnd timestamps from vrs into dummy directory
    2. Extract frames and timestamps from rosbag
    3. Detect the timestamped qr code in the aria frames and gripper frames, compoute offset
    4. Adjust the aria timestamps (and mps) by the offset. This brings forcec/torque readings and 
       slam poses into the same time frame.
    5. load transform between force/torque and aria slam poses from prior calibration
    6. cut off part of the recording where the gripper touches the floor (external force, do it manually 
       by looing for the first frame when the gripper is lifted and the last frame when the gripper
       is lowered)
    6. compute the tool parameters from the force/torque readings and slam poses.
    """
    

    if isinstance(vrs_path, str):
        vrs_path = Path(vrs_path)
    if isinstance(rosbag_path, str):
        rosbag_path = Path(rosbag_path)
    if isinstance(temp_path, str):
        temp_path = Path(temp_path)
    if isinstance(mps_file, str):
        mps_file = Path(mps_file)
    if isinstance(camcahin_imucam, str):
        camchain_imucam = Path(camcahin_imucam)

    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")
    
    if not rosbag_path.exists():
        raise FileNotFoundError(f"ROS bag file not found: {rosbag_path}")
    
    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    if not mps_file.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_file}")
    
    if not camchain_imucam.exists():
        raise FileNotFoundError(f"Camera-IMU calibration file not found: {camchain_imucam}")

    # Load camera-IMU calibration
    cam_calib = load_camchain(camchain_imucam, cam_name="cam2")
    imu_calib = load_imucam(camchain_imucam, imu_name="cam2")

    temp_path_rosbag = temp_path / "rosbag"
    temp_path_vrs = temp_path / "vrs"
    temp_path_rosbag.mkdir(parents=True, exist_ok=True)
    temp_path_vrs.mkdir(parents=True, exist_ok=True)

    # Load the MPS file
    if not mps_file.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_file}")
    mps_data = pd.read_csv(mps_file)
    
    # Extract frames and timestamps from VRS into a temporary directory
    vrs_utils = VRSUtils(vrs_path, undistort=False)
    if not any(temp_path_vrs.glob("*")):
        _, _ = vrs_utils.get_frames_from_vrs(out_dir=temp_path_vrs)

    # Extract frames and timestamps from ROS bag
    force_torque_topic = "/force_torque/ft_sensor0/ft_sensor_readings/wrench"
    temperature_topic = "/force_torque/ft_sensor0/ft_sensor_readings/temperature"
    if not any(temp_path_rosbag.glob("*")):
        get_topics_from_bag(
            image_topics=["/zedm/zed_node/left_raw/image_raw_color"],
            non_image_topics={force_torque_topic: "geometry_msgs/WrenchStamped",
                              temperature_topic: "sensor_msgs/Temperature"},
            bag_path=rosbag_path,
            out_dir=temp_path_rosbag
        )

    adjusted_mps_file = temp_path / "adjusted_mps.csv"
    if not adjusted_mps_file.exists():
        # Detect the timestamped QR code in the VRS frames
        qr = QRCodeDetectorDecoder(frame_dir=temp_path_vrs, ext=".png")
        time_pair_aria = qr.find_first_valid_qr()

        # Detect the timestamped QR code in the ROS bag frames
        frame_dir = temp_path_rosbag / "zedm/zed_node/left_raw/image_raw_color"
        qr = QRCodeDetectorDecoder(frame_dir=frame_dir, ext=".png")
        time_pair_gripper = qr.find_first_valid_qr()

        # get the offset between the two timestamps
        if time_pair_aria is None or time_pair_gripper is None:
            raise ValueError("Could not find valid QR codes in either VRS or ROS bag frames.")
        
        # flip time pairs, so we get aria delta to gripper 
        # (unlike in data extraction, where we had gripper delta to aria)
        timealigner = TimeAligner(
            aria_pair=time_pair_gripper,
            sensor_pair=time_pair_aria,
        )
        delta = timealigner.get_delta()
        print(f"Time delta between VRS and ROS bag: {delta} ns")

        # Adjust the timestamps of the VRS frames by the delta
        for frame in temp_path_vrs.glob("*.png"):
            ts = int(frame.stem)
            adjusted_ts = ts + delta
            new_frame_name = temp_path_vrs / f"{adjusted_ts}.png"
            frame.rename(new_frame_name)

        # Adjust the MPS timestamps by the delta
        adjusted_mps_data = mps_data.copy()
        adjusted_mps_data["tracking_timestamp_us"] = (adjusted_mps_data["tracking_timestamp_us"] * 1_000) + delta
        #rename the timestamp column to match the adjusted format
        adjusted_mps_data.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)
        adjusted_mps_data.to_csv(adjusted_mps_file, index=False)
    else:
        adjusted_mps_data = pd.read_csv(adjusted_mps_file)
    
    # read force/torque readings from the rosbag    
    force_torque_df = pd.read_csv(temp_path_rosbag / force_torque_topic.strip("/") / "data.csv")
    temperature_df = pd.read_csv(temp_path_rosbag / temperature_topic.strip("/") / "data.csv")
    # cut off part of the recording where the gripper touches the floor
    t_start = forceless_time_intervall[0]
    t_end = forceless_time_intervall[1]

    force_torque_df_cut = force_torque_df[
        (force_torque_df["timestamp"] >= t_start) & 
        (force_torque_df["timestamp"] <= t_end)
    ]
    adjusted_mps_data_cut = adjusted_mps_data[
        (adjusted_mps_data["timestamp"] >= t_start) & 
        (adjusted_mps_data["timestamp"] <= t_end)
    ]

    temperature_df_cut = temperature_df[
        (temperature_df["timestamp"] >= t_start) &
        (temperature_df["timestamp"] <= t_end)
    ]

    # load transform from cam aria to force/torque sensor
    T_ariacam_imuft = imu_calib.T_cam_imu

    # trafo from imu of forcetorque to forcetorque frame (offset)
    # exampple: (0,0,0) in ft frame is (0,0,0.0257) in imu frame
    T_imuft_ft = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.0257],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    ariacam_calibration = vrs_utils.device_calib.get_camera_calib("camera-rgb")
    T_ariadevice_ariacam = ariacam_calibration.get_transform_device_camera().to_matrix()

    # wrench in ft frame (sensor)
    wrench_ft = force_torque_df_cut[["timestamp", "wrench.force.x", "wrench.force.y", "wrench.force.z",
                                      "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]]
    wrench_timestamps_ns = wrench_ft["timestamp"].to_numpy(dtype=np.int64)
    temperature_ft = temperature_df_cut[["timestamp", "temperature"]]

    # aria slam poses in aria world frame
    poses_aria = adjusted_mps_data_cut[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device",
                                        "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]]
    
    T_ariadevice_ft = T_ariadevice_ariacam @ T_ariacam_imuft @ T_imuft_ft

    # slerp interpolate poeses_aria to get the poses at the timestamps of the wrench_ft
    R_ariaworld_ariadevice_list, t_ariaworld_ariadevice_list = _slerp_pose_series_to_targets(poses_aria, wrench_timestamps_ns)

    # convert to numpy arrays
    R_ariaworld_ariadevice = np.array(R_ariaworld_ariadevice_list, dtype=np.float64)  # (N, 3, 3)
    R_ariadevice_ft = T_ariadevice_ft[:3, :3]  # (3, 3)

    # compute the rotation from aria world to force/torque sensor frame
    R_ariaworld_ft = R_ariaworld_ariadevice @ R_ariadevice_ft  # (N, 3, 3)
    R_ft_ariaworld = np.transpose(R_ariaworld_ft, axes=(0, 2, 1))  # (N, 3, 3)


    f_meas_S = wrench_ft[["wrench.force.x", "wrench.force.y", "wrench.force.z"]].to_numpy(dtype=np.float64)  # (N, 3)
    tau_meas_S = wrench_ft[["wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]].to_numpy(dtype=np.float64)
    T_meas_S = temperature_ft["temperature"].to_numpy(dtype=np.float64)  # (N,)

    params = _estimate_tool_params_ls(
        F_meas_S=f_meas_S,
        tau_meas_S=tau_meas_S,
        R_S_W_list=R_ft_ariaworld,
        m_known=0.490,
        g=9.81
    )

    # save the parameters to a file
    params_file = temp_path / "tool_params_static_estimate.json"
    with open(params_file, 'w') as f:
        import json
        json.dump(params, f, indent=4)

    print(f"Tool parameters estimated and saved to {params_file}")
    print(f"Mass: {params['m']} kg")
    print(f"Center of mass in sensor frame: {params['c_S']}")
    print(f"Force offset: {params['f0']}")
    print(f"Torque offset: {params['tau0']}")
    print(f"RMS Force error: {params['rmsF']}")
    print(f"RMS Torque error: {params['rmsT']}")


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)


# def _estimate_tool_params_ls(F_meas_S, tau_meas_S, R_S_W_list, g=9.81):
#     """
#     Linear LS on quasi-static model:
#       F = f0 + m * g_S
#       tau = tau0 - [g_S]_x * p,   p = m c_S
#     Unknowns theta = [f0(3), m(1), p(3), tau0(3)] -> 10 params
#     """
#     N = F_meas_S.shape[0]
#     gW = np.array([0,0,-g], dtype=float)

#     rows = []
#     ys   = []
#     for i in range(N):
#         g_S = R_S_W_list[i] @ gW

#         AF = np.zeros((3,10))
#         AF[:, 0:3] = np.eye(3)        # f0
#         AF[:, 3:4] = g_S.reshape(3,1) # m
#         yF = F_meas_S[i]

#         AT = np.zeros((3,10))
#         AT[:, 4:7] = -_skew(g_S)      # p
#         AT[:, 7:10]= np.eye(3)        # tau0
#         yT = tau_meas_S[i]

#         rows.append(AF); ys.append(yF)
#         rows.append(AT); ys.append(yT)

#     A = np.vstack(rows)         # (6N,10)
#     y = np.hstack(ys)           # (6N,)

#     lam = 1e-6
#     ATA = A.T @ A + lam*np.eye(10)
#     ATy = A.T @ y
#     theta = np.linalg.solve(ATA, ATy)

#     f0   = theta[0:3]
#     m    = float(theta[3])
#     p    = theta[4:7]
#     tau0 = theta[7:10]
#     c_S  = p / m if abs(m) > 1e-9 else np.zeros(3)

#     # diagnostics
#     yhat = A @ theta
#     res  = y - yhat
#     res6 = res.reshape(-1,6)
#     rmsF = float(np.sqrt(np.mean(np.sum(res6[:,0:3]**2, axis=1))))
#     rmsT = float(np.sqrt(np.mean(np.sum(res6[:,3:6]**2, axis=1))))

#     return dict(m=m, c_S=c_S.tolist(), f0=f0.tolist(), tau0=tau0.tolist(), rmsF=rmsF, rmsT=rmsT, theta=theta.tolist())

def _estimate_tool_params_ls(
    F_meas_S,
    tau_meas_S,
    R_S_W_list,
    g: float = 9.81,
    m_known: float | None = None,
    c_S_known: np.ndarray | None = None,
):
    """
    Linear LS on quasi-static model with optional known mass and/or CoG.

    Model:
      F   = f0 + m * g_S
      tau = tau0 - [g_S]_x * (m * c_S)      (with p := m c_S)

    Unknowns depend on what is known:
      - If m and c_S unknown:           theta = [f0(3), m(1), p(3), tau0(3)]          -> 10 params (original)
      - If m known, c_S unknown:        theta = [f0(3), c_S(3), tau0(3)]              ->  9 params
      - If m unknown, c_S known:        theta = [f0(3), m(1), tau0(3)]                ->  7 params
      - If both m and c_S known:        theta = [f0(3), tau0(3)]                      ->  6 params

    Inputs:
      F_meas_S      : (N,3) forces in sensor frame S
      tau_meas_S    : (N,3) torques in sensor frame S
      R_S_W_list    : (N,3,3) rotations world->sensor per sample
      g             : gravity magnitude
      m_known       : float or None
      c_S_known     : (3,) or None  (CoG in sensor frame S, meters)

    Returns dict with:
      m, c_S, f0, tau0, rmsF, rmsT, theta (all lists except scalars), plus bookkeeping.
    """
    N = F_meas_S.shape[0]
    assert tau_meas_S.shape[0] == N and R_S_W_list.shape[0] == N
    gW = np.array([0.0, 0.0, -g], dtype=float)

    # Normalize c_S_known shape if provided
    if c_S_known is not None:
        c_S_known = np.asarray(c_S_known, dtype=float).reshape(3,)

    # Helper to build skew-symmetric matrix
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([[0.0, -z,   y],
                         [z,    0.0, -x],
                         [-y,   x,   0.0]], dtype=float)

    rows = []
    ys   = []

    # We’ll assemble a design matrix with named blocks so we can parse theta cleanly.
    col_index = 0
    cols = {}  # name -> (start, length)

    def add_cols(name, ncols):
        nonlocal col_index
        cols[name] = (col_index, ncols)
        col_index += ncols

    # Decide which blocks are present based on known/unknown settings
    # f0 and tau0 are ALWAYS estimated
    add_cols("f0",   3)

    if m_known is None and c_S_known is None:
        # Original: unknown m and p = m c_S
        add_cols("m",    1)
        add_cols("p",    3)   # first moment
        add_cols("tau0", 3)
        param_count = col_index
        for i in range(N):
            g_S = R_S_W_list[i] @ gW

            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            i_m,  _ = cols["m"]
            AF[:, i_f0:i_f0+3] = np.eye(3)            # f0
            AF[:, i_m:i_m+1]   = g_S.reshape(3,1)     # m
            rows.append(AF); ys.append(F_meas_S[i])

            AT = np.zeros((3, param_count))
            i_p, _    = cols["p"]
            i_tau0, _ = cols["tau0"]
            AT[:, i_p:i_p+3]      = -_skew(g_S)       # p
            AT[:, i_tau0:i_tau0+3]= np.eye(3)         # tau0
            rows.append(AT); ys.append(tau_meas_S[i])

    elif (m_known is not None) and (c_S_known is None):
        # Known mass, unknown c_S
        add_cols("c_S",  3)
        add_cols("tau0", 3)
        param_count = col_index
        m = float(m_known)

        for i in range(N):
            g_S = R_S_W_list[i] @ gW

            # Forces: F = f0 + m * g_S  -> move known gravity to RHS
            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            AF[:, i_f0:i_f0+3] = np.eye(3)            # f0
            rows.append(AF); ys.append(F_meas_S[i] - m * g_S)

            # Torques: tau = tau0 - m [g_S]_x c_S
            AT = np.zeros((3, param_count))
            i_cS, _   = cols["c_S"]
            i_tau0, _ = cols["tau0"]
            AT[:, i_cS:i_cS+3]    = -m * _skew(g_S)   # c_S
            AT[:, i_tau0:i_tau0+3]= np.eye(3)         # tau0
            rows.append(AT); ys.append(tau_meas_S[i])

    elif (m_known is None) and (c_S_known is not None):
        # Unknown mass, known c_S
        add_cols("m",    1)
        add_cols("tau0", 3)
        param_count = col_index
        cS = c_S_known

        for i in range(N):
            g_S = R_S_W_list[i] @ gW

            # Forces: F = f0 + m * g_S
            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            i_m,  _ = cols["m"]
            AF[:, i_f0:i_f0+3] = np.eye(3)            # f0
            AF[:, i_m:i_m+1]   = g_S.reshape(3,1)     # m
            rows.append(AF); ys.append(F_meas_S[i])

            # Torques: tau = tau0 - m [g_S]_x c_S  -> torque is linear in m with vector -(g_S × c_S)
            AT = np.zeros((3, param_count))
            i_m,  _    = cols["m"]
            i_tau0, _  = cols["tau0"]
            vec = -(_skew(g_S) @ cS).reshape(3,1)     # 3x1 column multiplying m
            AT[:, i_m:i_m+1]     = vec               # m
            AT[:, i_tau0:i_tau0+3]= np.eye(3)         # tau0
            rows.append(AT); ys.append(tau_meas_S[i])

    else:
        # Both m and c_S known: estimate only f0 and tau0
        add_cols("tau0", 3)
        param_count = col_index
        m = float(m_known)
        cS = c_S_known

        for i in range(N):
            g_S = R_S_W_list[i] @ gW
            # Forces: F - m g_S = f0
            AF = np.zeros((3, param_count))
            i_f0, _ = cols["f0"]
            AF[:, i_f0:i_f0+3] = np.eye(3)
            rows.append(AF); ys.append(F_meas_S[i] - m * g_S)

            # Torques: tau - ( - [g_S]_x (m c_S) ) = tau0
            AT = np.zeros((3, param_count))
            i_tau0, _ = cols["tau0"]
            AT[:, i_tau0:i_tau0+3] = np.eye(3)
            tau_g = -_skew(g_S) @ (m * cS)
            rows.append(AT); ys.append(tau_meas_S[i] - tau_g)

    # Stack and solve
    A = np.vstack(rows)          # (6N, P)
    y = np.hstack(ys)            # (6N,)
    lam = 1e-6
    ATA = A.T @ A + lam * np.eye(A.shape[1])
    ATy = A.T @ y
    theta = np.linalg.solve(ATA, ATy)

    # Parse solution into outputs
    out = {}
    # f0
    i, L = cols["f0"];  f0 = theta[i:i+L]
    out["f0"] = f0.tolist()

    if m_known is None and c_S_known is None:
        i_m, _ = cols["m"]; m_est = float(theta[i_m])
        i_p, _ = cols["p"]; p = theta[i_p:i_p+3]
        cS_est = (p / m_est) if abs(m_est) > 1e-9 else np.zeros(3)
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    elif (m_known is not None) and (c_S_known is None):
        m_est = float(m_known)
        i_cS,_ = cols["c_S"]; cS_est = theta[i_cS:i_cS+3]
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    elif (m_known is None) and (c_S_known is not None):
        cS_est = c_S_known
        i_m,_  = cols["m"]; m_est = float(theta[i_m])
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    else:
        # both known
        m_est = float(m_known)
        cS_est = c_S_known
        i_t0,_ = cols["tau0"]; tau0 = theta[i_t0:i_t0+3]
        out.update(m=m_est, c_S=cS_est.tolist(), tau0=tau0.tolist())

    # Diagnostics: compute residuals
    # Reconstruct predictions using the final (m, c_S, f0, tau0)
    m_use  = out["m"]
    cS_use = np.asarray(out["c_S"], float)
    tau0   = np.asarray(out["tau0"], float)
    f0     = np.asarray(out["f0"], float)

    F_pred = np.empty_like(F_meas_S)
    T_pred = np.empty_like(tau_meas_S)
    for i in range(N):
        g_S = R_S_W_list[i] @ gW
        F_pred[i] = f0 + m_use * g_S
        tau_g = -_skew(g_S) @ (m_use * cS_use)
        T_pred[i] = tau0 + tau_g

    rmsF = float(np.sqrt(np.mean(np.sum((F_meas_S - F_pred)**2, axis=1))))
    rmsT = float(np.sqrt(np.mean(np.sum((tau_meas_S - T_pred)**2, axis=1))))

    out.update(rmsF=rmsF, rmsT=rmsT, theta=theta.tolist())
    return out


def _slerp_pose_series_to_targets(poses_df: pd.DataFrame, target_ts_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    poses_df columns: ['timestamp','tx_world_device','ty_world_device','tz_world_device',
                       'qx_world_device','qy_world_device','qz_world_device','qw_world_device']
    Returns arrays aligned to target_ts_ns:
      R_W_D_list (N,3,3), t_W_D_list (N,3)
    """
    if len(poses_df) < 2:
        raise ValueError("Not enough poses for interpolation")

    # sort and unique
    poses_df = poses_df.sort_values('timestamp').drop_duplicates('timestamp')

    t_ref = poses_df['timestamp'].to_numpy(dtype=np.int64)
    pos   = poses_df[['tx_world_device','ty_world_device','tz_world_device']].to_numpy(dtype=float)
    quat  = poses_df[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy(dtype=float)

    # Normalize quaternions (format [x,y,z,w])
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    # Clamp targets to [t0, tf]
    t0 = t_ref[0]
    tf = t_ref[-1]
    if tf == t0:
        raise ValueError("Pose timestamps are constant")
    ts = np.clip(target_ts_ns.astype(np.int64), t0, tf)

    # For each target, find surrounding indices
    idx_right = np.searchsorted(t_ref, ts, side='left')
    idx_left  = np.clip(idx_right - 1, 0, len(t_ref)-1)
    idx_right = np.clip(idx_right,     0, len(t_ref)-1)

    # avoid identical indices (degenerate): push apart
    same = (idx_left == idx_right)
    idx_right[same] = np.clip(idx_right[same] + 1, 0, len(t_ref)-1)
    idx_left[same]  = np.clip(idx_left[same] - 1, 0, len(t_ref)-1)

    tL = t_ref[idx_left].astype(np.float64)
    tR = t_ref[idx_right].astype(np.float64)
    w  = np.zeros_like(ts, dtype=np.float64)
    mask = (tR != tL)
    w[mask] = (ts[mask] - tL[mask]) / (tR[mask] - tL[mask])
    w = np.clip(w, 0.0, 1.0)

    # ----- Vectorized SLERP between quat[idx_left] and quat[idx_right] with weight w -----
    qL = quat[idx_left]    # (N,4)
    qR = quat[idx_right]   # (N,4)

    # Ensure shortest path
    dot = np.sum(qL * qR, axis=1)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    qR = qR * sign[:, None]
    dot = np.abs(dot)

    # Avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    # Where angle is tiny, use LERP; else SLERP
    eps = 1e-8
    use_lerp = sin_theta < eps

    s0 = np.empty_like(w, dtype=float)
    s1 = np.empty_like(w, dtype=float)

    # SLERP weights
    idx = ~use_lerp
    s0[idx] = np.sin((1.0 - w[idx]) * theta[idx]) / sin_theta[idx]
    s1[idx] = np.sin(w[idx] * theta[idx]) / sin_theta[idx]

    # LERP fallback
    s0[use_lerp] = 1.0 - w[use_lerp]
    s1[use_lerp] = w[use_lerp]

    q_interp = (s0[:, None] * qL) + (s1[:, None] * qR)
    q_interp = q_interp / np.linalg.norm(q_interp, axis=1, keepdims=True)

    # Convert to rotation matrices
    R_W_D_list = R.from_quat(q_interp).as_matrix()

    # LERP translations
    pL = pos[idx_left]
    pR = pos[idx_right]
    t_W_D_list = (1.0 - w)[:, None] * pL + w[:, None] * pR

    return R_W_D_list, t_W_D_list


def compensate_wrench_batch(F_meas_S, tau_meas_S, R_S_W, params, g=9.81):
    # F_meas_S: (N,3), tau_meas_S: (N,3), R_S_W: (N,3,3)
    gW = np.array([0,0,-g], float)
    m    = params["m"]
    c_S  = params["c_S"]
    f0   = params["f0"]
    tau0 = params["tau0"]

    g_S = R_S_W @ gW                   # (N,3)
    Fg  = m * g_S                      # (N,3)
    taug = np.cross(np.broadcast_to(c_S, Fg.shape), Fg)

    F_contact   = F_meas_S  - f0   - Fg
    tau_contact = tau_meas_S - tau0 - taug
    return F_contact, tau_contact


def find_contact_free_segments(
    timestamps_ns: np.ndarray,   # (N,)
    F_meas_S: np.ndarray,        # (N,3)
    tau_meas_S: np.ndarray,      # (N,3)
    R_S_W: np.ndarray,           # (N,3,3)  world->sensor
    *,
    m: float,                    # known (or from prior offline est.)
    c_S: np.ndarray,             # (3,) known CoG in sensor frame
    g: float = 9.81,
    use_torque: bool = True,     # True: include torque residuals in the score
    smooth_len: int = 15,        # odd; moving median length (samples)
    k_thresh: float = 1.0,       # robust threshold multiplier (MAD)
    min_free_sec: float = 10.0,  # minimum window duration to keep (before erosion)
    erode_sec: float = 0.0,      # erode each free window by this much on both sides
) -> Tuple[List[Tuple[int,int]], np.ndarray, Dict[str,Any]]:
    """
    Contact-free detector with optional erosion:
      - Enforces min_free_sec first
      - Then erodes each window by erode_sec on both sides
    """
    N = len(timestamps_ns)
    assert F_meas_S.shape == (N,3) and tau_meas_S.shape == (N,3) and R_S_W.shape == (N,3,3)
    t = timestamps_ns.astype(np.int64)

    # 1) Gravity terms in sensor frame
    gW = np.array([0,0,-g], float)
    g_S = (R_S_W @ gW).reshape(N,3)           # (N,3)
    Fg  = m * g_S
    taug= np.cross(np.broadcast_to(c_S, Fg.shape), Fg)  # (N,3)

    # 2) Global bias
    f0   = np.median(F_meas_S - Fg, axis=0)
    tau0 = np.median(tau_meas_S - taug, axis=0)

    # 3) Residuals
    F_res   = F_meas_S  - (f0 + Fg)
    tau_res = tau_meas_S - (tau0 + taug)

    # scalar score per-sample
    if use_torque:
        L = 0.1
        r = np.sqrt(np.sum(F_res**2, axis=1) + np.sum((tau_res / L)**2, axis=1))
    else:
        r = np.linalg.norm(F_res, axis=1)

    # 4) Smooth
    if smooth_len > 1:
        k = max(1, int(smooth_len) | 1)
        pad = k // 2
        r_pad = np.pad(r, (pad, pad), mode="edge")
        r_s  = np.empty_like(r)
        for i in range(N):
            r_s[i] = np.median(r_pad[i:i+k])
    else:
        r_s = r

    # 5) Threshold
    med = float(np.median(r_s))
    mad = float(np.median(np.abs(r_s - med)) or 1e-9)
    thr = med + k_thresh * 1.4826 * mad
    free_mask = r_s <= thr

    # 6) Merge to windows
    Ts = float(np.median(np.diff(t))) / 1e9 if N > 1 else 0.01
    windows: List[Tuple[int,int]] = []
    i = 0
    while i < N:
        if free_mask[i]:
            j = i + 1
            while j < N and free_mask[j]:
                j += 1
            t0 = int(t[i])
            t1 = int(t[j-1] + max(1, int(round(Ts*1e9))))

            # enforce min duration BEFORE erosion
            if (t1 - t0)/1e9 >= min_free_sec:
                if erode_sec > 0:
                    margin = int(round(erode_sec / Ts))
                    if (j - i) > 2*margin:
                        i_e = i + margin
                        j_e = j - margin
                        t0 = int(t[i_e])
                        t1 = int(t[j_e-1] + max(1, int(round(Ts*1e9))))
                    else:
                        # window too short after erosion → drop
                        i = j
                        continue
                windows.append((t0, t1))
            i = j
        else:
            i += 1

    debug = dict(
        f0=f0, tau0=tau0, thr=thr, med=med, mad=mad,
        r=r, r_s=r_s, Ts_est_s=Ts,
    )
    return windows, free_mask, debug

if __name__ == "__main__":
    # Example usage
    # mp4_path = Path("/exchange/calib/calib_yellow.MP4")
    # bag_output_path = Path("/exchange/calib/calib_yellow.bag")

    # mp4_to_rosbag(mp4_path, bag_output_path)
    # print(f"Converted {mp4_path} to ROS bag at {bag_output_path}")

    # Example usage for merging VRS and ROS bag calibration
    vrs_path = Path("/exchange/calib/calib_gripper_blue/calib_250827_2.vrs")
    rosbag_path = Path("/exchange/calib/calib_gripper_blue/calib_2025-08-27_13-08-22.bag")
    temp_path = Path("/exchange/calib/calib_gripper_blue/temp")
    merge_calibration_vrs_and_calibration_bag(vrs_path, rosbag_path, temp_path)

    # mps_file = Path("/exchange/calib/gripper_yellow_gravity_compensation/mps_calib_2025-07-10_2_vrs/slam/closed_loop_trajectory.csv")
    # vrs_path = Path("/exchange/calib/gripper_yellow_gravity_compensation/calib_2025-07-10_2.vrs")
    # rosbag_path = Path("/exchange/calib/gripper_yellow_gravity_compensation/calib_2025-07-10_14-54-50.bag")
    # temp_path = Path("/exchange/calib/gripper_yellow_gravity_compensation/temp")
    # camcahin_imucam = Path("/exchange/calib/gripper_yellow_gravity_compensation/merged_calibration-camchain-imucam.yaml")
    # forceless_time_intervall = [1752159291893020650, 1752159352194227650]
    # estimate_tool_params(vrs_path, mps_file, rosbag_path, temp_path, camcahin_imucam, forceless_time_intervall)
    # print(f"Estimated tool parameters from {vrs_path}, {mps_file}, and {rosbag_path}")
    a = 2
