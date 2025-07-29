import cv2
from pathlib import Path
import telemetry_parser
from utils_mp4 import get_frames_from_mp4, get_imu_from_mp4
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import numpy as np

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
            t_ns = int(row.timestamp_ns)
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


if __name__ == "__main__":
    # Example usage
    mp4_path = Path("/exchange/calib/calib_blue.MP4")
    bag_output_path = Path("/exchange/calib/calib_blue.bag")

    mp4_to_rosbag(mp4_path, bag_output_path)
    print(f"Converted {mp4_path} to ROS bag at {bag_output_path}")
    a = 2