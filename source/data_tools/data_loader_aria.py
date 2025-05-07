from pathlib import Path
from typing import Optional, Tuple, List
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json

class AriaData:
    def __init__(self, base_path: Path, rec_name: str, sensor_module_name: str, voxel: float = 0.02*6):

        self.voxel = voxel

        self.rec_name = rec_name
        self.base_path = base_path
        self.sensor_module_name = sensor_module_name

        self.mps_path_raw = self.base_path / "raw" / self.rec_name / self.sensor_module_name / f"mps_gripper_recording_sync_{rec_name}_vrs"
        self.vrs_file_raw = self.base_path / "raw" / self.rec_name / self.sensor_module_name / f"gripper_recording_sync_{rec_name}.vrs"
        self.extraction_path = self.base_path / "extracted" / self.rec_name / self.sensor_module_name

        self.label_rgb = f"/camera_rgb"
        self.label_rgb_raw = f"/camera_rgb_raw"
        self.label_slam = f"/slam"
        self.label_features = f"/keyframes/features"
        self.label_features_raw = f"/keyframes_raw/features"
        self.label_keyframes = f"/keyframes/rgb"
        self.label_keyframes_raw = f"/keyframes_raw/rgb"

        self.label_clt = f"{self.label_slam}/closed_loop_trajectory"
        self.label_sdp = f"{self.label_slam}/semidense_points"
        self.label_fpfh = f"{self.label_slam}/fpfh"
        self.label_sdpd = f"{self.label_slam}/semidense_points_downsampled"

        self.semidense_points_ply_path = self.extraction_path / self.label_sdp.strip("/") / "data.ply"
        self.semidense_points_downsampled_ply_path = self.extraction_path / self.label_sdpd.strip("/") / "data.ply"
        self.fpfh_path = self.extraction_path / self.label_fpfh.strip("/") / "data.npz"
        self.visual_registration_output_path = self.extraction_path / "visual_registration"

        self.device_calib = None
        self.provider = None
        self.load_provider()

        self.calibration = self.get_calibration(undistort=True)

        self.extracted_vrs = Path(self.extraction_path / self.label_rgb.strip("/")).exists()
        self.extracted_vrs_raw = Path(self.extraction_path / self.label_rgb_raw.strip("/")).exists()
        self.extracted_mps = Path(self.extraction_path / self.label_slam.strip("/")).exists()
        self.time_aligned = False        

    def load_provider(self):
        if not self.vrs_file_raw.exists():
            raise FileNotFoundError(f"VRS file not found: {self.vrs_file_raw}")
        
        self.provider = data_provider.create_vrs_data_provider(str(self.vrs_file_raw))
        if not self.provider:
            raise RuntimeError(f"Failed to create data provider for {self.vrs_file_raw}")

        self.device_calib = self.provider.get_device_calibration()

    def get_calibration(self, undistort: bool = True) -> np.ndarray:

        """
        Returns the intrinsic matrix for the RGB camera.
        If undistort is True, it returns the undistorted intrinsic matrix.
        """

        if not self.device_calib:
            raise RuntimeError("Device calibration not loaded")
        
        clb = {}

        if undistort:
            f = self.device_calib.get_camera_calib("camera-rgb").get_focal_lengths()[0]
            h = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[0]
            w = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[1]

            pinhole = calibration.get_linear_camera_calibration(w, h, f)
            pinhole_rot = calibration.rotate_camera_calib_cw90deg(pinhole)
            f_x = pinhole_rot.get_projection_params()[0]
            f_y = pinhole_rot.get_projection_params()[1]
            c_x = pinhole_rot.get_projection_params()[2]
            c_y = pinhole_rot.get_projection_params()[3]
            K = np.array([f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]).reshape(3, 3)

            clb["K"] = K
            clb["h"] = h
            clb["w"] = w
            clb["model"] = "PINHOLE"
            clb["distortion"] = np.zeros(5, dtype=np.float32)
            clb["focal_length"] = np.array([f_x, f_y], dtype=np.float32)
            clb["principal_point"] = np.array([c_x, c_y], dtype=np.float32)
            clb["pinhole_T_device_camera"] = pinhole_rot.get_transform_device_camera().to_matrix()
            clb["T_device_camera"] = self.device_calib.get_camera_calib("camera-rgb").get_transform_device_camera().to_matrix()

            return clb
        else:
            calib = self.device_calib.get_camera_calib("camera-rgb")

            h, w = calib.get_image_size()
            f_x = calib.get_focal_lengths()[0]
            f_y = calib.get_focal_lengths()[1]
            c_x = calib.get_principal_point()[0]
            c_y = calib.get_principal_point()[1]
            K = np.array([f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]).reshape(3, 3)

            clb["K"] = K
            clb["h"] = h
            clb["w"] = w
            clb["model"] = "FISHEYE"
            clb["distortion"] = calib.get_projection_params()[3:7]
            clb["focal_length"] = np.array([f_x, f_y], dtype=np.float32)
            clb["principal_point"] = np.array([c_x, c_y], dtype=np.float32)
            clb["T_device_camera"] = calib.get_transform_device_camera().to_matrix()

            return clb
        
    def extract_vrs(self, undistort: bool = True):

        if undistort and self.extracted_vrs:
            print(f"[!] VRS data already extracted to {self.extraction_path}")
            return
        if not undistort and self.extracted_vrs_raw:
            print(f"[!] VRS data already extracted to {self.extraction_path}")
            return

        if not self.vrs_file_raw:
            raise FileNotFoundError(f"No vrs file found for {self.vrs_file_raw}")

        if not undistort:
            out_dir = self.extraction_path / self.label_rgb_raw.strip("/")
            self.extracted_vrs_raw = True
        else:
            out_dir = self.extraction_path / self.label_rgb.strip("/")
            self.extracted_vrs = True

        out_dir.mkdir(parents=True, exist_ok=True)

        calib = self.device_calib.get_camera_calib("camera-rgb")
        # pinhole = calibration.get_linear_camera_calibration(512, 512, 150)

        f = self.device_calib.get_camera_calib("camera-rgb").get_focal_lengths()[0]
        h = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[0]
        w = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[1]

        pinhole = calibration.get_linear_camera_calibration(w, h, f)
        # pinhole_rot = calibration.rotate_camera_calib_cw90deg(pinhole)

        print(f"[INFO] Data provider created successfully")
        stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        for i in tqdm(range(0, self.provider.get_num_data(stream_id)), total=self.provider.get_num_data(stream_id)):
            image_data =  self.provider.get_image_data_by_index(stream_id, i)
            sensor_data = self.provider.get_sensor_data_by_index(stream_id, i)
            ts = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
            image_array = image_data[0].to_numpy_array()
            if undistort:
                image_array = calibration.distort_by_calibration(image_array, pinhole, calib)
                image_array = np.rot90(image_array, k=3)
            out_file = out_dir / f"{ts}.png"
            image_array= cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)            
            cv2.imwrite(str(out_file), image_array)

        print(f"[✓] Saved RGB images to {out_dir}")

    def extract_mps(self):

        if self.extracted_mps:
            print(f"[!] MPS data already extracted to {self.extraction_path}")
            return

        # Path to closed loop SLAM trajectory
        closed_loop_trajectory_file = self.mps_path_raw / "slam" / "closed_loop_trajectory.csv"  # fixed typo in filename
        semidense_points_file = self.mps_path_raw / "slam" / "semidense_points.csv.gz"

        try:
            df = pd.read_csv(closed_loop_trajectory_file)
        except Exception as e:
            print(f"[!] Failed to read CSV {closed_loop_trajectory_file}: {e}")
            return
        
        try:
            df_pts = pd.read_csv(semidense_points_file, compression='gzip')
        except Exception as e:
            print(f"[!] Failed to read CSV {semidense_points_file}: {e}")
            return

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df.columns:
            df["tracking_timestamp_us"] = (df["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save to extracted location
        label_clt = f"{self.label_slam}/closed_loop_trajectory"
        csv_dir = self.extraction_path / label_clt.strip("/")
        csv_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_dir / "data.csv", index=False)

        print(f"[✓] Saved closed loop trajectory CSV: {csv_dir}/data.csv")

        label_sdp = f"{self.label_slam}/semidense_points"
        csv_dir = self.extraction_path / label_sdp.strip("/")
        csv_dir.mkdir(parents=True, exist_ok=True)
        df_pts.to_csv(csv_dir / "data.csv", index=False)

        print(f"[✓] Saved semidense points CSV: {csv_dir}/data.csv")

        # TODO - add more mps data extraction as needed
        # TODO - UTC timestamp is NOT changed at the moment, only device timestamp!!!!
        
        # Update extracted flag
        self.extracted_mps = True
        print(f"[✓] Extracted MPS data to {csv_dir}")

    def extract_video(self, out_dir: Optional[str | Path] = None, undistort: bool = True) -> None:
        """
        Extracts the video from the RGB images in the specified directory.
        """

        if out_dir is None and not undistort:
            out_dir = self.extraction_path / self.label_rgb_raw.strip("/")
        elif out_dir is None and undistort:
            out_dir = self.extraction_path / self.label_rgb.strip("/")

        video_name = out_dir / 'data.mp4'

        # Read and sort images
        images = sorted(
            [img for img in os.listdir(out_dir) if img.endswith(".png")],
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

    def extract_keyframes(self, undistort: bool = False, only_first_frame: bool = True) -> None:
        """
        Returns the keyframes as a numpy array.
        Reads every `stride`-th image from the directory.
        """

        if not self.extracted_vrs:
            raise FileNotFoundError(f"VRS data not extracted to {self.extraction_path}")
        
        if not self.extracted_mps:
            raise FileNotFoundError(f"MPS data not extracted to {self.extraction_path}")

        if undistort:
            in_dir = self.extraction_path / self.label_rgb.strip("/")
            out_dir = self.visual_registration_output_path / self.label_keyframes.strip("/")
        else:
            in_dir = self.extraction_path / self.label_rgb_raw.strip("/")
            out_dir = self.visual_registration_output_path / self.label_keyframes_raw.strip("/")

        out_dir.mkdir(parents=True, exist_ok=True)

        # get first mps pose timestamp as it takes some time to initialize
        # i.e. the first frame might not have a pose yet
        mps_traj = self.closed_loop_trajectory()
        time_zero = mps_traj["timestamp"].iloc[0]

        # Get sorted list of image files (assumes naming like 0.png, 1.png, ...)
        image_files = sorted(
            Path(in_dir).glob("*.png"),
            key=lambda x: int(x.stem)
        )

        # remove images before first mps pose
        image_files_valid = [img for img in image_files if int(img.stem) >= time_zero]

        if only_first_frame:
            img = cv2.imread(str(image_files_valid[0]))
            if img is not None:
                (out_dir / f"{image_files_valid[0].stem}.png").write_bytes(cv2.imencode('.png', img)[1])
        else:
            L = len(image_files_valid)
            indices = np.linspace(0, L - 1, num=min(10, L), dtype=int)
            for i in indices:
                img = cv2.imread(str(image_files_valid[i]))
                if img is not None:
                    (out_dir / f"{image_files_valid[i].stem}.png").write_bytes(cv2.imencode('.png', img)[1])

    def closed_loop_trajectory(self) -> Optional[pd.DataFrame]:
        """
        Returns the closed loop trajectory as a pandas DataFrame.
        """

        csv_dir = self.extraction_path / self.label_clt.strip("/") / "data.csv"

        if not Path(csv_dir).exists():
            raise FileNotFoundError(f"Closed loop trajectory CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    
    def get_semidense_points_df(self) -> Optional[pd.DataFrame]:
        """
        Returns the semidense points as a pandas DataFrame.
        """

        csv_dir = self.extraction_path / self.label_sdp.strip("/") / "data.csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Semidense points CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df
    
    def get_semidense_points_pcd(self, force: bool = False) -> o3d.geometry.PointCloud:
        """Return the full-resolution cloud, converting from E57 if needed."""

        if not self.semidense_points_ply_path.exists() or force:
            self._semidense_points_to_ply()
        return o3d.io.read_point_cloud(str(self.semidense_points_ply_path))
    
    def get_downsampled(self, force: bool = False) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        Returns (downsampled_cloud, FPFH_features).  Caches both to disk.
        """

        if force or not (self.semidense_points_downsampled_ply_path.exists()):
            self._make_downsampled()

        down = o3d.io.read_point_cloud(str(self.semidense_points_downsampled_ply_path))

        return down
    
    def get_mps_pose_at_timestamp(self, timestamp: int) -> Optional[np.ndarray]:
        
        trajectory_df = self.closed_loop_trajectory()
        if trajectory_df is None:
            print(f"[!] No closed loop trajectory data found. re-extract MPS data.")
            return None
        
        # Find the closest timestamp
        # closest_index = (np.abs(trajectory_df["timestamp"] - timestamp)).idxmin()
        
        # find next larger and smaller timestamp
        closest_index_later = trajectory_df[trajectory_df["timestamp"] >= timestamp].index.min()
        closest_index_prev = trajectory_df[trajectory_df["timestamp"] < timestamp].index.max()
    
        if timestamp < trajectory_df["timestamp"].iloc[0]:
            return None
            raise ValueError(f"Timestamp {timestamp} is before the first timestamp in the trajectory.")
        
        if timestamp > trajectory_df["timestamp"].iloc[-1]:
            return None
            raise ValueError(f"Timestamp {timestamp} is after the last timestamp in the trajectory.")

        if closest_index_prev is np.nan:
            # TODO assign it timestamp it currently has, no interpoaltion
            closest_index = (np.abs(trajectory_df["timestamp"] - timestamp)).idxmin()
            closest_row = trajectory_df.iloc[closest_index]
            t_world_device = closest_row[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
            q_world_device = closest_row[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()

            # Convert quaternion to rotation matrix
            r = R.from_quat(q_world_device)
            R_world_device = r.as_matrix()
            T_world_device = np.eye(4)
            T_world_device[:3, :3] = R_world_device
            T_world_device[:3, 3] = t_world_device
            return T_world_device
        
        # interpolate poses
        timestamp_prev = trajectory_df["timestamp"].iloc[closest_index_prev]
        timestamp_later = trajectory_df["timestamp"].iloc[closest_index_later]

        row_prev = trajectory_df.iloc[closest_index_prev]
        row_later = trajectory_df.iloc[closest_index_later]

        t_world_device_prev = row_prev[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
        t_world_device_later = row_later[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
        q_world_device_prev = row_prev[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()
        q_world_device_later = row_later[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()

        r_prev = R.from_quat(q_world_device_prev)
        r_later = R.from_quat(q_world_device_later)

        alpha = (timestamp - timestamp_prev) / (timestamp_later - timestamp_prev) 

        t_world_device = (1 - alpha) * t_world_device_prev + alpha * t_world_device_later

        slerp = Slerp([timestamp_prev, timestamp_later], R.concatenate([r_prev, r_later]))
        rot = slerp(timestamp)

        R_world_device = rot.as_matrix()

        T_world_device = np.eye(4)
        T_world_device[:3, :3] = R_world_device
        T_world_device[:3, 3] = t_world_device

        return T_world_device
        # TODO - add more mps data extraction as needed
        # TODO interpolate pose between timestamps if needed

    def get_transform_world_aria(self) -> np.ndarray:
        """
        Returns the transform from world to device coordinates.
        """

        if not Path(self.visual_registration_output_path / "T_wq.json").exists():
            raise FileNotFoundError(f"Transform file not found: {self.visual_registration_output_path / 'T_wq.json'}. Please run the visual and pointcloud registration first.")
    
        with open(self.visual_registration_output_path / "T_wq.json", "r") as f:
            transform = json.load(f)

        self.T_wq = transform["T_wq"]

        return self.T_wq

    def _semidense_points_to_ply(self, voxel: float | None = None) -> None:
        """
        Converts the semidense points DataFrame to a PLY file.
        """

        # Convert DataFrame to PLY format
        df = self.get_semidense_points_df()
        xyz = df[["px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]].to_numpy(dtype=np.float32)

        xyz = xyz[~np.isnan(xyz).any(axis=1)]

        mask = (xyz[:, 3] <= 0.005) & (xyz[:, 4] <= 0.01)
        xyz = xyz[mask]
        xyz = xyz[:, :3]
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if voxel:
            pcd = pcd.voxel_down_sample(voxel)
        
        with tqdm(total=1, desc="Saving PLY", unit="file") as pbar:
            o3d.io.write_point_cloud(str(self.semidense_points_ply_path), pcd, write_ascii=False)
            pbar.update(1)        
        
        print(f"[Aria] Saved full-resolution PLY → {self.semidense_points_ply_path}")

    def _make_downsampled(self) -> None:
        """
        Down-sample the semidense points.
        """

        if not self.semidense_points_downsampled_ply_path.parent.exists():
            self.semidense_points_downsampled_ply_path.parent.mkdir(parents=True, exist_ok=True)

        if self.semidense_points_downsampled_ply_path.exists() and self.fpfh_path.exists():
            print(f"[Aria] Downsampled cloud already exist at {self.semidense_points_downsampled_ply_path} and {self.fpfh_path}")
            return

        print(f"[Aria] Loading semidense points...")
        full = self.get_semidense_points_pcd()  # ensures .ply exists

        with tqdm(total=4, desc="[Aria] Downsample", unit="step") as pbar:
            print(f"[Aria] Down-sampling at voxel={self.voxel:.3f}")
            down = full.voxel_down_sample(voxel_size=self.voxel)
            pbar.update(1)

            print("[Aria] Estimating normals...")
            down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel * 2.0, max_nn=30
                )
            )
            pbar.update(1)

            print("[Aria] Saving downsampled cloud..") # stored in PLY comment
            o3d.io.write_point_cloud(str(self.semidense_points_downsampled_ply_path), down, write_ascii=False)
            pbar.update(1)

        print(
            f"[Aria] Cached ↓ cloud → {self.semidense_points_downsampled_ply_path.name}"
        )


if __name__ == "__main__":

    # Example usage
    rec_name = "door_6"
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    sensor_module_name = "aria_human_ego"

    aria_data = AriaData(base_path, rec_name, sensor_module_name)


    # aria_data.extract_vrs()
    # aria_data.extract_vrs(undistort=True)
    # aria_data.extract_video()
    # aria_data.extract_video(undistort=True)

    # aria_data.extract_mps()

    # aria_data.get_down_and_fpfh()


    aria_data.extract_keyframes()
    aria_data.extract_keyframes(undistort=True)
    # aria_data.get_mps_pose_at_timestamp()
    # aria_data.get_transform_world_aria()
