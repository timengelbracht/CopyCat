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
from utils import ensure_dir, save_image, clean_label, estimate_fps, load_sorted_images, is_valid_image
from mps_request import MPSClient
from data_indexer import RecordingIndex
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg") 
from PIL import Image
import math
from torchvision import transforms, models
import cv2
import shutil


class AriaData:

    # monodepth model setup
    DEVICE, _, _ = get_backend()
    MONO_DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-base-hf"
    PIPE_MONO_DEPTH = pipeline("depth-estimation", model=MONO_DEPTH_CHECKPOINT, device=DEVICE)

    # DINOv2 model setup
    DINO_CHECKPOINT = "facebook/dinov2-small-imagenet1k-1-layer"
    PIPE_DINO = pipeline("image-feature-extraction", model=DINO_CHECKPOINT, device=DEVICE, pool=True)

    # global image descriptor setup for keyframing
    BACKBONE = models.resnet50(pretrained=True)
    POOL     = BACKBONE.avgpool
    FEAT_EX  = torch.nn.Sequential(
        *(list(BACKBONE.children())[:-2]),
        POOL, torch.nn.Flatten()
    ).eval().cuda()
    PREPROCESS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std= [0.229,0.224,0.225],
        ),
    ])

    def __init__(self, base_path: Path, 
                 rec_loc: str, 
                 rec_type: str, 
                 rec_module: str, 
                 interaction_indices: str,
                 data_indexer: Optional[RecordingIndex] = None,
                 voxel: float = 0.02*6):

        self.voxel = voxel
        self.rec_loc = rec_loc
        self.base_path = base_path
        self.rec_module = rec_module
        self.rec_type = rec_type
        self.interaction_indices = interaction_indices

        self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}_vrs"

        self.mps_path_raw = self.base_path / "raw" / self.rec_loc / self.rec_type / self.rec_module / f"mps_{self.rec_loc}_{self.interaction_indices}_{self.rec_type}_vrs"
        self.vrs_file_raw = self.base_path / "raw" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}.vrs"
        self.mps_path_raw_all_devices = self.base_path / "raw" / self.rec_loc / "mps_all_devices"

        self.label_rgb = f"/camera_rgb"
        self.label_rgb_raw = f"/camera_rgb_raw"
        self.label_slam = f"/slam"
        self.label_keyframes = f"visual_registration/keyframes/rgb"
        self.label_keyframes_raw = f"/keyframes_raw/rgb"
        self.label_depth = f"/camera_depth"

        self.label_clt = f"{self.label_slam}/closed_loop_trajectory"
        self.label_sdp = f"{self.label_slam}/semidense_points"
        self.label_sdpd = f"{self.label_slam}/semidense_points_downsampled"

        self.semidense_points_ply_path = self.extraction_path / self.label_sdp.strip("/") / "data.ply"
        self.semidense_points_downsampled_ply_path = self.extraction_path / self.label_sdpd.strip("/") / "data.ply"
        self.visual_registration_output_path = self.extraction_path / "visual_registration"

        self.device_calib = None
        self.provider = None
        self.load_provider()

        self.calibration = self.get_calibration(undistort=True)

        self.extracted_vrs = Path(self.extraction_path / self.label_rgb.strip("/")).exists()
        self.extracted_vrs_raw = Path(self.extraction_path / self.label_rgb_raw.strip("/")).exists()
        self.extracted_mps = Path(self.extraction_path / self.label_slam.strip("/")).exists()
        self.time_aligned = False        

        self.data_indexer = data_indexer

        self.logging_tag = f"{self.rec_loc}_{self.rec_type}_{self.rec_module}".upper()

        self.rgb_extension = ".png"  # Assuming RGB images are in PNG format

        self.statistics = {
            "keypoints_per_image": [],
            "motion_blur_per_image": [],
            "average_depth_per_image": [],
            "average_depth_range_per_image": [],
            "relative_depth_range_per_image": [],
            "depth_cv_per_image": []
        }


    def _extracted(self, label: str) -> bool:
        """
        Check if the data for the given label has been extracted.
        """
        label_path = self.extraction_path / label.strip("/")
        return label_path.exists() and any(label_path.iterdir())

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
        
    def request_mps(self, force: bool = False) -> None:

        if self.mps_path_raw.exists() and not force:
            print(f"[{self.logging_tag}] MPS data already exists at {self.mps_path_raw}")
            return
        
        if not self.vrs_file_raw.exists():
            raise FileNotFoundError(f"VRS file not found: {self.vrs_file_raw}")
        
        print(f"[{self.logging_tag}] Requesting MPS data from {self.vrs_file_raw}")

        mps_client = MPSClient()
        try: # will prompt for credentials
            mps_client.request_single(
                input_path=str(self.vrs_file_raw),
                features=["SLAM", "HAND_TRACKING", "EYE_GAZE"],
                force=force,
                no_ui=True
            )
            print(f"[{self.logging_tag}] MPS data requested successfully")
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to request MPS data: {e}")

    def request_mps_all_devices(self, force: bool = False) -> None:

        mps_client = MPSClient()

        all_vrs_files = self.data_indexer.vrs_files(
            location=self.rec_loc
        )

        if self.mps_path_raw_all_devices.exists() and len(list(self.mps_path_raw_all_devices.iterdir())) >= len(all_vrs_files) and not force:
            print(f"[{self.logging_tag}] MPS data for all devices already exists at {self.mps_path_raw_all_devices}")
            return
        
        # try:
        mps_client.request_multi(
            input_paths=all_vrs_files,
            output_dir=self.mps_path_raw_all_devices,
            force=force,
            no_ui=True
        )
        print(f"[{self.logging_tag}] MPS data requested successfully")
        # except Exception as e:
        #     print(f"[{self.logging_tag}] Failed to request MPS data: {e}")


    def extract_vrs(self, undistort: bool = True):

        if undistort and self.extracted_vrs:
            print(f"[{self.logging_tag}] VRS data already extracted to {self.extraction_path}")
            return
        if not undistort and self.extracted_vrs_raw:
            print(f"[{self.logging_tag}] VRS data already extracted to {self.extraction_path}")
            return

        if not self.vrs_file_raw:
            raise FileNotFoundError(f"No vrs file found for {self.vrs_file_raw}")

        if not undistort:
            out_dir = self.extraction_path / self.label_rgb_raw.strip("/")
            self.extracted_vrs_raw = True
        else:
            out_dir = self.extraction_path / self.label_rgb.strip("/")
            self.extracted_vrs = True

        ensure_dir(out_dir)

        calib = self.device_calib.get_camera_calib("camera-rgb")
        # pinhole = calibration.get_linear_camera_calibration(512, 512, 150)

        f = self.device_calib.get_camera_calib("camera-rgb").get_focal_lengths()[0]
        h = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[0]
        w = self.device_calib.get_camera_calib("camera-rgb").get_image_size()[1]

        pinhole = calibration.get_linear_camera_calibration(w, h, f)
        # pinhole_rot = calibration.rotate_camera_calib_cw90deg(pinhole)

        print(f"[{self.logging_tag}] Data provider created successfully")
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

        print(f"[{self.logging_tag}] Saved RGB images to {out_dir}")

    def extract_mps(self, mps_path: Optional[str | Path | os.PathLike] = None) -> None:

        if self.extracted_mps:
            print(f"[{self.logging_tag}] MPS data already extracted to {self.extraction_path}")
            return

        # Path to closed loop SLAM trajectory
        closed_loop_trajectory_file = self.mps_path_raw / "slam" / "closed_loop_trajectory.csv"  # fixed typo in filename
        semidense_points_file = self.mps_path_raw / "slam" / "semidense_points.csv.gz"

        try:
            df = pd.read_csv(closed_loop_trajectory_file)
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {closed_loop_trajectory_file}: {e}")
            return
        
        try:
            df_pts = pd.read_csv(semidense_points_file, compression='gzip')
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {semidense_points_file}: {e}")
            return

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df.columns:
            df["tracking_timestamp_us"] = (df["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save to extracted location
        label_clt = f"{self.label_slam}/closed_loop_trajectory"
        csv_dir = self.extraction_path / label_clt.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved closed loop trajectory CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Closed loop trajectory CSV already exists: {csv_dir}/data.csv")


        label_sdp = f"{self.label_slam}/semidense_points"
        csv_dir = self.extraction_path / label_sdp.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df_pts.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved semidense points CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Semidense points CSV already exists: {csv_dir}/data.csv")
            return

        # TODO - add more mps data extraction as needed
        # TODO - UTC timestamp is NOT changed at the moment, only device timestamp!!!!
        
        # Update extracted flag
        self.extracted_mps = True
        print(f"[{self.logging_tag}] Extracted MPS data to {csv_dir}")

    def extract_mps_multi(self, force: bool = False) -> None:
        """
        Extracts multi MPS data for this device.
        """

        all_vrs_files = self.data_indexer.vrs_files(
            location=self.rec_loc
        )

        if not self.mps_path_raw_all_devices.exists() or len(list(self.mps_path_raw_all_devices.iterdir())) < len(all_vrs_files) :
            raise FileNotFoundError(f"MPS data for all devices not found at {self.mps_path_raw_all_devices}, please run request_mps_all_devices() first.")

        multi_slam_config_file = self.mps_path_raw_all_devices / "vrs_to_multi_slam.json"

        if not multi_slam_config_file.exists():
            raise FileNotFoundError(f"Multi SLAM config file not found: {multi_slam_config_file}. Please run request_mps_all_devices() first.")
        
        with open(multi_slam_config_file, "r") as f:
            multi_slam_config = json.load(f)

        multi_slam_index = multi_slam_config.get(str(self.vrs_file_raw))

        # Path to closed loop SLAM trajectory
        closed_loop_trajectory_file = self.mps_path_raw_all_devices / multi_slam_index / "slam" / "closed_loop_trajectory.csv"  # fixed typo in filename
        semidense_points_file = self.mps_path_raw_all_devices / multi_slam_index / "slam" / "semidense_points.csv.gz"

        try:
            df = pd.read_csv(closed_loop_trajectory_file)
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {closed_loop_trajectory_file}: {e}")
            return
        
        try:
            df_pts = pd.read_csv(semidense_points_file, compression='gzip')
        except Exception as e:
            print(f"[{self.logging_tag}] Failed to read CSV {semidense_points_file}: {e}")
            return

        # Normalize to 'timestamp' naming
        if "tracking_timestamp_us" in df.columns:
            df["tracking_timestamp_us"] = (df["tracking_timestamp_us"].astype(np.int64) * 1_000)
            df.rename(columns={"tracking_timestamp_us": "timestamp"}, inplace=True)

        # Save to extracted location
        label_clt = f"multi_slam/closed_loop_trajectory"
        csv_dir = self.extraction_path / label_clt.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved closed loop trajectory CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Closed loop trajectory CSV already exists: {csv_dir}/data.csv")

        label_sdp = f"multi_slam/semidense_points"
        csv_dir = self.extraction_path / label_sdp.strip("/")

        if not csv_dir.exists():
            ensure_dir(csv_dir)
            df_pts.to_csv(csv_dir / "data.csv", index=False)
            print(f"[{self.logging_tag}] Saved semidense points CSV: {csv_dir}/data.csv")
        else:
            print(f"[{self.logging_tag}] Semidense points CSV already exists: {csv_dir}/data.csv")
            return

        # TODO - add more mps data extraction as needed
        # TODO - UTC timestamp is NOT changed at the moment, only device timestamp!!!!
        
        print(f"[{self.logging_tag}] Extracted multi MPS data to {csv_dir}")


        a = 2

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

        print(f"[{self.logging_tag}] Estimated fps: {fps:.2f}")

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

        print(f"[{self.logging_tag}] Saved video to {out_dir}")

    def extract_mono_depth(
        self,
        downsampling_factor: int = 2,
        batch_size: int = 8,
        force: bool = False,
    ) -> None:
        """Batch the images in pure Python, call PIPE once per batch, save each depth map."""
        if not self.extracted_vrs:
            raise FileNotFoundError(f"[{self.logging_tag}] …")

        out_dir = self.extraction_path / self.label_depth.strip("/")
        if out_dir.exists() and any(out_dir.glob("*.npy")) and not force:
            print(f"[{self.logging_tag}] already extracted → {out_dir}")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Gather & sort all the image paths
        img_paths = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.png")
        )
        total = len(img_paths)
        if total == 0:
            print(f"[{self.logging_tag}] no images found")
            return

        # 2) Process in Python chunks of batch_size
        n_batches = math.ceil(total / batch_size)
        pbar = tqdm(total=total, desc=f"[{self.logging_tag}] Processing monodepth batches", unit="batch")
        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, total)
            batch_paths = img_paths[start:end]

            # load + preprocess into PIL list
            batch_pils = []
            stems      = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[{self.logging_tag}] skipping {p}")
                    continue
                h, w = img.shape[:2]
                img = cv2.resize(img, (w // downsampling_factor, h // downsampling_factor))
                batch_pils.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                stems.append(p.stem)

            # 3) Single batched call
            with torch.no_grad():
                preds = self.PIPE_MONO_DEPTH(batch_pils)  # returns list of dicts

            # 4) Save out each result
            for pred, stem in zip(preds, stems):
                depth_map = pred["predicted_depth"].squeeze().cpu().numpy()
                np.save(out_dir / f"{stem}.npy", depth_map)

            pbar.update(len(batch_pils))

        print(f"[{self.logging_tag}] done! depth maps in {out_dir}")

    # def extract_keyframes(self, undistort: bool = False, only_first_frame: bool = True) -> None:
    #     """
    #     Returns the keyframes as a numpy array.
    #     Reads every `stride`-th image from the directory.
    #     """

    #     if not self.extracted_vrs:
    #         raise FileNotFoundError(f"[{self.logging_tag}] VRS data not extracted to {self.extraction_path}")
        
    #     if not self.extracted_mps:
    #         raise FileNotFoundError(f"[{self.logging_tag}] MPS data not extracted to {self.extraction_path}")

    #     if undistort:
    #         in_dir = self.extraction_path / self.label_rgb.strip("/")
    #         out_dir = self.visual_registration_output_path / self.label_keyframes.strip("/")
    #     else:
    #         in_dir = self.extraction_path / self.label_rgb_raw.strip("/")
    #         out_dir = self.visual_registration_output_path / self.label_keyframes_raw.strip("/")

    #     ensure_dir(out_dir)

    #     # get first mps pose timestamp as it takes some time to initialize
    #     # i.e. the first frame might not have a pose yet
    #     mps_traj = self.get_closed_loop_trajectory()
    #     time_zero = mps_traj["timestamp"].iloc[0]

    #     # Get sorted list of image files (assumes naming like 0.png, 1.png, ...)
    #     image_files = sorted(
    #         Path(in_dir).glob("*.png"),
    #         key=lambda x: int(x.stem)
    #     )

    #     # remove images before first mps pose
    #     image_files_valid = [img for img in image_files if int(img.stem) >= time_zero]

    #     if only_first_frame:
    #         img = cv2.imread(str(image_files_valid[0]))
    #         if img is not None:
    #             (out_dir / f"{image_files_valid[0].stem}.png").write_bytes(cv2.imencode('.png', img)[1])
    #     else:
    #         L = len(image_files_valid)
    #         indices = np.linspace(0, L - 1, num=min(15, L), dtype=int)
    #         for i in indices:
    #             img = cv2.imread(str(image_files_valid[i]))
    #             if img is not None:
    #                 (out_dir / f"{image_files_valid[i].stem}.png").write_bytes(cv2.imencode('.png', img)[1])

    def extract_keyframes(self, 
                                n_keyframes: int = 20,
                                stride: int = 2,
                                feature_percentile: float = 10.0,
                                depth_percentile: float = 75.0,
                                blur_percentile: float = 10.0,
                                force: bool = False) -> None:


        if self._extracted(self.label_keyframes) and not force:
            print(f"[{self.logging_tag}] Keyframes already extracted to {self.visual_registration_output_path / self.label_keyframes.strip('/')}")
            return
        
        if not (
            self._extracted(self.label_rgb)
            or self._extracted(self.label_depth)
            or self._extracted(self.label_slam)
        ):
            raise FileNotFoundError(
            f"[{self.logging_tag}] No RGB, depth, or SLAM data found in "
            f"{self.extraction_path / self.label_rgb.strip('/')} or "
            f"{self.extraction_path / self.label_depth.strip('/')} or "
            f"{self.extraction_path / self.label_slam.strip('/')}"
            )
        
        out_dir = self.extraction_path / self.label_keyframes.strip("/")
        ensure_dir(out_dir)

        if not any(self.statistics.values()):
            print(f"[{self.logging_tag}] No statistics found, computing...")
            self.get_statistics(stride=stride)
        
        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.png"),
            key=lambda x: int(x.stem)
        )

        depth_dir = self.extraction_path / self.label_depth.strip("/")

        # 2) Compute thresholds
        feats = np.array(self.statistics["keypoints_per_image"])
        depths = np.array(self.statistics["average_depth_per_image"])
        blurs = np.array(self.statistics["motion_blur_per_image"])

        feat_thr  = np.percentile(feats, feature_percentile)        # drop bottom X%
        depth_thr = np.percentile(depths, depth_percentile)     # keep top (100-X)%
        blur_thr  = np.percentile(blurs, blur_percentile)

        # corresponding stats lists (already pure Python types)
        feats_list  = self.statistics["keypoints_per_image"]
        depths_list = self.statistics["average_depth_per_image"]
        blurs_list  = self.statistics["motion_blur_per_image"]

        candidates = []
        for p, kp_count, mean_d, blur_var in zip(rgb_files, feats_list, depths_list, blurs_list):
            # fast threshold checks
            if kp_count  < feat_thr:  continue
            if mean_d    < depth_thr: continue
            if blur_var  < blur_thr:  continue
            candidates.append(p)

        if not candidates:
            print(f"[{self.logging_tag}] No frames passed filtering.")
            return

        descs = []
        batch_size = 16
        for i in tqdm(range(0, len(candidates), batch_size), desc="Extracting DINO features"):
            batch = candidates[i : i + batch_size]

            # load as PIL RGB
            pil_imgs = [ Image.open(str(p)).convert("RGB") for p in batch ]

            # now call the pipeline
            feats = self.PIPE_DINO(pil_imgs)  # returns list of [batch_size, 1, dim]

            for out in feats: # dim [1, D]
                descs.append(np.asarray(out[0]))

        descs = np.stack(descs)  # shape [N, D]

        # 3) Farthest‐point sampling
        N = len(descs)
        if N <= n_keyframes:
            selected = list(range(N))
        else:
            selected = [0]
            min_dists = np.full(N, np.inf)
            for _ in range(1, n_keyframes):
                last = selected[-1]
                dists = np.linalg.norm(descs - descs[last], axis=1)
                min_dists = np.minimum(min_dists, dists)
                sel = int(np.argmax(min_dists))
                selected.append(sel)

        # 4) Collect, save list, and copy images
        self.selected_keyframes = [candidates[i] for i in selected]
        out_dir.mkdir(parents=True, exist_ok=True)

        # write the list
        with open(out_dir / "keyframes.txt", "w") as f:
            for p in self.selected_keyframes:
                f.write(p.name + "\n")

        # copy the actual image files
        for p in self.selected_keyframes:
            dst = out_dir / p.name
            shutil.copy2(p, dst)  # preserves metadata

        print(f"[{self.logging_tag}] Wrote {len(self.selected_keyframes)} keyframes to {out_dir}")

    def get_closed_loop_trajectory(self) -> pd.DataFrame:
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
            self._points_raw_to_ply()
        return o3d.io.read_point_cloud(str(self.semidense_points_ply_path))
    
    def get_downsampled(self, force: bool = False) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        Returns downsampled_cloud.  Caches both to disk.
        """

        if force or not (self.semidense_points_downsampled_ply_path.exists()):
            self._make_downsampled()

        down = o3d.io.read_point_cloud(str(self.semidense_points_downsampled_ply_path))

        return down
    
    def get_mps_pose_at_timestamp(self, timestamp: int) -> Optional[np.ndarray]:
        
        trajectory_df = self.get_closed_loop_trajectory()
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

    def get_transform_world_query(self) -> np.ndarray:
        """
        Returns the transform from world to device coordinates.
        """

        if not Path(self.visual_registration_output_path / "T_wq.json").exists():
            raise FileNotFoundError(f"Transform file not found: {self.visual_registration_output_path / 'T_wq.json'}. Please run the visual and pointcloud registration first.")
    
        with open(self.visual_registration_output_path / "T_wq.json", "r") as f:
            transform = json.load(f)

        self.T_wq = transform["T_wq"]

        return self.T_wq

    def _points_raw_to_ply(self, voxel: float | None = None) -> None:
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
        
        print(f"[{self.logging_tag}] Saved full-resolution PLY → {self.semidense_points_ply_path}")

    def _make_downsampled(self) -> None:
        """
        Down-sample the semidense points.
        """

        ensure_dir(self.semidense_points_downsampled_ply_path.parent)

        print(f"[{self.logging_tag}] Loading semidense points...")
        full = self.get_semidense_points_pcd()  # ensures .ply exists

        with tqdm(total=4, desc="[{self.logging_tag}] Downsample", unit="step") as pbar:
            print(f"[{self.logging_tag}] Down-sampling at voxel={self.voxel:.3f}")
            down = full.voxel_down_sample(voxel_size=self.voxel)
            pbar.update(1)

            print("[{self.logging_tag}] Estimating normals...")
            down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel * 2.0, max_nn=30
                )
            )
            pbar.update(1)

            print("[{self.logging_tag}] Saving downsampled cloud..") # stored in PLY comment
            o3d.io.write_point_cloud(str(self.semidense_points_downsampled_ply_path), down, write_ascii=False)
            pbar.update(1)

        print(
            f"[{self.logging_tag}] Cached ↓ cloud → {self.semidense_points_downsampled_ply_path.name}"
        )

    def get_statistics(self, stride: int = 1, visualize: bool = False, force: bool = False) -> None:
        """
        Computes and visualizes statistics from the RGB and depth data.
        Strides the RGB and depth files by the given stride.
        """

        if not self._extracted(self.label_depth) or not self._extracted(self.label_rgb):
            raise FileNotFoundError(f"[{self.logging_tag}] RGB or depth data not extracted to {self.extraction_path}")

        statistics_file = self.extraction_path / "statistics.json"
        if statistics_file.exists() and not force:
            print(f"[{self.logging_tag}] Statistics already computed and saved to {statistics_file}")
            self.load_statistics()
            return

        # get RGB files sorted by timestamp
        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*{self.rgb_extension}"),
            key=lambda x: int(x.stem)
        )
        # stride the RGB files
        rgb_files = rgb_files[::stride]
        
        # get depth files sorted by timestamp
        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )
        # stride the depth files
        depth_files = depth_files[::stride]
        
        # depth statistics
        average_depth_per_image = []
        average_depth_range_per_image = []
        relative_depth_range_per_image = []
        depth_cv_per_image = []
        for depth_file in tqdm(depth_files, desc=f"[{self.logging_tag}] Processing depth statistics", unit="file"):
            depth_map = np.load(depth_file)
            if depth_map.size == 0:
                continue
            average_depth = np.mean(depth_map)
            average_depth_per_image.append(average_depth)

            d = depth_map.flatten()
            d = d[~np.isnan(d)]

            mean_d  = np.mean(d)
            std_d   = np.std(d)
            depth_cv = std_d / (mean_d + 1e-6)

            p5, p95 = np.percentile(d, [5, 95])
            depth_prange = (p95 - p5) / (p95 + 1e-6)

            relative_depth_range_per_image.append(float(depth_prange))
            depth_cv_per_image.append(float(depth_cv))
            average_depth_range = np.max(depth_map) - np.min(depth_map)
            average_depth_range_per_image.append(float(average_depth_range))

        # local feature statistics (ORB)
        orb = cv2.ORB_create()
        keypoints_per_image = []
        for rgb_file in tqdm(rgb_files, desc=f"[{self.logging_tag}] Processing RGB statistics", unit="file"):
            img = cv2.imread(str(rgb_file))
            if img is None:
                continue
            keypoints, _ = orb.detectAndCompute(img, None)
            keypoints_per_image.append(int(len(keypoints)))  

        # motion blur statistics
        motion_blur_per_image = []
        for rgb_file in tqdm(rgb_files, desc=f"[{self.logging_tag}] Processing RGB files for motion blur", unit="file"):
            img = cv2.imread(str(rgb_file))
            if img is None:
                continue
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            motion_blur_per_image.append(float(laplacian_var))

        self.statistics = {
            "keypoints_per_image": keypoints_per_image,
            "motion_blur_per_image": motion_blur_per_image,
            "average_depth_per_image": average_depth_per_image,
            "average_depth_range_per_image": average_depth_range_per_image,
            "relative_depth_range_per_image": relative_depth_range_per_image,
            "depth_cv_per_image": depth_cv_per_image
        }

        # save statistics to file
        self.save_statistics()

        if visualize:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

            # Top‑left: Keypoints per Image
            axes[0, 0].hist(keypoints_per_image, bins=50, color='green', alpha=0.7)
            axes[0, 0].set_title(f"Keypoints per Image ({self.rec_loc})")
            axes[0, 0].set_xlabel("Number of Keypoints")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].grid(True)
            # Top‑right: Motion Blur per Image
            axes[0, 1].hist(motion_blur_per_image, bins=50, color='blue', alpha=0.7)
            axes[0, 1].set_title(f"Motion Blur (Laplacian Var) ({self.rec_loc})")
            axes[0, 1].set_xlabel("Laplacian Variance")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(True)
            # Middle‑left: Avg Depth per Image
            axes[1, 0].hist(average_depth_per_image, bins=50, alpha=0.7, color='teal')
            axes[1, 0].set_title(f"Avg Depth per Image ({self.rec_loc})")
            axes[1, 0].set_xlabel("Depth (m)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].grid(True)
            # Middle‑right: Avg Depth Range per Image
            axes[1, 1].hist(average_depth_range_per_image, bins=50, alpha=0.7, color='red')
            axes[1, 1].set_title(f"Avg Depth Range per Image ({self.rec_loc})")
            axes[1, 1].set_xlabel("Range (m)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True)
            # Bottom‑left: Relative Depth Range
            axes[2, 0].hist(relative_depth_range_per_image, bins=50, alpha=0.7, color='orange')
            axes[2, 0].set_title(f"Relative Depth Range ({self.rec_loc})")
            axes[2, 0].set_xlabel("Relative Range")
            axes[2, 0].set_ylabel("Frequency")
            axes[2, 0].grid(True)
            # Bottom‑right: Depth CoV per Image
            axes[2, 1].hist(depth_cv_per_image, bins=50, alpha=0.7, color='purple')
            axes[2, 1].set_title(f"Depth Coefficient of Variation ({self.rec_loc})")
            axes[2, 1].set_xlabel("Coefficient (sigma/mu)")
            axes[2, 1].set_ylabel("Frequency")
            axes[2, 1].grid(True)
            plt.show()

    def save_statistics(self, out_path: str | Path | None = None) -> None:
        """
        Saves the computed statistics to a JSON file.
        """
        if not self.statistics:
            raise ValueError(f"[{self.logging_tag}] No statistics computed yet. Call get_statistics() first.")

        if out_path is None:
            out_path = self.extraction_path / "statistics.json"

        with open(out_path, 'w') as f:
            json.dump(self.statistics, f, indent=4)
        print(f"[{self.logging_tag}] Statistics saved to {out_path}")

    def load_statistics(self, in_path: str | Path | None = None) -> None:
        """
        Loads the statistics from a JSON file.
        """
        if in_path is None:
            in_path = self.extraction_path / "statistics.json"

        if not Path(in_path).exists():
            raise FileNotFoundError(f"[{self.logging_tag}] Statistics file not found: {in_path}")

        with open(in_path, 'r') as f:
            self.statistics = json.load(f)
        print(f"[{self.logging_tag}] Statistics loaded from {in_path}")


if __name__ == "__main__":

    test = True
    location = False
    if location:
        # extract data from a specific location
        rec_location = "bedroom_1"
        base_path = Path(f"/data/ikea_recordings")
        # rec_type_aria = "gripper"
        # rec_module = "aria_gripper"
        # interaction_indices = "1-8"
        
        data_indexer = RecordingIndex(
            os.path.join(str(base_path), "raw") 
        )

        aria_queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction=None, 
            recorder="aria*"
        )

        # Uncomment the following lines to process each aria query
        for loc, inter, rec, ii, path in aria_queries_at_loc:
            print(f"Found recorder: {rec} at {path}")

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            aria_data.request_mps(force=False)
            aria_data.request_mps_all_devices(force=False)
            
            aria_data.extract_vrs(undistort=True)
            aria_data.extract_mps()
            aria_data.extract_mps_multi(force=False)

            # can be done after time synchronization to save time
            aria_data.extract_mono_depth(force=False)

    if test:
        
        rec_location = "bedroom_1"
        base_path = Path(f"/data/ikea_recordings")
        data_indexer = RecordingIndex(
            os.path.join(str(base_path), "raw") 
        )

        aria_queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction="gripper", 
            recorder="aria_human"
        )

        for loc, inter, rec, ii, path in aria_queries_at_loc:
            print(f"Found recorder: {rec} at {path}")

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
 
            aria_data.extract_keyframes()

        a = 2