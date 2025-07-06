import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from pathlib import Path
import json
from typing import Optional, Union
import numpy as np
import cv2
from tqdm import tqdm
import zipfile
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from data_indexer import RecordingIndex
import liblzfse
import subprocess

class IPhoneData:
    def __init__(self, base_path: Path, 
                 rec_loc: str, 
                 rec_type: str,
                 rec_module: str,
                 interaction_indices: str,
                 data_indexer: Optional[RecordingIndex] = None):

        self.rec_loc = rec_loc
        self.base_path = base_path
        self.rec_type = rec_type
        self.rec_module = rec_module
        self.interaction_indices = interaction_indices
        if data_indexer is not None:
            self.data_indexer = data_indexer


        root_raw = Path(self.base_path) / "raw" / self.rec_loc / self.rec_type

        pattern = (
            f"{self.rec_module}*/"
            f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}/"
            "Shareable/*.r3d"
        )
        self.zip_path = list(root_raw.glob(pattern))[0]

        # rgbd_path = self.base_path / "raw" / self.rec_loc / self.rec_type / self.rec_module* / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}" / "Shareable" / "*.r3d"
        # self.ply_zip_path = self.base_path / "raw" / self.rec_name / self.sensor_module_name / f"{self.rec_name}" / "Zipped_PLY" / f"{self.rec_name}.zip" 
        # self.rgb = self.rgbd_path / "rgb"
        # self.depth = self.rgbd_path / "depth"
        
        self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}"
        self.meta_data = self.extraction_path / "metadata"
        # self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.sensor_module_name
        # self.extracted_rgbd = (self.extraction_path / "camera_rgb").exists()
        # self.extracted_plys = (self.extraction_path / "points").exists()

        self.label_rgb = f"/camera_rgb"
        self.label_depth = f"/camera_depth"
        self.label_conf = f"/camera_conf"
        self.label_keyframes = f"/keyframes/rgb"

        self.visual_registration_output_path = self.extraction_path / "visual_registration"

        self.K = None
        self.fps = None
        self.timestamps = None
        self.logging_tag = f"{self.rec_loc}_{self.rec_type}_{self.rec_module}".upper()


        self.extract_zip()
        self.load_metadata()

        self.t_ns_init = 0

        self.extracted_rgbd = Path(self.extraction_path / self.label_rgb.strip("/")).exists()
        
        self.rgb_extension = ".jpg"

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
        self.calibration["dh"] = metadata.get("dh")
        self.calibration["dw"] = metadata.get("dw")

        self.fps = metadata.get("fps")
        self.timestamps = metadata.get("frameTimestamps")
        self.poses = metadata.get("poses")

        # TODO - add more metadata fields as needed

    def extract_zip(self):

        if not self.zip_path.exists():
            raise FileNotFoundError(f"RGB-D path not found: {self.zip_path}")
        
        if Path(os.path.join(self.extraction_path, "rgbd")).exists():
            print(f"[{self.logging_tag}] iPhone RGB-D data already extracted to {self.extraction_path}")
            return
    
        zip_extraction_path = self.extraction_path
        zip_extraction_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting files", total=len(zip_ref.namelist())):
                zip_ref.extract(file, zip_extraction_path)

    def extract_rgbd(self):
        if self.extracted_rgbd:
            print(f"[{self.logging_tag}] iPhone RGB-D data already extracted to {self.extraction_path}")
            return

        rgbd_dir = self.extraction_path / "rgbd"
        if not rgbd_dir.exists():
            raise FileNotFoundError(f"RGB-D directory not found: {rgbd_dir}")
        

        timestamps_ns = (np.array(self.timestamps) * 1e9).astype(np.int64).tolist()

        out_dir_rgb = self.extraction_path / self.label_rgb.strip("/")
        out_dir_depth = self.extraction_path / self.label_depth.strip("/")

        out_dir_rgb.mkdir(parents=True, exist_ok=True)
        out_dir_depth.mkdir(parents=True, exist_ok=True)

        dh = self.calibration["dh"]
        dw = self.calibration["dw"]

        for i, timestamp_ns in tqdm(enumerate(timestamps_ns), total=len(timestamps_ns)):
            rgb_img = cv2.imread(str(rgbd_dir / f"{i}.jpg"))
            # depth_img = cv2.imread(str(rgbd_dir / f"{i}.exr"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            with open(rgbd_dir / f"{i}.depth", 'rb') as depth_fh:
                raw_bytes = depth_fh.read()
                decompressed_bytes = liblzfse.decompress(raw_bytes)
                depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

            depth_img = depth_img.reshape((dh, dw)) 
            

            if rgb_img is None or depth_img is None:
                print(f"[{self.logging_tag}] Error loading image at index {i}")
                continue

            out_file_rgb = out_dir_rgb / f"{timestamp_ns}.jpg"
            out_file_depth = out_dir_depth / f"{timestamp_ns}.npy"

            cv2.imwrite(str(out_file_rgb), rgb_img)
            # cv2.imwrte(str(out_file_depth), depth_img)
            np.save(str(out_file_depth), depth_img)

        # Remove unnecessary files and directories
        # Remove unnecessary files and directories
        for file in [*rgbd_dir.iterdir(), 
                 self.extraction_path / "icon"
                 "", self.extraction_path / "sound.m4a"]:
            if file.exists():
                file.unlink() if file.is_file() else file.rmdir()

        # Remove the rgbd directory itself
        # if rgbd_dir.exists():
        #     rgbd_dir.rmdir()

        self.extracted_rgbd = True

    # def extract_plys(self):

    #     """
    #     Extracts the PLY files from the zip file.
    #     """

    #     if self.extracted_plys:
    #         print(f"[{self.logging_tag}] PLY files already extracted to {self.extraction_path}")
    #         return

    #     if not self.ply_zip_path.exists():
    #         raise FileNotFoundError(f"PLY zip file not found: {self.ply_zip_path}")

    #     timestamps_ns = (np.array(self.timestamps) * 1e9).astype(np.int64).tolist()

    #     out_dir = self.extraction_path / "points"
    #     out_dir.mkdir(parents=True, exist_ok=True)

    #     with zipfile.ZipFile(self.ply_zip_path, 'r') as zip_ref:
    #         ply_members = [m for m in zip_ref.namelist() if m.endswith('.ply')]
            
    #         if len(ply_members) != len(timestamps_ns):
    #             raise ValueError("Number of PLY files does not match number of timestamps.")

    #         for idx, member in tqdm(enumerate(ply_members), desc="Extracting PLY files", total=len(ply_members)):
    #             # Extract to full path (may include subdirs)
    #             extracted_path = Path(zip_ref.extract(member, out_dir))

    #             # Define flat renamed path
    #             timestamp_ns = timestamps_ns[idx]
    #             renamed_path = out_dir / f"{timestamp_ns}.ply"

    #             # Move file (from nested path to flat target)
    #             extracted_path.rename(renamed_path)

    #     print(f"[{self.logging_tag}] Extracted PLY files to {out_dir}")

    def extract_poses(self):
        """
        Extracts the poses from the metadata.
        """

        poses = self.poses
        timestamps_ns = (np.array(self.timestamps) * 1e9).astype(np.int64).tolist()
        out_dir = self.extraction_path / "poses"
        out_dir.mkdir(parents=True, exist_ok=True)

        # pandas
        import pandas as pd
        df = pd.DataFrame(poses)
        df["timestamp"] = timestamps_ns

        df = pd.DataFrame(poses, columns=["qx", "qy", "qz", "qw", "tx", "ty", "tz"])
        df.insert(0, "timestamp", timestamps_ns)
        
        df.to_csv(out_dir / "data.csv", index=False)

        print(f"[{self.logging_tag}] Extracted poses to {out_dir}")

    def extract_keyframes(self, mode: str = "appearance_based") -> None:
        """
        Returns the keyframes as a numpy array.
        Reads every `stride`-th image from the directory.
        """

        if not self.extracted_rgbd:
            raise FileNotFoundError(f"VRS data not extracted to {self.extraction_path}")
        
        if mode not in ["first_frame", "every_15th_frame", "appearance_based"]:
            raise ValueError("Mode must be one of: 'first_frame', 'every_15th_frame', 'appearance_based'")
        
        in_dir = self.extraction_path / self.label_rgb.strip("/")
        out_dir = self.visual_registration_output_path / self.label_keyframes.strip("/")

        out_dir.mkdir(parents=True, exist_ok=True)

        traj = self.get_trajectory()
        time_zero = traj["timestamp"].iloc[0]

        # Get sorted list of image files (assumes naming like 0.png, 1.png, ...)
        image_files = sorted(
            Path(in_dir).glob("*.jpg"),
            key=lambda x: int(x.stem)
        )

        # Filter images based on time zero
        image_files_valid = [img for img in image_files if int(img.stem) >= time_zero]

        if mode == "first_frame":
            img = cv2.imread(str(image_files_valid[0]))
            if img is not None:
                (out_dir / f"{image_files_valid[0].stem}.png").write_bytes(cv2.imencode('.png', img)[1])
        elif mode == "every_15th_frame":
            L = len(image_files_valid)
            indices = np.linspace(0, L - 1, num=min(15, L), dtype=int)
            for i in indices:
                img = cv2.imread(str(image_files_valid[i]))
                if img is not None:
                    (out_dir / f"{image_files_valid[i].stem}.png").write_bytes(cv2.imencode('.png', img)[1])
        elif mode == "appearance_based":
            # Define the input and output patterns
            input_pattern = str(in_dir / '*.jpg')
            output_pattern = str(out_dir / f"{image_files_valid[0].stem}.png")

            # Run the ffmpeg command using subprocess
            ffmpeg_command = [
                'ffmpeg',
                '-pattern_type', 'glob',
                '-i', input_pattern,
                '-vf', "select='gt(scene,0.4)',showinfo",
                '-vsync', 'vfr',
                output_pattern
            ]

            subprocess.run(ffmpeg_command, check=True)


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

    # def extract_depth_video(self, out_dir: Optional[str | Path] = None) -> None:
    #     """
    #     Extracts the video from the depth images in the specified directory.
    #     """

    #     if out_dir is None:
    #         label_depth = f"/camera_depth"
    #         out_dir = self.extraction_path / label_depth.strip("/")

    #     video_name = out_dir / 'data.mp4'

    #     # Read and sort images
    #     images = sorted(
    #         [img for img in os.listdir(out_dir) if img.endswith(".exr")],
    #         key=lambda x: int(os.path.splitext(x)[0]))

    #     # Estimate average fps
    #     timestamps = [int(os.path.splitext(img)[0]) for img in images]
    #     time_diffs = np.diff(timestamps)  # nanoseconds
    #     avg_dt = np.mean(time_diffs)  # average nanosecond difference
    #     fps = 1e9 / avg_dt  # frames per second

    #     print(f"[{self.logging_tag}] Estimated fps: {fps:.2f}")

    #     # Initialize video writer
    #     frame = cv2.imread(os.path.join(out_dir, images[0]))
    #     height, width, layers = frame.shape
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    #     vmin = None
    #     vmax = None
    #     cmap = "turbo"

    #     # Write frames
    #     for image in tqdm(images, desc="Creating video", total=len(images)):
    #         depth = cv2.imread(os.path.join(out_dir, image))

    #         depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    #         if vmin is None:
    #             vmin = np.percentile(depth, 5)     # or hard-code e.g. 0.2 m
    #         if vmax is None:
    #             vmax = np.percentile(depth, 95)    # or hard-code e.g. 3.0 m
    #         depth_clipped = np.clip(depth, vmin, vmax)
    #         depth_norm   = (depth_clipped - vmin) / (vmax - vmin + 1e-8)

    #         img8 = (depth_norm * 255).astype(np.uint8)

    #         heat = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)

    #         video.write(heat)

    #     video.release()

        print(f"[{self.logging_tag}] Saved video to {out_dir}")

    def get_trajectory(self) -> pd.DataFrame:
        """Return the trajectory as a pandas DataFrame"""

        csv_dir = self.extraction_path / "poses" / "data.csv"

        if not Path(csv_dir).exists():
            raise FileNotFoundError(f"Closed loop trajectory CSV not found: {csv_dir}")
        
        df = pd.read_csv(csv_dir)
        return df

    def get_rgbd_at_timestamp(self, timestamp: int) -> tuple[np.ndarray, np.ndarray]:
        """Return RGB and depth images at the given timestamp"""

        if not self.extracted_rgbd:
            raise FileNotFoundError(f"[{self.logging_tag}] RGB-D data not extracted to {self.extraction_path}")

        rgb_path = self.extraction_path / self.label_rgb.strip("/") / f"{timestamp}.jpg"
        depth_path = self.extraction_path / self.label_depth.strip("/") / f"{timestamp}.exr"
        depth_path_raw = self.depth / f"{0}.exr"

        if not rgb_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"[{self.logging_tag}] Image files not found: {rgb_path}, {depth_path}")

        rgb_img = cv2.imread(str(rgb_path))
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depthImg_raw = cv2.imread(str(depth_path_raw), cv2.IMREAD_UNCHANGED)
        return rgb_img, depth_img
    
    def get_pose_at_timestamp(self, timestamp: int) -> np.ndarray:

        """Return the pose at the given timestamp"""

        if not self.extracted_rgbd:
            raise FileNotFoundError(f"[{self.logging_tag}] pose data not extracted to {self.extraction_path}")

        trajectory_df = self.get_trajectory()
        row = trajectory_df[trajectory_df["timestamp"] == timestamp]

        t_world_device = row[["tx", "ty", "tz"]].to_numpy()
        q_world_device = row[["qx", "qy", "qz", "qw"]].to_numpy()

        # Convert quaternion to rotation matrix
        r = R.from_quat(q_world_device)
        R_world_device = r.as_matrix()
        T_world_device = np.eye(4)
        T_world_device[:3, :3] = R_world_device
        T_world_device[:3, 3] = t_world_device
        return T_world_device

    def get_cloud_at_timestamp(self, timestamp: int, voxel: float | None = None) -> o3d.geometry.PointCloud:
        """Return the full-resolution cloud generated from rgbd"""

        if not self.extracted_plys:
            raise FileNotFoundError(f"[{self.logging_tag}] PLY files not extracted to {self.extraction_path}")

        ply_path = self.extraction_path / "points" / f"{timestamp}.ply"

        if not ply_path.exists():
            raise FileNotFoundError(f"[{self.logging_tag}] PLY file not found: {ply_path}")

        full_cloud = o3d.io.read_point_cloud(str(ply_path))

        if voxel is not None:
            full_cloud = full_cloud.voxel_down_sample(voxel_size=voxel)

        return full_cloud



if __name__ == "__main__":
    rec_location = "bedroom_1"
    base_path = Path(f"/data/ikea_recordings")

    
    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    iphone_queries_at_loc = data_indexer.query(
        location=rec_location, 
        interaction=None, 
        recorder="iphone*"
    )

    
    for loc, inter, rec, ii, path in iphone_queries_at_loc:
        print(f"Found recorder: {rec} at {path}")

        rec_type = inter
        rec_module = rec
        interaction_indices = ii

        iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)

        iphone_data.extract_rgbd()
        iphone_data.extract_poses()

    # iphone_data.extract_plys()
    # iphone_data.extract_rgbd()
    # iphone_data.extract_keyframes(only_first_frame=False)
    # iphone_data.get_full_cloud_at_timestamp(271350881132)
    # iphone_data.extract_video()
