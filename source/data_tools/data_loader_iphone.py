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
from utils_keyframing import KeyframeExtractor
from utils import ensure_dir
import shutil


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

        module_base = self.rec_module.split(" ")[0] 

        pattern = (
            f"{module_base}*/"
            f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}*/"
            "Shareable/*.r3d"
        )
        self.zip_path = list(root_raw.glob(pattern))[0]
   
        self.extraction_path_base = self.base_path / "extracted" / self.rec_loc / self.rec_type
        self.extraction_path = self.base_path / "extracted" / self.rec_loc / self.rec_type / self.rec_module / f"{self.rec_loc}_{self.interaction_indices}_{self.rec_type}"
        self.meta_data = self.extraction_path / "metadata"

        self.label_rgb = f"/camera_rgb"
        self.label_depth = f"/camera_depth"
        self.label_conf = f"/camera_conf"
        self.label_keyframes = f"visual_registration/keyframes/rgb"
        self.label_poses = f"/poses"

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

        self.statistics = {}

    def load_metadata(self):
        if not self.meta_data.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_data}")

        with open(self.meta_data, 'r') as f:
            metadata = json.load(f)

        K = metadata.get("K")
        K_mat = np.array(K).reshape(3, 3).T

        self.calibration = {}
        self.calibration["PINHOLE"] = {}

        w = metadata.get("w")
        h = metadata.get("h")
        dh = metadata.get("dh")
        dw = metadata.get("dw")
        f_x = K_mat[0, 0]
        f_y = K_mat[1, 1]
        c_x = K_mat[0, 2]
        c_y = K_mat[1, 2]

        self.calibration["PINHOLE"]["K"] = K_mat
        self.calibration["PINHOLE"]["h"] = h
        self.calibration["PINHOLE"]["w"] = w
        self.calibration["PINHOLE"]["dh"] = dh
        self.calibration["PINHOLE"]["dw"] = dw
        self.calibration["PINHOLE"]["colmap_camera_cfg"] = {
            "model":  "PINHOLE",
            "width":   w,
            "height":  h,
            "params": [f_x, f_y, c_x, c_y],
        }

        self.fps = metadata.get("fps")
        self.timestamps = metadata.get("frameTimestamps")
        self.poses = metadata.get("poses")

        # TODO - add more metadata fields as needed

    def extract_zip(self):

        if not self.zip_path.exists():
            raise FileNotFoundError(f"RGB-D path not found: {self.zip_path}")
        
        if Path(os.path.join(self.extraction_path, "rgbd")).exists() or Path(os.path.join(self.extraction_path, "camera_rgb")).exists():
            print(f"[{self.logging_tag}] iPhone RGB-D data already extracted to {self.extraction_path}")
            return
    
        zip_extraction_path = self.extraction_path
        zip_extraction_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting files", total=len(zip_ref.namelist())):
                zip_ref.extract(file, zip_extraction_path)

    def extract_rgbd(self):
        """
        Extracts RGB and depth images from the iPhone recording.
        This method reads the RGB and depth images from the extracted directory,
        decompresses the depth images, and saves them in the specified output directories.
        """

        if self._extracted(self.label_rgb) and self._extracted(self.label_depth):
            print(f"[{self.logging_tag}] RGB-D data already extracted to {self.extraction_path / self.label_rgb.strip('/')}")
            return

        rgbd_dir = self.extraction_path / "rgbd"
        if not rgbd_dir.exists():
            raise FileNotFoundError(f"RGB-D directory not found: {rgbd_dir}")
        

        timestamps_ns = (np.array(self.timestamps) * 1e9).astype(np.int64).tolist()

        out_dir_rgb = self.extraction_path / self.label_rgb.strip("/")
        out_dir_depth = self.extraction_path / self.label_depth.strip("/")

        out_dir_rgb.mkdir(parents=True, exist_ok=True)
        out_dir_depth.mkdir(parents=True, exist_ok=True)

        dh = self.calibration["PINHOLE"]["dh"]
        dw = self.calibration["PINHOLE"]["dw"]

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
            np.save(str(out_file_depth), depth_img)

        # Remove unnecessary files and directories
        for file in [*rgbd_dir.iterdir(), 
                 self.extraction_path / "icon"
                 "", self.extraction_path / "sound.m4a"]:
            if file.exists():
                file.unlink() if file.is_file() else file.rmdir()

        self.extracted_rgbd = True

    def extract_poses(self):
        """
        Extracts the poses from the metadata.
        """

        if self._extracted(self.label_poses):
            print(f"[{self.logging_tag}] poses data already extracted to {self.extraction_path / self.label_rgb.strip('/')}")
            return
        
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


    def extract_keyframes(self, stride: int = 2, force: bool = False, visualize: bool = False) -> None:

        if self._extracted(self.label_keyframes) and not force:
            print(f"[{self.logging_tag}] Keyframes already extracted to {self.visual_registration_output_path / self.label_keyframes.strip('/')}")
            return
        
        if not (
            self._extracted(self.label_rgb)
            or self._extracted(self.label_depth)
        ):
            raise FileNotFoundError(
            f"[{self.logging_tag}] No RGB, depth data found in "
            f"{self.extraction_path / self.label_rgb.strip('/')} or "
            f"{self.extraction_path / self.label_depth.strip('/')}"
            )
        
        out_dir = self.extraction_path / self.label_keyframes.strip("/")
        ensure_dir(out_dir)

        rgb_files = sorted(
            self.extraction_path.glob(f"{self.label_rgb.strip('/')}/**/*.jpg"),
            key=lambda x: int(x.stem)
        )

        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )
    
        # setup the keyframe extractor
        keyframe_extractor = KeyframeExtractor(
            rgb_files=rgb_files,
            depth_files=depth_files,
            n_keyframes=20,
            stride=stride)

        if not any(self.statistics.values()):
            print(f"[{self.logging_tag}] No statistics found, computing...")
            self.get_statistics(stride=stride)
        
        keyframe_extractor.statistics = self.statistics
        if visualize:
            keyframe_extractor.visualize_statistics()
        self.selected_keyframes = keyframe_extractor.extract_keyframes()
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

    def _extracted(self, label: str) -> bool:
        """
        Check if the data for the given label has been extracted.
        """
        label_path = self.extraction_path / label.strip("/")
        return label_path.exists() and any(label_path.iterdir())
    
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

        # get depth files sorted by timestamp
        depth_files = sorted(
            self.extraction_path.glob(f"{self.label_depth.strip('/')}/**/*.npy"),
            key=lambda x: int(x.stem)
        )

        keyframe_extractor = KeyframeExtractor(
            rgb_files=rgb_files,
            depth_files=depth_files,
            n_keyframes=20,
            stride=stride
        )

        self.statistics = keyframe_extractor.get_statistics(
            force=force
        )
        
        # Save the statistics to a JSON file
        self.save_statistics(statistics_file)


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

    location = False
    test = True
    if location:
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

    if test:
        rec_location = "bedroom_1"
        base_path = Path(f"/data/ikea_recordings")

        data_indexer = RecordingIndex(
            os.path.join(str(base_path), "raw") 
        )

        iphone_queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction="gripper", 
            recorder="iphone_1*"
        )

        for loc, inter, rec, ii, path in iphone_queries_at_loc:
            print(f"Found recorder: {rec} at {path}")

            rec_type = inter
            rec_module = rec
            interaction_indices = ii

            iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)

            iphone_data.extract_keyframes()

        a = 2