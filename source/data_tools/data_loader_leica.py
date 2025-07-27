from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import open3d as o3d
import pye57
from tqdm import tqdm
import math
import json
import re
import cv2
import OpenEXR, Imath
from scipy.spatial.transform import Rotation as R
from data_indexer import RecordingIndex
import os
import scipy.io as sio

class LeicaData:
    def __init__(self, base_path: Path, rec_loc: str, initial_setup: str, voxel: float = 0.05) -> None:
    
        self.base_path = base_path
        self.rec_loc = rec_loc

        self.extraction_path = base_path / "extracted" / rec_loc / "leica"

        self.leica_path_raw = base_path / "raw" / rec_loc / "leica"

        self.initial_setup = initial_setup
        self.setups = []

        self.label_points = "points"
        self.label_downsampled = "points_downsampled"
        self.label_renderings = "renderings"
        self.label_pano_tiles = "pano_tiles"
        self.label_images = "images"
        self.label_mesh = "mesh"

        self.voxel = voxel

        self.extract_all_setups()

        
    def extract_all_setups(self) -> Tuple[Path, Path]:

        """ Extracts setups from the raw Leica data directory.
        This method scans the raw directory for files containing "Setup ..." in their names,
        and creates a directory for each setup in the extraction path.
        """

        if self._extracted():
            print(f"[Leica] Data already extracted for {self.rec_loc}.")
            self.setups = sorted([d.name for d in self.extraction_path.iterdir() if d.is_dir()])
            return

        # List files in raw directory
        raw_files = list(self.leica_path_raw.glob("*"))

        # Find files containing "Setup ..." in their names and get all setups
        setup_files = [f for f in raw_files if "Setup " in f.stem]
        setups = [f.stem.split("Setup ")[-1] for f in setup_files]
        setups = [s[0:3] for s in setups]  # take only the first 3 characters
        self.setups = sorted(set(setups))  # unique setups


        for setup in self.setups:
            pattern = f"Setup {setup}"
            # find all files in dir that contrain the pattern
            matching_files = [f for f in raw_files if pattern in f.stem]
            # create setup dir in extration path
            setup_dir = self.extraction_path / f"{setup}"
            setup_dir.mkdir(parents=True, exist_ok=True)
            # copy files to setup dir
            for file in matching_files:
                if file.suffix == ".e57":
                    target_file = setup_dir / self.label_points / f"points.ply"
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        self._points_e57_to_ply(e57_path=file, ply_path=target_file)
                        print(f"[Leica] Converted {file.name} to PLY and saved to {target_file}")
                    # downsample
                    self._make_downsampled(setup=setup)
                else:
                    target_file = setup_dir / self.label_images / file.name
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        file.rename(target_file)
                        print(f"[Leica] Moved {file.name} to {target_file}")

        print(f"[Leica] Extracted {len(self.setups)} setups from {self.rec_loc}.")

    
    def get_downsampled_points(self, setup: str | None = None) -> o3d.geometry.PointCloud:
        """
        Returns downsampled_cloud.  Caches both to disk.
        """

        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")
        
        down_path = self.extraction_path / setup / self.label_downsampled / f"points_voxel_{self.voxel:.3f}.ply"

        return o3d.io.read_point_cloud(str(down_path))
    
    def get_full_points(self, setup: str | None = None) -> o3d.geometry.PointCloud:
        """
        Returns the full point cloud for the given setup.
        Caches the PLY file if it does not exist.
        """
        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        ply_path = self.extraction_path / setup / self.label_points / f"points.ply"

        return o3d.io.read_point_cloud(str(ply_path))
    
    def get_mesh(self, setup: str | None = None) -> o3d.t.geometry.TriangleMesh:
        """
        Returns the mesh for the given setup.
        Caches the mesh if it does not exist.
        """
        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        mesh_path = self.extraction_path / setup / self.label_mesh / f"mesh.ply"

        if not mesh_path.exists():
            print(f"[Leica] Mesh not found at {mesh_path}. Exists only for the first setup of the recording.")

        return o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(mesh_path)))
    
    def get_panos(self, setup: str | None = None) -> dict:
        """
        Returns a dictionary containing the paths to the panorama images and their associated metadata. 
        The dictionary contains the following keys:
        - "depth": Path to the depth image (EXR file).
        - "rgb": Path to the RGB image (PNG file).
        - "pose": Parsed pose information from the TXT file.
        """
        if setup is None:
            setup = self.setups[0]

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        panos_dir = self.extraction_path / setup / self.label_images
    
        # Check if the panos directory contains 1 JPG, 1 EXR, 2 PNGs, and 1 TXT
        if not panos_dir.exists():
            raise FileNotFoundError(f"Panos directory not found: {panos_dir}")

        jpg_files = list(panos_dir.glob("*.jpg"))
        exr_files = list(panos_dir.glob("*.exr"))
        png_files = list(panos_dir.glob("*.png"))
        txt_files = list(panos_dir.glob("*.txt"))

        if len(jpg_files) != 1:
            raise ValueError(f"Expected 1 JPG file in {panos_dir}, found {len(jpg_files)}")
        if len(exr_files) != 1:
            raise ValueError(f"Expected 1 EXR file in {panos_dir}, found {len(exr_files)}")
        if len(png_files) != 2:
            raise ValueError(f"Expected 2 PNG files in {panos_dir}, found {len(png_files)}")
        if len(txt_files) != 1:
            raise ValueError(f"Expected 1 TXT file in {panos_dir}, found {len(txt_files)}")

        # get files
        pose_info_file = list(panos_dir.glob(f"*{setup}.txt"))[0]
        depth_pano_file = list(panos_dir.glob(f"*{setup}.exr"))[0]
        rgb_pano_file = list(panos_dir.glob(f"*{setup}.png"))[0]

        depth_pano = self._read_exr(str(depth_pano_file))
        
        rgb_pano = cv2.imread(rgb_pano_file, cv2.IMREAD_UNCHANGED)

        panos = {}

        panos["depth"] = depth_pano
        panos["rgb"] = rgb_pano
        panos["pose"] = self._parse_pano_pose(pose_info_file)
        
        return panos

    def _points_e57_to_ply(self, e57_path: str | Path, ply_path: str | Path) -> None:

        e57 = pye57.E57(str(e57_path))
        data = e57.read_scan(self.scan_idx, colors=True, ignore_missing_fields=True)

        xyz = np.column_stack(
            (data["cartesianX"], data["cartesianY"], data["cartesianZ"])
        ).astype(np.float32)

        # Filter invalids ------------------------------------------------------
        mask = ~np.isnan(xyz).any(axis=1)
        xyz = xyz[mask]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

        if {"colorRed", "colorGreen", "colorBlue"}.issubset(data.keys()):
            rgb = np.stack(
                (data["colorRed"], data["colorGreen"], data["colorBlue"]),
                axis=1
            ).astype(np.float32) / 255.0
            rgb = rgb[mask]
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        if {"normalX", "normalY", "normalZ"}.issubset(data.keys()):
            normals = np.column_stack(
                (data["normalX"], data["normalY"], data["normalZ"])
            ).astype(np.float32)
            normals = normals[mask]
            pcd.normals = o3d.utility.Vector3dVector(normals)

        with tqdm(total=1, desc="Saving PLY", unit="file") as pbar:
            o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
            pbar.update(1)        
        
        print(f"[Leica] Saved full-resolution PLY at {ply_path}")

    def make_rendered_360_views(self, setup: str | None = None) -> None:
        
        if setup is None:
            setup = self.setups[0]  # default to first setup

        if setup not in self.setups:
            raise ValueError(f"Setup {setup} not found in {self.setups}")

        out_dir = self.extraction_path / setup / self.label_renderings

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[Leica] Renderings already exist for setup {setup} at {out_dir}.")
            return

        # ---------- Load point cloud and downsample ------------------------------
        pcd = self.get_full_points(setup=setup)#.voxel_down_sample(0.01)

        # ---------- Camera intrinsics --------------------------------------------
        N, fov, W, H = 20, 80, 640, 480
        fx = fy = (W / 2) / math.tan(math.radians(fov / 2))
        cx, cy = W / 2, H / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        # ---------- Setup renderer ------------------------------------------------
        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit" if pcd.has_colors() else "defaultUnlit"
        renderer.scene.add_geometry("leica", pcd, mat)
        renderer.scene.set_background([1, 1, 1, 1])

        (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (out_dir / "poses").mkdir(parents=True, exist_ok=True)

        center = np.zeros(3)  # Leica scanner at origin

        # ---------- Render loop ---------------------------------------------------
        from tqdm import tqdm
        for i in tqdm(range(N), desc=f"Rendering {setup}"):

            r = 1.5  # distance from the center of the scanner
            if i < N / 2:
                yaw = 2 * math.pi * i / (N / 2)  
                translation_low = np.array([r * math.cos(yaw), -0.3, r * math.sin(yaw)])
                translation_high = np.array([r * math.cos(yaw), 0.3, r * math.sin(yaw)])
            else:
                yaw = 2 * math.pi * (-(i - N)) / (N / 2) - math.pi
                translation_low = np.array([r * math.cos(yaw + math.pi), -0.3, r * math.sin(yaw + math.pi)])
                translation_high = np.array([r * math.cos(yaw + math.pi), 0.3, r * math.sin(yaw + math.pi)])

            translation_no_radius = np.array([0, 0, 0])

            # Convert from Z-up to Y-up coordinate system
            R = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])

            R_1 = np.array([
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)]
            ])

            ext = np.eye(4, dtype=float)
            ext[:3, :3] = R_1 @ R

            for idx, translation in enumerate([translation_low, translation_high, translation_no_radius]):
                ext[:3, 3] = translation + center
                renderer.setup_camera(intrinsic, ext)
                img = renderer.render_to_image()
                name = idx + i * 3
                o3d.io.write_image(str(out_dir / "rgb" / f"{name:04d}.png"), img, 9)

                meta = dict(
                width=W,
                height=H,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                extrinsic=ext.tolist())


                with open(out_dir / "poses" / f"{name:04d}.json", "w") as f:
                    json.dump(meta, f, indent=2)

        print(f"[Leica] Rendered {N* 2} rotating tripod views → {out_dir/'images'}")

    def make_360_views_from_pano(self, setup: str | None = None) -> None:
        # TODO visualize the camera fustrum in the panorama/pcd
        """
        Slice a full-equirect pano into 45degx45deg pinhole crops (32 total),
        writing both RGB + depth tiles + 3D points and a per-tile JSON with intrinsics+pose.
        """
        if setup is None:
            setup = self.setups[0]

        pano_data = self.get_panos(setup=setup)
        rgb_pano  = pano_data["rgb"]
        depth_pano= pano_data["depth"]
        meta      = pano_data["pose"] ["HDR"]
           # {"t": [...], "q": [...]}

        # load once
        equi_rgb   = rgb_pano
        equi_depth = depth_pano

        # output dirs
        out_base = self.extraction_path / setup / "pano_tiles"
        rgb_out  = out_base / "rgb"
        depth_out= out_base / "depth"
        depth_vis_out = out_base / "depth_vis"
        pose_out = out_base / "poses"
        xyz_out = out_base / "xyz"
        for d in (rgb_out, depth_out, pose_out, xyz_out, depth_vis_out):
            d.mkdir(parents=True, exist_ok=True)

        # tiling params
        hfov = 90.0
        vfov = 120.0
        W, H = 1024, 1364
        step = hfov * 0.1

        # get initial camera pose and rots around world axes
        euler0 = R.from_quat(meta["orientation"], scalar_first=True).as_euler("xyz", degrees=True)
        rot_initial_around_world_z = euler0[2]
        rot_initial_around_world_y = euler0[1]  
        rot_initial_around_world_x = euler0[0]  

        # define default camera pose for tiel cropping and rendering
        # no rotation, translation from metadata
        t0 = np.array(meta["position"])
        R_wc_initial = R.from_quat([1.0, 0.0, 0.0, 0.0], scalar_first=True).as_matrix()

        # create a grid of yaw angles to cover the full 360 degrees
        yaws = np.arange(0.0, 360.0, step) 

        T_o3d_leica = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
        
        R_o3d_leica = T_o3d_leica[:3, :3]  # 3x3 rotation matrix
        
        idx = 0
        for yaw_deg in yaws:
            yaw   = math.radians(yaw_deg)

            # compute target camera pose
            # yaw the camera around its own y-axis for horizontal rotation in camera frame
            R_yaw = R.from_euler("y", yaw, degrees=False).as_matrix()
            R_wc_target = R_yaw # can prolly remove R_wc_initial, since we use identny quat for initial pose
            t_wc_target = t0 

            # compute new camera pose in world coordinates after rotation
            ext = np.eye(4, dtype=float)
            ext[:3, :3] = R_wc_target  # convert from Leica to Open3D coordinate system
            ext[:3, 3] = t_wc_target
            ext_inv = np.linalg.inv(ext)  # convert from Open3D to Leica coordinate system
            ext_inv_o3d = np.linalg.inv(ext) @ T_o3d_leica  # convert from Leica to Open3D coordinate system  # convert from Open3D to Leica coordinate system
            ext_save = np.linalg.inv(T_o3d_leica) @ ext  # save in Open3D coordinate system

            #RGB pinhole tile in the equirectangular image
            # adjust yaw to match the initial rotation for crops
            # this is needed to align the crops with the original panorama
            yaw_adjusted = yaw_deg + rot_initial_around_world_z  
            R_yaw = R.from_euler("y", yaw_adjusted, degrees=True).as_matrix()
            tile_rgb = self._equirect_to_pinhole(
                equi_rgb, R_yaw, hfov, vfov, W, H
            )

            # write RGB tile
            fn = f"{idx:03d}.jpg"
            cv2.imwrite(str(rgb_out/fn), tile_rgb)

            # compute intrinsics
            fx = (W/2) / math.tan(math.radians(hfov/2))
            fy = (H/2) / math.tan(math.radians(vfov/2))
            K  = np.array([[fx,0,W/2],
                [0, fy,H/2],
                [0,  0,  1]])

            # render depth from mesh
            mesh = self.get_mesh(setup=setup)
            tile_depth = self._render_depth(mesh, K, ext_inv_o3d)

            # 2) write depth tile
            depth_fn = fn.replace(".jpg", ".exr")
            self._write_exr(str(depth_out/depth_fn), tile_depth)

            # 3) write depth visualization
            depth_vis_fn = fn.replace(".jpg", "_vis.png")
            self._write_depth_vis(str(depth_vis_out/depth_vis_fn), tile_depth)

            # 4) write xyz tile to mat
            xyz_tile = self._depth_to_world_xyz(tile_depth, K,  ext_save) #ext_save
            xyz_fn = fn + ".mat"
            self._save_mat(str(xyz_out/xyz_fn), rgb_image=tile_rgb, xyz_array=xyz_tile)

            q = R.from_matrix(ext_save[:3, :3]).as_quat(scalar_first=False) 
             # convert to quaternion

            # 5) Dump JSON
            pose = {
                "w_T_wc": ext_save.tolist(),
                "K": K.tolist(),
                "h": H,
                "w": W}
            
            with open(pose_out/fn.replace(".jpg",".json"), "w") as f:
                json.dump(pose, f, indent=2)

            idx += 1

        print(f"[LEICA] Exported {idx} tiles → {out_base}")


    def _make_downsampled(self, setup: str) -> None:

        down_path = self.extraction_path / setup / self.label_downsampled / f"points_voxel_{self.voxel:.3f}.ply"

        if down_path.exists():
            print(f"[Leica] Downsampled cloud already exist at {down_path}")
            return
        
        # ensure dir exists
        down_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[Leica] Loading full cloud for setupn {setup} ...")
        full = self.get_full_points(setup=setup)  # ensures .ply exists

        with tqdm(total=4, desc="[Leica] Downsample", unit="step") as pbar:
            print(f"[Leica] Down-sampling at voxel={self.voxel:.3f}")
            down = full.voxel_down_sample(voxel_size=self.voxel)
            pbar.update(1)

            print("[Leica] Estimating normals...")
            down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel * 2.0, max_nn=30
                )
            )
            pbar.update(1)

            print("[Leica] Saving downsampled cloud ...") # stored in PLY comment
            o3d.io.write_point_cloud(str(down_path), down, write_ascii=False)
            pbar.update(1)

        print(
            f"[Leica] Cached downsampled cloud → {down_path.name}"
        )

    def _extracted(self) -> bool:
        """
        Check if the data for the given label has been extracted.
        """
        label_path = self.extraction_path 
        return label_path.exists() and any(label_path.iterdir())

    def _parse_pano_pose(self, txt_path: str | Path):
        """
        Parses a Leica panorama TXT file containing lines like:
        position = [x, y, z];
        orientation = [qx, qy, qz, qw];
        Returns a dict with keys "LDR" and "HDR", each a dict with "position" and "orientation" lists.
        """
        data = {"LDR": {}, "HDR": {}}
        section = None

        # regex to capture numbers inside the brackets
        array_re = re.compile(r"\[([^\]]+)\]")
        
        if isinstance(txt_path, str):
            txt_path = Path(txt_path)

        for line in txt_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("Ldr Image"):
                section = "LDR"
            elif line.startswith("Hdr Image"):
                section = "HDR"
            elif section and line.startswith("position"):
                m = array_re.search(line)
                if m:
                    nums = [float(x) for x in m.group(1).split(",")]
                    data[section]["position"] = nums
            elif section and line.startswith("orientation"):
                m = array_re.search(line)
                if m:
                    nums = [float(x) for x in m.group(1).split(",")]
                    data[section]["orientation"] = nums
        return data
    

    def _equirect_to_pinhole(self,
                            equi_img: np.ndarray,
                            rot_mat: np.ndarray,
                            hfov_deg: float,
                            vfov_deg: float,
                            out_w: int,
                            out_h: int) -> np.ndarray:
        """
        Turn an equirectangular image (H_e x W_e) into a pinhole view
        at (yaw, pitch) with horizontal FOV=hfov_deg and vertical FOV=vfov_deg.
        Returns an out_h x out_w x C BGR image.
        """
        H_e, W_e = equi_img.shape[:2]

        # compute the tangent extents for each axis
        tan_h = math.tan(math.radians(hfov_deg / 2))
        tan_v = math.tan(math.radians(vfov_deg / 2))

        # screen coords in camera space
        xs = np.linspace(-tan_h, +tan_h, out_w)
        ys = np.linspace(-tan_v, +tan_v, out_h)
        xv, yv = np.meshgrid(xs, -ys)       # note the -ys to flip vertically
        zv = np.ones_like(xv)

        dirs = (rot_mat @ np.stack([xv, yv, zv], -1).reshape(-1,3).T).T

        # convert to spherical coords
        lon = np.arctan2(dirs[:,0], dirs[:,2])   # range [-π, π]
        lat = np.arcsin(dirs[:,1] / np.linalg.norm(dirs, axis=1))  # [-π/2, π/2]

        # map to equirectangular pixel coords
        uf = (lon / (2 * math.pi) + 0.5) * W_e
        vf = (0.5 - lat / math.pi) * H_e

        map_x = uf.reshape(out_h, out_w).astype(np.float32)
        map_y = vf.reshape(out_h, out_w).astype(np.float32)

        # sample with wrap‑around horizontally
        return cv2.remap(
            equi_img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
    
    def _render_equirect_depth(self,
                               pcd: o3d.geometry.PointCloud,
                               width: int,
                               height: int,
                               vis: bool = True) -> np.ndarray:
        """
        Render depth from a point cloud as an equirectangular image.
        Args:
            pcd: Open3D point cloud.
            width: Width of the output equirectangular image.
            height: Height of the output equirectangular image.
        Returns:
            depth: Depth image as a NumPy array of shape (height, width).
        """


        vox = self._pcd_to_voxel_grid(pcd, voxel=self.voxel)
        
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_voxel_grid(vox)

        # build longitude/latitude rays
        lon = (np.arange(width)   / width)  * 2*np.pi - np.pi     # [-π, +π]
        lat = (0.5 - np.arange(height) / height) * np.pi          # [+π/2…−π/2]
        lon, lat = np.meshgrid(lon, lat)
        dirs = np.stack([ np.sin(lon)*np.cos(lat),
                        np.sin(lat),
                        np.cos(lon)*np.cos(lat)], -1)   # shape (H,W,3)
        dirs = dirs.reshape(-1,3).astype(np.float32)

        origin = np.zeros_like(dirs)                 # scanner at (0,0,0)
        rays = np.hstack([origin, dirs])
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        depth = ans['t_hit'].numpy().reshape(height, width)
        depth[depth == np.inf] = 0.0

        if vis:
            # visualize depth plt
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('plasma')
            vmin = np.nanmin(depth[depth > 0])          # ignore zeros if they mark missing rays
            vmax = np.nanmax(depth)

            plt.figure(figsize=(8, 6))
            plt.imshow(depth, vmin=vmin, vmax=vmax, cmap=cmap)
            plt.colorbar(label='depth (m)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return depth

    def _depth_to_world_xyz(self, depth: np.ndarray, K: np.ndarray, w_T_wc: np.ndarray) -> np.ndarray:
        
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))  # pixel coordinates

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Reconstruct 3D camera coordinates
        x_cam = ((u - cx) * depth) / fx
        y_cam = ((v - cy) * depth) / fy
        z_cam = depth

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
        pts_cam_flat = pts_cam.reshape(-1, 3)  # (H*W, 3)

        # Convert camera-to-world rotation
        R_wc = w_T_wc[:3, :3]  # 3×3 rotation matrix
        t_wc = w_T_wc[:3, 3]   # 3×1
        pts_w = (R_wc @ pts_cam_flat.T + t_wc[:, None]).T  # 3×N

        return pts_w.reshape(H, W, 3).astype(np.float32)


    def _render_depth(self,
                    mesh: o3d.t.geometry.TriangleMesh,
                    K: np.ndarray,
                    w_T_wc: np.ndarray,
                    clip_max_dist: float = 10.0,
                    ):
        """
        Renders a depth map for the current pin‑hole view.

        Returns
        -------
        depth : (H, W) float32            — always
        xyz   : (H, W, 3) float32 or None — only if `return_xyz`
        """

        K = np.asarray(K, dtype=np.float64)
        width  = int(round(K[0, 2] * 2))
        height = int(round(K[1, 2] * 2))

        # transform from Leica world to Open3D world
        # to o3d coord system
        # Convert from Z-up to Y-up coordinate system
        R_o3d_leica = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        T_o3d_leica = np.eye(4, dtype=float)
        T_o3d_leica[:3, :3] = R_o3d_leica

        mesh_o3d = mesh.clone().transform(T_o3d_leica)

        # --- set‑up off‑screen renderer --------------------------------
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0, 0, 0, 0])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh.to_legacy(), mat)

        renderer.setup_camera(K, w_T_wc, width, height)

        # --- render -----------------------------------------------------
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)
        depth = np.asarray(depth_img, dtype=np.float32)  # (H, W)

        # clip depth values
        depth[depth > clip_max_dist] = 0.0  # mark as invalid
        depth[depth < 0.01] = 0.0           # mark as

        return depth


    def _pcd_to_mesh(self,
                pcd: o3d.geometry.PointCloud,
                depth: int = 8,         # Poisson reconstruction depth
                scale: float = 1.1) -> o3d.t.geometry.TriangleMesh:
        """
        Turns a point‑cloud into a watertight triangle mesh (Poisson) 
        Returns an *Open3D‑Tensor* mesh (ready for cuda or cpu scene).
        """
        pcd = pcd.voxel_down_sample(0.01)     # optional light decimation
        pcd.estimate_normals()

        mesh_legacy, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=scale)
        mesh_legacy.remove_duplicated_vertices()
        mesh_legacy.remove_degenerate_triangles()
        mesh_legacy.remove_non_manifold_edges()

        # convert to Tensor mesh on the default device (CPU or CUDA)
        return o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    def _write_exr(self, path: str, img: np.ndarray):
        """
        Write a single‐channel or 3‐channel float32 NumPy image to an EXR.
        The img must be H×W (single‐channel) or H×W×3 (RGB), dtype=float32.
        """
        assert img.dtype == np.float32, "convert to float32 first"
        H, W = img.shape[:2]

        # 1) Create an empty header with the correct size
        header = OpenEXR.Header(W, H)

        # 2) Define your channels in that header
        #    Here we assume RGB; if single‐channel, just define 'R'
        if img.ndim == 2:
            header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        else:
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }

        # 3) Open the file for writing
        exr = OpenEXR.OutputFile(path, header)

        # 4) Prepare raw byte strings per channel
        if img.ndim == 2:
            # single‐channel
            r_chan = img
            channel_data = {'R': r_chan.tobytes()}
        else:
            # RGB, split into three planes
            r_chan, g_chan, b_chan = cv2.split(img)
            channel_data = {
                'R': r_chan.tobytes(),
                'G': g_chan.tobytes(),
                'B': b_chan.tobytes()
            }

        # 5) Write out the pixels
        exr.writePixels(channel_data)
        exr.close()

    def _read_exr(self, path: str) -> np.ndarray:
        """
        Read a single‐channel EXR file into a NumPy array.
        Assumes the EXR has a channel named "R" with float32/float16/uint32 data.
        Returns an H×W NumPy array with the pixel values.
        """
        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        # determine dtype from the channel’s pixel type
        chan = header['channels']['R'].type
        if   chan == Imath.PixelType(Imath.PixelType.FLOAT): dtype = np.float32
        elif chan == Imath.PixelType(Imath.PixelType.HALF):  dtype = np.float16
        elif chan == Imath.PixelType(Imath.PixelType.UINT):  dtype = np.uint32
        else:
            raise ValueError(f"Unsupported EXR channel type: {chan}")

        # read raw bytes and convert
        raw = exr.channel('R', chan)
        arr = np.frombuffer(raw, dtype=dtype)

        # reshape into H×W
        arr = arr.reshape(H, W)

        return arr
    
    def _save_mat(self, mat_path: str | Path, rgb_image: np.ndarray, xyz_array) -> None:

        sio.savemat(str(mat_path),
                    {"RGBcut": rgb_image, "XYZcut": xyz_array},
                    do_compression=False)
        
    def _load_mat_as_ply(self, mat_path: str | Path) -> o3d.geometry.PointCloud:
        """
        Load a .mat file containing RGB and XYZ data and convert it to an Open3D PointCloud.
        The .mat file should contain 'RGBcut' and 'XYZcut' keys.
        """
        mat_data = sio.loadmat(str(mat_path))
        rgb = mat_data['RGBcut'].astype(np.float32) / 255.0
        xyz = mat_data['XYZcut'].astype(np.float32)

        # filter out zero coords
        valid_mask = np.all(xyz != 0, axis=2)
        xyz = xyz[valid_mask]
        rgb = rgb[valid_mask]

        if rgb.ndim == 2:  # single channel
            rgb = np.repeat(rgb[:, :, np.newaxis], 3, axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3)) 
        return pcd
        
    def _write_depth_vis(self, path: str, depth: np.ndarray) -> None:
        # Mask out invalid (zero or NaN) values
        valid_mask = (depth > 0) & np.isfinite(depth)
        if not np.any(valid_mask):
            print(f"Warning: No valid depth values to visualize in {path}")
            return

        # Compute per-image min/max for normalization
        min_depth = np.min(depth[valid_mask])
        max_depth = np.max(depth[valid_mask])
        
        # Normalize to [0, 255] and convert to uint8
        depth_vis = np.clip(depth, min_depth, max_depth)
        depth_vis = ((depth_vis - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Save
        cv2.imwrite(str(path), depth_colored)

    def show_crop_frustums(
            self,
            setup: str | None = None,
            pose_dir: Path | None = None,
            mesh: o3d.t.geometry.TriangleMesh | None = None,
            frustum_depth: float = 0.5,       # metres from pin‑hole to image‑plane
            max_tiles: int | None = None,     # None = show all
            color: tuple[float, float, float] = (1, 0, 0),
        ) -> None:
        """
        Visualise the mesh + camera frusta for every 45°×45° crop tile.

        *Requirements*: the JSON files produced by `make_360_views_from_pano()`
        must live in   .../pano_tiles/poses/###.json.

        Args
        ----
        setup          : Leica setup name (defaults to first in `self.setups`)
        pose_dir       : override path to the pose JSONs
        mesh           : pass your own mesh if you already have it in memory
        frustum_depth  : distance from camera centre to image plane (m)
        max_tiles      : display only the N first tiles (speed / clarity)
        color          : RGB line colour for all frusta  (0‑1 floats)
        """
        if setup is None:
            setup = self.setups[0]

        if mesh is None:
            mesh = self.get_mesh(setup)                        # t.geometry
        mesh_legacy = mesh.to_legacy()

        pcd = self.get_downsampled_points(setup=setup)  # t.geometry

        R_o3d_leica = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])

        T_o3d_leica = np.eye(4, dtype=float)
        T_o3d_leica[:3, :3] = R_o3d_leica
        # mesh_legacy.transform(T_o3d_leica)  # convert to Open3D world coords

        # ---------------------------------------------------------------------
        # 1) Gather pose JSON files
        if pose_dir is None:
            pose_dir = self.extraction_path / setup / "pano_tiles" / "poses"
        pose_files = sorted(pose_dir.glob("*.json"))
        if max_tiles is not None:
            pose_files = pose_files[:max_tiles]

        # get mat files for xyz tiles
        xyz_files = sorted((pose_dir.parent / "xyz").glob("*.mat"))
        if max_tiles is not None:
            xyz_files = xyz_files[:max_tiles]

        # ---------------------------------------------------------------------
        # 2) Build one big LineSet with all frusta
        all_pts: list[np.ndarray]  = []
        all_lines: list[tuple[int, int]] = []
        all_cols: list[tuple[float, float, float]] = []
        all_mats: list[o3d.geometry.PointCloud] = []

        for idx, jf in enumerate(pose_files):
            with open(jf) as f:
                meta = json.load(f)

            # open the corresponding XYZ mat file
            if idx < 5:
                xyz_file = xyz_files[idx]
                xyz_pcd = self._load_mat_as_ply(xyz_file)
                # add the XYZ points from the mat file
                all_mats.append(xyz_pcd)

            K   = np.asarray(meta["K"], dtype=np.float64)
            w_T = np.asarray(meta["w_T_wc"], dtype=np.float64)
            W   = int(meta["w"])
            H   = int(meta["h"])

            # ----- camera‑space corner pixels at z = frustum_depth ----------
            corners_px = np.array([[0,   0,   1],
                                [W,   0,   1],
                                [W,   H,   1],
                                [0,   H,   1]], dtype=np.float64).T  # 3×4
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # un‑project (pinhole) → camera coords
            z     = np.full((1, 4), frustum_depth)
            x_cam = (corners_px[0] - cx) / fx * z
            y_cam = (corners_px[1] - cy) / fy * z
            cam_pts = np.vstack([x_cam, y_cam, z])              # 3×4

            # include optical centre at origin
            cam_pts = np.hstack([np.zeros((3, 1)), cam_pts])    # 3×5

            # ----- transform to world ---------------------------------------
            R = w_T[:3, :3]  # 3×3 rotation matrix
            t = w_T[:3, 3:4]
            world_pts = (R @ cam_pts) + t                       # 3×5

            base = len(all_pts)
            all_pts.extend(world_pts.T)                         # add 5 points

            # pyramid edges (indices relative to this frustum’s base)
            edges = [(0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (2, 3), (3, 4), (4, 1)]
            all_lines.extend([(base + a, base + b) for a, b in edges])
            all_cols.extend([color] * len(edges))


        # ---------------------------------------------------------------------
        # 3) Create a single LineSet
        ls = o3d.geometry.LineSet()
        ls.points  = o3d.utility.Vector3dVector(np.array(all_pts))
        ls.lines   = o3d.utility.Vector2iVector(np.array(all_lines, dtype=int))
        ls.colors  = o3d.utility.Vector3dVector(np.array(all_cols))

        # ---------------------------------------------------------------------
        # 4) Show everything
        o3d.visualization.draw_geometries([mesh_legacy, ls, pcd]+all_mats,
                                        zoom=0.6,
                                        window_name=f"Frusta for '{setup}'")

if __name__ == "__main__":
    from pathlib import Path

    base_path = Path(f"/data/ikea_recordings")
    rec_location = "bedroom_1"

    leica_data = LeicaData(base_path, rec_location, initial_setup="001")
    leica_data.extract_all_setups()
    leica_data.make_360_views_from_pano()
    leica_data.show_crop_frustums(max_tiles=20)
