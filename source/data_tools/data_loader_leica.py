from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import open3d as o3d
import pye57
from tqdm import tqdm
import math
import json


class LeicaData:
    def __init__(self, base_path: Path, scan_idx: int = 0, voxel: float = 0.02*6,):
    
        self.base_path = base_path
        self.e57_pre = base_path / "raw" / "Leica" / "e57" / "dlab_pre.e57"
        self.e57_post = base_path / "raw" / "Leica" / "e57" / "dlab_post.e57"
        self.pre_cache_dir = base_path / "extracted" / "Leica" / "pre"
        self.post_cache_dir = base_path / "extracted" / "Leica" / "post"

        self.pre_cache_dir = Path(self.pre_cache_dir)
        self.post_cache_dir = Path(self.post_cache_dir)

        self.e57_pre_data = None
        self.e57_post_data = None

        self.scan_idx = scan_idx
        self.voxel = voxel

        stem_pre = self.e57_pre.stem + f"_scan{scan_idx}"
        self.ply_pre_path = self.pre_cache_dir / "points" / f"{stem_pre}.ply"
        self.down_pre_path = self.pre_cache_dir / "points_downsampled" / f"{stem_pre}_voxel{voxel:.3f}.ply"
        self.fpfh_pre_path = self.pre_cache_dir / "features_fpfh" / f"{stem_pre}_voxel{voxel:.3f}_fpfh.npz"
        self.renderings_pre_path = self.pre_cache_dir / "renderings" / f"{stem_pre}"
        self.features_2D_pre_path = self.pre_cache_dir / "renderings" / "features_2D"

        stem_post = self.e57_pre.stem + f"_scan{scan_idx}"
        self.ply_post_path = self.post_cache_dir / "points" / f"{stem_post}.ply"
        self.down_post_path = self.post_cache_dir / "points_downsampled" / f"{stem_post}_voxel{voxel:.3f}.ply"
        self.fpfh_post_path = self.post_cache_dir / "features_fpfh" / f"{stem_post}_voxel{voxel:.3f}_fpfh.npz"
        self.renderings_post_path = self.post_cache_dir / "renderings" 
        self.features_2D_post_path = self.post_cache_dir / "renderings" 

        for path in [
            self.ply_pre_path,
            self.down_pre_path,
            self.fpfh_pre_path,
            self.ply_post_path,
            self.down_post_path,
            self.fpfh_post_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

        self._e57_to_ply(scan="pre")
        self._e57_to_ply(scan="post")
        
    def get_full_cloud(self, scan: str, force: bool = False) -> o3d.geometry.PointCloud:
        """Return the full-resolution cloud, converting from E57 if needed."""

        if scan not in ["pre", "post"]:
            raise ValueError("Scan must be either 'pre' or 'post'.")
        if scan == "pre":
            self.e57_path = self.e57_pre
            self.ply_path = self.ply_pre_path
        else:
            self.e57_path = self.e57_post
            self.ply_path = self.ply_post_path

        if not self.ply_path.exists() or force:
            self._e57_to_ply()
        return o3d.io.read_point_cloud(str(self.ply_path))
    
    def get_downsampled(self, scan: str, force: bool = False) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        Returns (downsampled_cloud).  Caches both to disk.
        """

        if scan not in ["pre", "post"]:
            raise ValueError("Scan must be either 'pre' or 'post'.")
        if scan == "pre":
            down_path = self.down_pre_path
        else:
            down_path = self.down_post_path

        if force or not (down_path.exists()):
            self._make_downsampled(scan=scan)

        down = o3d.io.read_point_cloud(str(down_path))
        return down

    def _e57_to_ply(self, scan: str) -> None:

        if scan not in ["pre", "post"]:
            raise ValueError("Scan must be either 'pre' or 'post'.")
        if scan == "pre":
            e57_path = self.e57_pre
            ply_path = self.ply_pre_path
        else:
            e57_path = self.e57_post
            ply_path = self.ply_post_path

        if Path(ply_path).exists():
            print(f"[Leica] Full-resolution PLY already exists at {ply_path}")
            return

        print(f"[Leica] Reading {e57_path}  (scan {self.scan_idx})")
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
        
        print(f"[Leica] Saved full-resolution PLY → {ply_path}")

    def make_rendered_360_views(self, scan: str) -> None:
        if scan not in ("pre", "post"):
            raise ValueError("scan must be 'pre' or 'post'")
        out_dir = self.renderings_pre_path if scan == "pre" else self.renderings_post_path

        # ---------- Load point cloud and downsample ------------------------------
        pcd = self.get_full_cloud(scan=scan).voxel_down_sample(0.01)

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
        for i in tqdm(range(N), desc=f"Rendering {scan}"):

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


            # fixed tilt around new z axis by 30 degrees
            R_tilt = np.array([
                [1, 0, 0],
                [0, math.cos(math.radians(30)), -math.sin(math.radians(30))],
                [0, math.sin(math.radians(30)), math.cos(math.radians(30))]
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

            # ext[:3, 3] = translation + center

            # renderer.setup_camera(intrinsic, ext)
            # img = renderer.render_to_image()
            # o3d.io.write_image(str(out_dir / "rgb" / f"{i:04d}.png"), img, 9)



        print(f"[Leica] Rendered {N* 2} rotating tripod views → {out_dir/'images'}")

    def _make_downsampled(self, scan: str) -> None:
        if scan not in ["pre", "post"]:
            raise ValueError("Scan must be either 'pre' or 'post'.")
        if scan == "pre":
            down_path = self.down_pre_path
        else:
            down_path = self.down_post_path

        if down_path.exists():
            print(f"[Leica] Downsampled cloud and FPFH already exist at {down_path}")
            return

        print(f"[Leica] Loading full cloud for {scan} scan...")
        full = self.get_full_cloud(scan=scan)  # ensures .ply exists

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

            print("[Leica] Saving downsampled cloud and FPFH...") # stored in PLY comment
            o3d.io.write_point_cloud(str(down_path), down, write_ascii=False)
            pbar.update(1)

        print(
            f"[Leica] Cached ↓ cloud → {down_path.name}"
        )




if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")


    leica_data = LeicaData(base_path)

    # pcd_pre_down, fpfh_pre = leica_data.get_down_and_fpfh(scan="pre")
    # leica_data._make_rendered_360_views(scan="pre")
    leica_data.make_rendered_360_views(scan="post")