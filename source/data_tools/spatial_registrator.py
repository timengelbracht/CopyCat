from pathlib import Path
import open3d as o3d
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, DefaultDict, Dict
import copy
import time
import os
import pickle
from scipy.spatial.transform import Rotation as R

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    triangulation
)

from hloc.localize_sfm import main as localize_sfm_main
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

if not hasattr(pycolmap, "absolute_pose_estimation"):
    def absolute_pose_estimation(points2D, points3D, camera,
                                 estimation_options=None,
                                 refinement_options=None):
        return pycolmap.estimate_and_refine_absolute_pose(
            points2D, points3D, camera,
            estimation_options or {},
            refinement_options or {},
        )
    
    pycolmap.absolute_pose_estimation = absolute_pose_estimation

if not hasattr(pycolmap.Rigid3d, "essential_matrix"):

    def _skew(t):
        tx, ty, tz = t
        return np.array([[ 0, -tz,  ty],
                         [ tz,   0, -tx],
                         [-ty,  tx,   0]])

    def essential_matrix(self: pycolmap.Rigid3d) -> np.ndarray:
        """
        Return E = [t]_x R  (world→left, right pose = self)
        Equivalent to the old C++ helper that existed in pycolmap < 0.7.
        """
        R = self.rotation.matrix()      
        t = self.translation            
        return _skew(t) @ R            

    pycolmap.Rigid3d.essential_matrix = essential_matrix

import shutil
from PIL import Image
import json
from scipy.spatial.transform import Rotation

from data_loader_aria import AriaData
from data_loader_leica import LeicaData
from data_loader_iphone import IPhoneData
from data_loader_gripper import GripperData

class SpatialRegistrator:
    """
    Class to handle spatial registration of sensor modules to Leica scans.
    """

    def __init__(self, loader_map: object, loader_query: object):
        
        if not isinstance(loader_map, LeicaData):
            raise TypeError("loader_map must be an instance of LeicaData.")
        
        if not isinstance(loader_query, (AriaData, IPhoneData, GripperData)):
            raise TypeError("loader_query must be an instance of AriaData, IPhoneData, or GripperData.")

        self.loader_map = loader_map
        self.loader_query = loader_query

        self.image_path_map = Path(self.loader_map.renderings_post_path / "rgb")
        self.pose_path_map = Path(self.loader_map.renderings_post_path / "poses")
        # TODO pre/post selection

        if isinstance(self.loader_query, AriaData):
            self.image_path_query = Path(self.loader_query.visual_registration_output_path / self.loader_query.label_keyframes.strip("/"))
            # TODO raw/undistorted selection
        elif isinstance(self.loader_query, IPhoneData):
            self.image_path_query = Path(self.loader_query.visual_registration_output_path / self.loader_query.label_keyframes.strip("/"))
        elif isinstance(self.loader_query, GripperData):
            # TODO - implement extraction for GripperData
            pass

        # HLoc configuration
        self.retrieval_conf = extract_features.confs["netvlad"]
        self.feature_conf = extract_features.confs["superpoint_inloc"]
        self.matcher_conf = match_features.confs["superglue"]

        self.visual_registration_output_path = self.loader_query.visual_registration_output_path  
        self.visual_registration_output_path.mkdir(parents=True, exist_ok=True)

        self.sfm_pairs = self.visual_registration_output_path / "outputs" / "pairs-netvlad.txt"
        self.loc_pairs = self.visual_registration_output_path / "outputs" / "pairs-loc.txt"
        self.sfm_dir = self.visual_registration_output_path / "outputs" / "sfm"
        self.initial_sfm_dir = self.visual_registration_output_path / "outputs" / "initial_sfm"
        self.results_dir = self.visual_registration_output_path / "outputs" / "results.txt"
        self.matches = self.visual_registration_output_path / "outputs" / f"{self.matcher_conf['output']}.h5"
        self.features = self.visual_registration_output_path / "outputs" / f"{self.feature_conf['output']}.h5"
        self.features_retrieval = self.visual_registration_output_path / "outputs" / f"{self.retrieval_conf['output']}.h5"

        self._set_up_visual_registration()

        self.images = self.visual_registration_output_path / "images"
        self.references = [p.relative_to(self.images).as_posix() for p in (self.images / "mapping/").iterdir()]
        self.query = [p.relative_to(self.images).as_posix() for p in (self.images / "query/").iterdir()]

        self.T_world_query = None

    def _set_up_visual_registration(self):
        """
        Set up the visual registration HLoc directory structure and copy images.
        """

        out_dir = self.visual_registration_output_path / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        image_dir = self.visual_registration_output_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_dir_mapping = image_dir / "mapping"
        image_dir_mapping.mkdir(parents=True, exist_ok=True)

        image_dir_query = image_dir / "query"
        image_dir_query.mkdir(parents=True, exist_ok=True)


        # Copy images to the mapping and query directories
        for image in tqdm(self.image_path_map.glob("*"), desc="Copying images to mapping", total=len(list(self.image_path_map.glob("*")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_mapping / f"{image.stem}.png"
                try:
                    img = Image.open(image)
                    img.convert("RGB").save(dest_path, "PNG")
                except Exception as e:
                    print(f"Failed to convert {image}: {e}")

        for image in tqdm(self.image_path_query.glob("*.png"), desc="Copying images to query", total=len(list(self.image_path_query.glob("*.png")))):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                dest_path = image_dir_query / f"{image.stem}.png"
                try:
                    img = Image.open(image)
                    img.convert("RGB").save(dest_path, "PNG")
                except Exception as e:
                    print(f"Failed to convert {image}: {e}")

        print(f"[REGISTRATION] Visual registration set up at {self.visual_registration_output_path}")

    def visual_registration(self, from_gt: bool = True):
        """
        End-to-end HLoc pipeline:
            1  NetVLAD for map + query
            2  SuperPoint + SuperGlue on map → SfM model
            3  NetVLAD query→map pairs  (single call)
            4  SuperGlue matching on those pairs
            5  localize_sfm with intrinsics
        """

        print(f"[REGISTRATION] Starting visual registration...")

        # delete old files in output directory
        if Path(self.visual_registration_output_path / "outputs").exists():
            shutil.rmtree(self.visual_registration_output_path / "outputs")

        extract_features.main(
            conf=self.feature_conf,
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features,
            overwrite=True)
        
        extract_features.main(
            conf=self.feature_conf, 
            image_dir=self.images,
            image_list=self.query,
            feature_path=self.features,
            overwrite=False)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir=self.images,
            image_list=self.references,
            feature_path=self.features_retrieval, 
            overwrite=True)

        extract_features.main(                       
            conf=self.retrieval_conf, 
            image_dir = self.images,
            image_list=self.query,
            feature_path=self.features_retrieval, 
            overwrite=False)

        pairs_from_retrieval.main(                  
            descriptors=self.features_retrieval,
            output=self.sfm_pairs, 
            num_matched=20,
            query_list=self.references,
            db_list=self.references)

        match_features.main(      
            conf=self.matcher_conf,
            pairs=self.sfm_pairs,
            features=self.features,
            matches=self.matches)

        if from_gt:
            self.create_reconstruction_from_gt_poses()
            triangulation.main(
                sfm_dir=self.sfm_dir,
                reference_model=self.initial_sfm_dir,
                image_dir=self.images,
                pairs=self.sfm_pairs,
                features=self.features,
                matches=self.matches,
                mapper_options=dict(
                    ba_refine_extra_params = False,
                    ba_refine_focal_length = False,
                    ba_refine_principal_point = False,
                    fix_existing_images = True),
                estimate_two_view_geometries = False,)
            model = pycolmap.Reconstruction(self.initial_sfm_dir)
        else:
            model = reconstruction.main(            
                sfm_dir=self.sfm_dir, 
                image_dir=self.images,
                image_list=self.references,
                pairs=self.sfm_pairs, 
                features=self.features, 
                matches=self.matches)

        pairs_from_retrieval.main(
            descriptors=self.features_retrieval,
            output=self.loc_pairs,
            num_matched=20,
            query_list=self.query,  
            db_list=self.references)

        match_features.main(
            conf=self.matcher_conf, 
            pairs=self.loc_pairs,
            features=self.features,
            matches=self.matches, 
            overwrite=True)
        
        good_refs = {j for _,j in map(str.split, open(self.loc_pairs))}
        self.references = good_refs
        
        query_list_path = self.visual_registration_output_path / "outputs" / "query_list.txt"
        with open(query_list_path, "w") as f:
            for name in self.query:
                K_pinhole = self.loader_query.calibration["K"]
                f_x = K_pinhole[0, 0]
                f_y = K_pinhole[1, 1]
                c_x = K_pinhole[0, 2]
                c_y = K_pinhole[1, 2]
                h = self.loader_query.calibration["h"]
                w = self.loader_query.calibration["w"]
                
                f.write(f"{name} PINHOLE {w} {h} {f_x} {f_y} {c_x} {c_y}\n")


        localize_sfm_main(self.sfm_dir, query_list_path, self.loc_pairs,
                        self.features, self.matches, self.results_dir)
        
        print(f"[REGISTRATION] Visual registration completed. Results saved to {self.results_dir}")

    def pcd_to_pcd_registration(self, vis: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        
        # get the point clouds from the map (leica) and query
        pcd_map_gt = self.loader_map.get_downsampled(scan="post")
        pcd_query = self.loader_query.get_downsampled()

        # get image timestamp and pose of query in Leica world frame
        pose_query = self.get_poses_query() 
        name = list(pose_query.keys())[0]

        # get the poses of the query images and compute rigid body transformation
        # between the query and the map
        T_was = []
        for name in pose_query.keys():
            T_wc = pose_query[name]["w_T_wc"]
            timestamp = int(name)

            if self.loader_query.sensor_module_name == "aria_human_ego":
                # get the pose of the Aria sensor module
                # c- camera
                # w- world
                # a- Aria
                # d- device
                T_ad = self.loader_query.get_mps_pose_at_timestamp(timestamp)
                if T_ad is None:
                    print(f"[!] No pose found for timestamp {timestamp}. Skipping.")
                    continue
                T_dc = self.loader_query.calibration["T_device_camera"]
                T_cRaw_cRect = self.loader_query.calibration["pinhole_T_device_camera"]
                T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_ad)
                T_wa = T_wc @ T_ca
                T_was.append(T_wa)
            elif self.loader_query.sensor_module_name == "iphone_left":
                raise NotImplementedError("iPhone data not implemented yet.")
            elif self.loader_query.sensor_module_name == "gripper":
                raise NotImplementedError("Gripper data not implemented yet.")


        # get average transformation and filter outliers
        T_wa = self.mean_transformation(T_was)

        pcd_query_aligned = copy.deepcopy(pcd_query)
        pcd_query_aligned.transform(T_wa)

        # icp alignment
        threshold = 0.02
        reg_icp = o3d.pipelines.registration.registration_icp(
            pcd_query_aligned, pcd_map_gt, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        w_pcd_query_aligned_icp = copy.deepcopy(pcd_query_aligned)
        w_pcd_query_aligned_icp.transform(reg_icp.transformation)
        w_pcd_query_aligned_icp.paint_uniform_color([0, 1, 0]) 

        if vis:
            pcd_query_aligned.paint_uniform_color([0, 0, 1])
            pcd_query.paint_uniform_color([1, 0, 0])  # Red

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10.0,    # length of axes
                origin=[0, 0, 0])
        
            # Visualize both point clouds
            o3d.visualization.draw_geometries(
                [axis, w_pcd_query_aligned_icp, pcd_query],
                window_name="Before (red) and After (green) Registration",
                point_show_normal=False
            )

        T_wa_final = reg_icp.transformation @ T_wa
        self.T_world_query = T_wa_final

        # save the transformation matrix as a json file
        T_wa_final_dict = {
            "T_wq": T_wa_final.tolist()
        }
        with open(self.visual_registration_output_path / "T_wq.json", "w") as f:
            json.dump(T_wa_final_dict, f, indent=4)

        print(f"[REGISTRATION] Transformation matrix saved to {self.visual_registration_output_path / 'T_wq.json'}")
        print(f"[REGISTRATION] Transformation matrix: {T_wa_final}")

        return T_wa_final


    def get_sfm_model(self) -> pycolmap.Reconstruction:
        """
        Get the SFM model from the visual registration output.
        Returns:
            pycolmap.Reconstruction: The SFM model.
        """
        if not self.sfm_dir.exists():
            raise FileNotFoundError(f"SFM directory not found: {self.sfm_dir}, re-run visual registration.")

        model = pycolmap.Reconstruction(self.sfm_dir)

        if model.num_images == 0:
            raise ValueError(f"No images found in SFM directory: {self.sfm_dir}")

        return model


    def get_poses_query(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get the poses of the query images.
        Returns:
            dict: {
                "w_T_wc": world to camera transformation matrix (4x4)
                "K": camera intrinsics matrix (3x3)
                "h": image height
                "w": image width
            }
        """

        if not Path(str(self.results_dir) + "_logs.pkl").exists():
            raise FileNotFoundError(f"Results log file not found: {self.results_dir}, re-run visual registration.")

        with open(str(self.results_dir) + "_logs.pkl", "rb") as f:
            logs = pickle.load(f)

        K = self.loader_query.calibration["K"]
        h = self.loader_query.calibration["h"]
        w = self.loader_query.calibration["w"]

        poses = {}

        for name, log in logs['loc'].items():
                a = 2
                R_cw = log['PnP_ret']['cam_from_world'].rotation.matrix()
                c_t_cw = log['PnP_ret']['cam_from_world'].translation

                # world to cam in world system
                R_wc = R_cw.T
                w_t_wc = -R_wc @ c_t_cw

                # to Transformation matrix 4x4
                w_T_wc = np.eye(4)
                w_T_wc[:3, :3] = R_wc
                w_T_wc[:3, 3] = w_t_wc

                poses[Path(name).stem] = {
                    "w_T_wc": w_T_wc,
                    "K": K,
                    "h": h,
                    "w": w,
                }

        return poses
    
    def get_poses_map(self) -> Dict[str, Dict[str, np.ndarray]]:
        # TODO - implement this function
        """
        Get the poses of the map images.
        Returns:
            dict: {
                "w_T_wc": world to camera transformation matrix (4x4)
                "K": camera intrinsics matrix (3x3)
                "h": image height
                "w": image width
            }
        """
        raise NotImplementedError("get_poses_map is not implemented yet.")
                

    def get_poses_gt(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get the gt poses of the rendered map images.
        Returns:
            dict: {
                "w_T_wc": world to camera transformation matrix (4x4)
                "K": camera intrinsics matrix (3x3)
                "h": image height
                "w": image width
            }
        """

        poses = {}
        json_files = sorted(self.pose_path_map.glob("*.json"))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            fx, fy, cx, cy = data['fx'], data['fy'], data['cx'], data['cy']
            w = data['width']
            h = data['height']

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ])

            T = np.array(data['extrinsic'])

            poses[json_file.stem] = {
                "w_T_wc": T,
                "K": K,
                "w": w,
                "h": h
            }

        return poses
    
    def mean_transformation(self, T_list: list) -> np.ndarray:

        """
        Compute the mean transformation matrix from a list of transformation matrices, filtering out outliers.
        Args:
            T_list (list): List of transformation matrices (4x4).
        Returns:
            np.ndarray: Mean transformation matrix (4x4).
        """

        N = len(T_list)
        if N == 0:
            raise ValueError("Empty pose list.")
           
        translations = np.array([T[:3, 3] for T in T_list])
        rotations = R.from_matrix([T[:3, :3] for T in T_list]) 

        t_norm = np.linalg.norm(translations, axis=1)
        t_norm_median = np.median(t_norm)

        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                r1 = rotations[i].as_matrix()
                r2 = rotations[j].as_matrix()

                frob_dist = np.sqrt(np.trace((r1 - r2).T @ (r1 - r2)))
                dist[i, j] = frob_dist
                
        # compute average distance to others
        avg_dist = np.mean(dist, axis=1)
        median_dist = np.median(avg_dist)

        # filter outlier if distance is greater than 1.5 * median distance or less than 0.5 * median distance
        outlier_mask_rot = np.logical_and(avg_dist < 1.25 * median_dist, avg_dist > 0.75 * median_dist)
        outlier_mask_trans = np.logical_and(t_norm < 1.25 * t_norm_median, t_norm > 0.75 * t_norm_median)
        outlier_mask = np.logical_and(outlier_mask_rot, outlier_mask_trans)

        # compute mean transformation
        T_mean = np.eye(4)
        T_mean[:3, :3] = np.mean(rotations[outlier_mask].as_matrix(), axis=0)
        T_mean[:3, 3] = np.mean(translations[outlier_mask], axis=0)

        return T_mean

    
    def create_reconstruction_from_gt_poses(self):

        poses_gt = self.get_poses_gt()

        rec = pycolmap.Reconstruction()

        for name, pose in poses_gt.items():

            f_x = pose["K"][0, 0]
            f_y = pose["K"][1, 1]
            c_x = pose["K"][0, 2]
            c_y = pose["K"][1, 2]
            w = pose["w"]
            h = pose["h"]

            tvec = pose["w_T_wc"][:3, 3]
            R = pose["w_T_wc"][:3, :3]
            qvec = Rotation.from_matrix(R).as_quat()

            cam_from_world = pycolmap.Rigid3d(pose["w_T_wc"][:3, :4])
            image_name = f"mapping/{name}.png"

            # Create a camera object
            camera = pycolmap.Camera(
                model="PINHOLE",
                width=w,
                height=h,
                params=np.array([f_x, f_y, c_x, c_y], dtype=np.float32),
                camera_id=int(name)
            )

            # Add the camera to the reconstruction
            rec.add_camera(camera)

            # Create an image object
            image = pycolmap.Image(
                name=image_name,
                camera_id=camera.camera_id,
                image_id=camera.camera_id,
                cam_from_world=cam_from_world
            )

            rec.add_image(image)

        # Save the reconstruction
        self.initial_sfm_dir.mkdir(parents=True, exist_ok=True)
        rec.write(self.initial_sfm_dir)
        print(f"[REGISTRATION] Reconstruction created from gt poses. Saved to {self.initial_sfm_dir}")

    def visual_registration_viz(self):
        """
        Visualise the map (points + map cameras) together with all
        localised query cameras inside an interactive Plotly view.
        """

        if not self.sfm_dir.exists():
            raise FileNotFoundError(f"SFM directory not found: {self.sfm_dir}, re-run visual registration.")

        model = pycolmap.Reconstruction(self.sfm_dir)

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model,
            color="rgba(255,0,0,0.5)",
            name="mapping",
            points_rgb=True)

        K_pinhole = self.loader_query.calibration["K"]

        
        query_poses = self.get_poses_query()
        for name, pose in query_poses.items():
            w_T_wc = pose["w_T_wc"]
            R_wc = w_T_wc[:3, :3]
            C_wc = w_T_wc[:3, 3]

            viz_3d.plot_camera(fig,
                R_wc,
                C_wc,  
                K=K_pinhole,
                name=name,
                text=name,
                color="rgba(0,255,0,0.5)")

        fig.update_layout(height=800)
        fig.show()
        a = 2


    def viz_2d(self):

        from hloc.visualization import visualize_loc
        model = pycolmap.Reconstruction(self.sfm_dir)
        # visualization.visualize_sfm_2d(model,self.images, color_by="visibility", n=2)
        from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

        camera = pycolmap.infer_camera_from_image(self.images / self.query[0])
        camera.model = "PINHOLE"
        camera.width = self.loader_query.calibration["w"]
        camera.height = self.loader_query.calibration["h"]
        camera.params[0] = self.loader_query.calibration["K"][0, 0]
        camera.params[1] = self.loader_query.calibration["K"][1, 1]
        camera.params[2] = self.loader_query.calibration["K"][0, 2]
        camera.params[3] = self.loader_query.calibration["K"][1, 2]

        good_refs = {j for _,j in map(str.split, open(self.loc_pairs))}
        self.references = good_refs

        ref_ids = [model.find_image_with_name(r).image_id for r in self.references]
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(model, conf)
        ret, log = pose_from_cluster(localizer, self.query[0], camera, ref_ids, self.features, self.matches)

        mask = log["PnP_ret"]["inlier_mask"]
        log["PnP_ret"]["inliers"] = np.where(mask)[0]

        # print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
        visualization.visualize_loc_from_log(self.images, self.query[0], log, model)
        a = 2
 
        fig = viz_3d.init_figure()
        pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
        viz_3d.plot_camera_colmap(
            fig, pose, camera, color="rgba(0,255,0,0.5)", name=self.query[0], fill=True
        )
        # visualize 2D-3D correspodences
        inl_3d = np.array(
            [model.points3D[pid].xyz for pid in np.array(log["points3D_ids"])[ret["inliers"]]]
        )
        viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=self.query[0])
        fig.show()

        # visualize_loc(results=self.results_dir,image_dir=self.images,reconstruction=self.sfm_dir,db_image_dir=None)

if __name__ == "__main__":
    # Example usage
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    leica_data = LeicaData(base_path)

    rec_name = "door_6"
    sensor_module_name = "aria_human_ego"
    aria_data = AriaData(base_path=base_path, rec_name=rec_name, sensor_module_name=sensor_module_name)

    # sensor_module_name = "iphone_left"
    # iphone_data = IPhoneData(base_path=base_path, rec_name=rec_name, sensor_module_name=sensor_module_name)

    spatial_registrator = SpatialRegistrator(loader_map=leica_data, loader_query=aria_data)
    spatial_registrator.visual_registration(from_gt=True)
    spatial_registrator.pcd_to_pcd_registration()

    # spatial_registrator.visual_registration_viz()
    # spatial_registrator.viz_2d()
    # spatial_registrator.get_poses_query()


