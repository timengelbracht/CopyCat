from pathlib import Path
import os
from typing import List, Dict
from spatial_registrator import SpatialRegistrator
from data_indexer import RecordingIndex
from data_loader_aria import AriaData
from data_loader_iphone import IPhoneData
from data_loader_leica import LeicaData
from data_loader_umi import UmiData
from data_loader_gripper import GripperData

import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors




class SpatialVisualizer:
    """
    Class to handle spatial visualisations of sensor modules to Leica scans.

    """

    def __init__(self, loader_map: object, loaders_for_location: Dict[str, Dict[str, object]]):

        self.loader_map = loader_map
        self.loaders_for_location = loaders_for_location

        # self.loaders_gripper = {key: value for key, value in loaders_for_location.items() if "gripper" in key}
        # self.loaders_wrist = {key: value for key, value in loaders_for_location.items() if "wrist" in key}
        # self.loaders_hand = {key: value for key, value in loaders_for_location.items() if "hand" in key}
        # self.loaders_umi = {key: value for key, value in loaders_for_location.items() if "umi" in key}

    def visualize_single_aria_trajectory(self, 
                                         rec_type: str,
                                         rec_module: str,
                                         interaction_indices: str,
                                         stride: int = 200, 
                                         traj: List[str] = ["aria", "palm", "hand"], 
                                         mode: str = "point", 
                                         pcd_sampling: str = "downsampled"):
        """
        Visualize the trajectory of the AriaData query module aligned to the map.
        Args:
            stride (int): Step size for downsampling the trajectory points.
            mode (str): Visualization mode, can be "point", "device_frame", or "camera_frame".
            pcd_sampling (str): Point cloud sampling method, can be "downsampled" or "raw".
        """

        loader_rec_type = self.loaders_for_location[rec_type]
        loader_aria = loader_rec_type[rec_module]
        
        if mode not in ["point", "device_frame", "camera_frame"]:
            raise ValueError("mode must be one of ['point', 'device_frame', 'camera_frame']")
        
        if pcd_sampling not in ["downsampled", "raw"]:
            raise ValueError("pcd_sampling must be one of ['downsampled', 'raw']")
        
        if pcd_sampling == "downsampled":
            pcd_map = self.loader_map.get_downsampled_points()
        elif pcd_sampling == "raw":
            pcd_map = self.loader_map.get_full_points()
        
        markers_palm = []
        markers = []

        if "aria" in traj:
            trajectory_query = loader_aria.get_closed_loop_trajectory_aligned()
            trajectory_query = trajectory_query[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device", "qw_world_device", "qx_world_device", "qy_world_device", "qz_world_device"]]
            trajectory_query.columns = ["timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
            trajectory_query = trajectory_query.iloc[::stride, :].reset_index(drop=True)

            cmap = cm.get_cmap("inferno")  # 'viridis', 'plasma', 'cool', 'inferno'
            norm = mcolors.Normalize(vmin=0, vmax=len(trajectory_query)-1)

            T_dc = loader_aria.calibration["PINHOLE"]["T_device_camera"]
            T_cRaw_cRect = loader_aria.calibration["PINHOLE"]["pinhole_T_device_camera"]
            for i in range(len(trajectory_query)):
                qw = trajectory_query["qw"].iloc[i]
                qx = trajectory_query["qx"].iloc[i]
                qy = trajectory_query["qy"].iloc[i]
                qz = trajectory_query["qz"].iloc[i]
                tx = trajectory_query["tx"].iloc[i]
                ty = trajectory_query["ty"].iloc[i]
                tz = trajectory_query["tz"].iloc[i]

                T_wd = np.eye(4)
                T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_wd[:3, 3] = np.array([tx, ty, tz])

                color = cmap(norm(i))[:3]

                if mode == "point":
                    # show point in world frame
                    T = T_wd
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    marker.paint_uniform_color(color)  # red color
                    marker.transform(T)
                    markers.append(copy.deepcopy(marker))
                elif mode == "device_frame":
                    # show device frame in world frame
                    T = T_wd
                    marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    marker.transform(T)
                    markers.append(copy.deepcopy(marker))
                if mode == "camera_frame":
                    # show camera frame in world frame
                    # T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_wd)
                    T_dcRect = T_dc @ T_cRaw_cRect
                    T = T_wd @ T_dcRect
                    marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    marker.transform(T)
                    markers.append(copy.deepcopy(marker))

        if "palm" in traj:
            palms = loader_aria.get_palm_and_wrist_tracking_aligned()
            palms_right = palms[["timestamp", "tx_right_palm_world", "ty_right_palm_world", "tz_right_palm_world", "nx_right_palm_world", "ny_right_palm_world", "nz_right_palm_world"]]
            palms_right.columns = ["timestamp", "tx", "ty", "tz", "nx", "ny", "nz"]
            palms_right = palms_right.iloc[::2, :].reset_index(drop=True)
            
            for i in range(len(palms_right)):
                nx = palms_right["nx"].iloc[i]
                ny = palms_right["ny"].iloc[i]
                nz = palms_right["nz"].iloc[i]
                tx = palms_right["tx"].iloc[i]
                ty = palms_right["ty"].iloc[i]
                tz = palms_right["tz"].iloc[i]

                T_wd = np.eye(4)
                # T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_wd[:3, 3] = np.array([tx, ty, tz])

                # show point in world frame
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color([1,0,0])  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

        if "wrist" in traj: 
            wrists = loader_aria.get_palm_and_wrist_tracking_aligned()
            wrists_right = palms[["timestamp", "tx_right_wrist_world", "ty_right_wrist_world", "tz_right_wrist_world", "nx_right_wrist_world", "ny_right_wrist_world", "nz_right_wrist_world"]]
            wrists_right.columns = ["timestamp", "tx", "ty", "tz", "nx", "ny", "nz"]
            wrists_right = wrists_right.iloc[::5, :].reset_index(drop=True)
            
            for i in range(len(wrists_right)):
                nx = wrists_right["nx"].iloc[i]
                ny = wrists_right["ny"].iloc[i]
                nz = wrists_right["nz"].iloc[i]
                tx = wrists_right["tx"].iloc[i]
                ty = wrists_right["ty"].iloc[i]
                tz = wrists_right["tz"].iloc[i]

                T_wd = np.eye(4)
                # T_wd[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T_wd[:3, 3] = np.array([tx, ty, tz])

                # show point in world frame
                T = T_wd
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color([0,0.5,0.5])  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

        o3d.visualization.draw_geometries(
            [pcd_map] + markers + markers_palm,
            point_show_normal=False
        )

    def visualize_single_gripper_trajectory_static(self, 
                                            rec_type: str,
                                            rec_module: str,
                                            interaction_indices: str,
                                            stride: int = 200, 
                                            mode: str = "point", 
                                            pcd_sampling: str = "downsampled"):
        
        
        if pcd_sampling not in ["downsampled", "raw"]:
            raise ValueError("pcd_sampling must be one of ['downsampled', 'raw']")
        
        if mode not in ["point", "device_frame", "camera_frame", "tool_point"]:
            raise ValueError("mode must be one of ['point', 'device_frame', 'camera_frame', 'tool_frame']")
        
        if pcd_sampling == "downsampled":
            pcd_map = self.loader_map.get_downsampled_points()
        elif pcd_sampling == "raw":
            pcd_map = self.loader_map.get_full_points()

        # loader_gripper = self.loaders_gripper[rec_type][f"_{interaction_indices}"]
        loader_gripper = self.loaders_for_location[rec_type][f"{rec_module}_{interaction_indices}"]


        # trajectory of aria on gripper in world frame
        trajectory_query = loader_gripper.loader_aria_gripper.get_closed_loop_trajectory_aligned()
        trajectory_query = trajectory_query[["timestamp", "tx_world_device", "ty_world_device", "tz_world_device", "qw_world_device", "qx_world_device", "qy_world_device", "qz_world_device"]]
        trajectory_query.columns = ["timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
        trajectory_query = trajectory_query.iloc[::stride, :].reset_index(drop=True)

        # forces and torques in sensor frame
        df_ft = loader_gripper.get_force_torque_measurements()

        # get all necessary transforms
        T_imu_tool = loader_gripper.calibration["imu0"]["T_imu_tool"]
        T_imu_sensor = loader_gripper.calibration["imu0"]["T_imu_sensor"]
        T_device_camera = loader_gripper.loader_aria_gripper.calibration["PINHOLE"]["T_device_camera"]
        T_cameraRaw_cameraRect = loader_gripper.loader_aria_gripper.calibration["PINHOLE"]["pinhole_T_device_camera"]
        T_camera_imu = loader_gripper.calibration["cam2"]["T_cam_imu"]
        T_tool_sensor = np.linalg.inv(T_imu_tool) @ T_imu_sensor
        T_sensor_tool = np.linalg.inv(T_tool_sensor)

        # get force torque origin in sensor frame
        T_device_sensor = T_device_camera @ T_camera_imu @ T_imu_sensor
        T_device_tool = T_device_camera @ T_camera_imu @ T_imu_tool

        cmap = cm.get_cmap("inferno")  # 'viridis', 'plasma', 'cool', 'inferno'
        norm = mcolors.Normalize(vmin=0, vmax=len(trajectory_query)-1)

        markers = []     
        markers_force = []    
        markers_torque = []   
        for i in range(len(trajectory_query)):
            qw = trajectory_query["qw"].iloc[i]
            qx = trajectory_query["qx"].iloc[i]
            qy = trajectory_query["qy"].iloc[i]
            qz = trajectory_query["qz"].iloc[i]
            tx = trajectory_query["tx"].iloc[i]
            ty = trajectory_query["ty"].iloc[i]
            tz = trajectory_query["tz"].iloc[i]
            timestamp = trajectory_query["timestamp"].iloc[i]

            # get closest force torque measurement in sensor frame
            idx_closest = (np.abs(df_ft["timestamp"] - timestamp)).argmin()
            force_x = df_ft["wrench_ext.force.x"].iloc[idx_closest]
            force_y = df_ft["wrench_ext.force.y"].iloc[idx_closest]
            force_z = df_ft["wrench_ext.force.z"].iloc[idx_closest]
            torque_x = df_ft["wrench_ext.torque.x_filt"].iloc[idx_closest]
            torque_y = df_ft["wrench_ext.torque.y_filt"].iloc[idx_closest]
            torque_z = df_ft["wrench_ext.torque.z_filt"].iloc[idx_closest]

            T_world_device = np.eye(4)
            T_world_device[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T_world_device[:3, 3] = np.array([tx, ty, tz])

            color = cmap(norm(i))[:3]
            
            if mode == "point":
                # show point in world frame
                T = T_world_device
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color(color)  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))
            elif mode == "device_frame":
                # show device frame in world frame
                T = T_world_device
                marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                marker.transform(T)
                markers.append(copy.deepcopy(marker))
            if mode == "camera_frame":
                # show camera frame in world frame
                # T_ca = np.linalg.inv(T_dc @ T_cRaw_cRect) @ np.linalg.inv(T_wd)
                T_dcRect = T_device_camera @ T_cameraRaw_cameraRect
                T = T_world_device @ T_dcRect
                marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                marker.transform(T)
                markers.append(copy.deepcopy(marker))
            if mode == "tool_point":
                # show tool frame in world frame
                T = T_world_device @ T_device_tool
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color(color)  # red color
                marker.transform(T)
                markers.append(copy.deepcopy(marker))

            T_world_tool = T_world_device @ T_device_tool

            force_vector_sensor = np.array([force_x, force_y, force_z, 0.0], dtype=np.float64).reshape(4, 1)
            torque_vector_sensor = np.array([torque_x, torque_y, torque_z], dtype=np.float64).reshape(3, 1)

            # move torque to tool center
            torque_vector_tool = T_tool_sensor[:3,:3] @ torque_vector_sensor + np.atleast_2d(np.cross(T_tool_sensor[:3,3], T_tool_sensor[:3,:3] @ force_vector_sensor[:3,0])).T

            torque_vector_world = T_world_tool[:3,:3] @ torque_vector_tool
            markers_torque.extend(torque_ring(T_world_tool, torque_vector_world.flatten(), thr=0.2, ring_radius=0.03,
                                            tube_r0=0.004, color=[0.84, 0.33, 0.00], arrowhead=True))

            ex_w = (T_world_tool @ np.array([1, 0, 0, 0], dtype=np.float64).reshape(4, 1))[:3, 0]
            ey_w = (T_world_tool @ np.array([0, 1, 0, 0], dtype=np.float64).reshape(4, 1))[:3, 0]
            ez_w = (T_world_tool @ np.array([0, 0, 1, 0], dtype=np.float64).reshape(4, 1))[:3, 0]

            force_vector_tool = (T_tool_sensor @ force_vector_sensor)[:3, 0]

            force_vector_x_world = ex_w * force_vector_tool[0]
            force_vector_y_world = ey_w * force_vector_tool[1]
            force_vector_z_world = ez_w * force_vector_tool[2]
            f_tang_world  = ex_w * force_vector_tool[0] + ey_w * force_vector_tool[1]

            markers_force.extend(force_arrow(T_world_tool, force_vector_z_world, thr=1.0, color=[0.0,0.0,1.0]))
            markers_force.extend(force_arrow(T_world_tool, f_tang_world, thr=1.0, color=[0.0,1.0,0.0]))
            # markers_force.extend(force_arrow(T_world_tool, force_vector_y_world, thr=1.0, color=[0.0,1.0,0.0]))
            # markers_force.extend(force_arrow(T_world_tool, force_vector_z_world, thr=1.0, color=[0.0,0.0,1.0]))

        o3d.visualization.draw_geometries(
            [pcd_map] + markers + markers_force + markers_torque,
            point_show_normal=False
        )

        a = 2

def R_from_a_to_b(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a /= max(np.linalg.norm(a), 1e-12); b /= max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-12:                            # parallel / antiparallel
        if c > 0: return np.eye(3)
        u = np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0])
        u -= a * np.dot(a, u); u /= np.linalg.norm(u)
        return -np.eye(3) + 2*np.outer(u, u) # 180°
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K@K * ((1 - c) / (s*s))


def force_arrow(Tw, f_world, thr=1.0, scale=0.05, Lmin=0.03, Lmax=0.25, r0=0.004, color=None):
    mag = float(np.linalg.norm(f_world))
    if mag <= thr: return []
    dirw = f_world / mag
    L = float(np.clip(mag * scale, Lmin, Lmax))
    r = r0 * (0.8 + 0.6 * np.tanh(mag / 10.0))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=r, cone_radius=r*2.0, cylinder_height=L*0.8, cone_height=L*0.2, resolution=24
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color([0,1,0] if color is None else color)
    R = R_from_a_to_b([0,0,1], dirw)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = Tw[:3,3]
    arrow.transform(T)
    return [arrow]

def torque_ring(Tw, tau_world, thr=0.1, ring_radius=0.05,
                tube_r0=0.003, color=(1.0, 0.7, 0.2), arrowhead=False):
    """
    Tw: 4x4 pose of where to place the torque glyph (sensor or tool origin)
    tau_world: (3,) torque vector expressed in world
    """
    mag = float(np.linalg.norm(tau_world))
    if mag <= thr: 
        return []

    axis = tau_world / mag
    mag_ref = 3.0  # N at which you want near-max thickness
    r_min, r_max = tube_r0*1.4, min(0.44*ring_radius, tube_r0*8.0)
    alpha = np.clip(mag / mag_ref, 0.0, 1.0)
    tube_r = float(r_min + (r_max - r_min) * alpha)

    ring = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=ring_radius,
        tube_radius=tube_r,
        radial_resolution=90,
        tubular_resolution=18
    )
    ring.compute_vertex_normals()
    ring.paint_uniform_color(color)

    # orient +Z -> axis, place at Tw
    R = R_from_a_to_b([0,0,1], axis)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = Tw[:3,3]
    ring.transform(T)

    geoms = [ring]

    if arrowhead:
        # tiny cone tangent to the ring to hint rotation direction (right-hand rule)
        cone = o3d.geometry.TriangleMesh.create_cone(radius=tube_r*2.8, height=ring_radius*0.5, resolution=24)
        cone.compute_vertex_normals()
        cone.paint_uniform_color(color)
        # tangent in the ring's local frame at angle 0 is +Y
        tang_dir_world = (R @ np.array([0,1,0])).astype(float)
        R_tang = R_from_a_to_b([0,0,1], tang_dir_world)
        Tcone = np.eye(4)
        Tcone[:3,:3] = R_tang
        # place the cone on the ring at angle 0: point at [R,0,0] in ring-local
        pos_world = Tw[:3,3] + (R @ np.array([ring_radius,0,0]))
        Tcone[:3,3] = pos_world
        cone.transform(Tcone)
        geoms.append(cone)

    return geoms
if __name__ == "__main__":
    # Example usage
    base_path = Path(f"/data/ikea_recordings")
    rec_location = "office_1"

    leica_data = LeicaData(base_path,rec_loc=rec_location, initial_setup="001")

    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )


    loaders_for_location = {}
    interactions = ["gripper", "wrist", "hand", "umi"]
    for interaction in interactions:
        queries_at_loc = data_indexer.query(
            location=rec_location, 
            interaction=interaction, 
            recorder=None,
            interaction_index="8-14"
        )
        loaders_for_interaction_type = {}
        for loc, inter, rec, ii, path in queries_at_loc:
            rec_type = inter
            rec_module = rec
            interaction_indices = ii
            if rec == "gripper":
                GRIPPER_DATA = GripperData(base_path, 
                               rec_loc=rec_location, 
                               rec_type=rec_type, 
                               rec_module=rec_module, 
                               interaction_indices=interaction_indices,
                               data_indexer=data_indexer)
                loaders_for_interaction_type[f"{rec_module}_{ii}"] = GRIPPER_DATA
            elif "aria" in rec:
                ARIA_DATA = AriaData(base_path, 
                                    rec_loc=rec_location, 
                                    rec_type=rec_type, 
                                    rec_module=rec_module, 
                                    interaction_indices=interaction_indices,
                                    data_indexer=data_indexer)
                loaders_for_interaction_type[f"{rec_module}_{ii}"] = ARIA_DATA
            elif "iphone" in rec:
                IPHONE_DATA = IPhoneData(base_path, 
                                        rec_loc=rec_location, 
                                        rec_type=rec_type, 
                                        rec_module=rec_module, 
                                        interaction_indices=interaction_indices,
                                        data_indexer=data_indexer)
                loaders_for_interaction_type[f"{rec_module}_{ii}"] = IPHONE_DATA
            elif "umi" in rec:
                UMI_DATA = UmiData(base_path, 
                                   rec_loc=rec_location, 
                                   rec_type=rec_type, 
                                   rec_module=rec_module, 
                                   interaction_indices=interaction_indices,
                                   data_indexer=data_indexer)
                loaders_for_interaction_type[f"{rec_module}_{ii}"] = UMI_DATA
        loaders_for_location[interaction] = loaders_for_interaction_type


    spatial_visualizer = SpatialVisualizer(loader_map=leica_data, loaders_for_location=loaders_for_location)
    spatial_visualizer.visualize_single_gripper_trajectory_static(rec_type="gripper", 
                                                           rec_module="gripper", 
                                                           interaction_indices="8-14",
                                                           stride=200, 
                                                           mode="tool_point", 
                                                           pcd_sampling="raw")