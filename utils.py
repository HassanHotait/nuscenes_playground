import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

def filter_depth_by_radar_model(nusc, depth_data, sample_data, radar_channel="RADAR_FRONT"):
    """
    Filters a depth image to only include pixels whose 3D projections fall within
    the radar's true FOV model, based on ARS 408-21 spec.

    Parameters:
    - nusc: NuScenes instance.
    - depth_data: (H, W) numpy array of depth values.
    - sample_data: dict, current camera sample_data.
    - radar_channel: Radar sensor channel name.

    Returns:
    - A depth image where out-of-FOV pixels are set to 0.
    """

    # Calibration transforms
    cs_cam = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    K = np.array(cs_cam["camera_intrinsic"])
    T_cam_to_ego = transform_matrix(cs_cam["translation"], Quaternion(cs_cam["rotation"]), inverse=False)

    sample = nusc.get("sample", sample_data["sample_token"])
    radar_token = sample["data"][radar_channel]
    radar_sd = nusc.get("sample_data", radar_token)
    cs_radar = nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])
    T_radar_to_ego = transform_matrix(cs_radar["translation"], Quaternion(cs_radar["rotation"]), inverse=False)

    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    T_ego_to_world = transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=False)

    T_cam_to_world = T_ego_to_world @ T_cam_to_ego
    T_world_to_radar = np.linalg.inv(T_ego_to_world @ T_radar_to_ego)
    T_cam_to_radar = T_world_to_radar @ T_cam_to_world

    # Project all valid depth pixels to radar frame
    H, W = depth_data.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T
    z = depth_data.flatten()
    valid = z > 0
    uv1 = uv1[:, valid]
    z = z[valid]

    xyz_cam = np.linalg.inv(K) @ (uv1 * z)
    xyz_cam_hom = np.vstack([xyz_cam, np.ones((1, xyz_cam.shape[1]))])
    xyz_radar = T_cam_to_radar @ xyz_cam_hom
    x, y, z = xyz_radar[:3, :]

    # Compute range, azimuth, elevation
    r = np.linalg.norm(xyz_radar[:3, :], axis=0)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))

    # Build full radar FOV mask from specs
    fov_mask = np.zeros_like(r, dtype=bool)

    # -------------------------
    # Far field: r > 20 m, ±15.8°, ±9°
    ff_mask = (r > 20) & (r <= 250) & (np.abs(azimuth) <= 15.8) & (np.abs(elevation) <= 9)

    # Near field mid: r <= 70 m @ ±45°
    nf_mid_mask = (r <= 70) & (np.abs(azimuth) <= 45) & (np.abs(elevation) <= 60)

    # Near field wide: r <= 20 m @ ±60°
    nf_wide_mask = (r <= 20) & (np.abs(azimuth) <= 60) & (np.abs(elevation) <= 60)

    fov_mask = ff_mask | nf_mid_mask | nf_wide_mask

    # Apply mask to depth image
    depth_filtered = depth_data.flatten()
    keep_mask = np.zeros_like(depth_filtered, dtype=bool)
    keep_mask[valid] = fov_mask
    depth_filtered[~keep_mask] = 0.0

    return depth_filtered.reshape(H, W)


import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.utils import category_to_detection_name
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def filter_depth_by_gt_boxes_camera_frame(nusc, depth_data, sample_data):
    """
    Filters a dense depth map to keep only points that lie inside 3D GT boxes,
    working entirely in the CAMERA coordinate frame.

    Parameters:
    - nusc: NuScenes instance
    - depth_data: (H, W) depth map from a camera (must be in camera frame)
    - sample_data: camera sample_data dict

    Returns:
    - Filtered depth map (H, W), zeroed outside of 3D GT boxes
    """

    # Camera intrinsics and calibration
    cs_cam = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    K = np.array(cs_cam["camera_intrinsic"])
    T_ego_to_cam = transform_matrix(cs_cam["translation"], Quaternion(cs_cam["rotation"]), inverse=True)

    # Ego pose
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    T_world_to_ego = transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]), inverse=True)

    # Full world → camera transform
    T_world_to_cam = T_ego_to_cam @ T_world_to_ego

    # Project depth map to 3D points in camera frame
    H, W = depth_data.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # (3, N)
    z = depth_data.flatten()
    valid = z > 0
    uv1 = uv1[:, valid]
    z = z[valid]

    xyz_cam = np.linalg.inv(K) @ (uv1 * z)  # (3, N)
    points_xyz = xyz_cam.T  # (N, 3)

    # Get GT boxes and transform them to camera frame
    sample = nusc.get("sample", sample_data["sample_token"])
    gt_boxes = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if category_to_detection_name(ann["category_name"]) is None:
            continue  # skip non-detection classes

        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
        box.transform(T_world_to_cam)  # world → camera frame
        gt_boxes.append(box)

    if len(gt_boxes) == 0:
        return np.zeros_like(depth_data)

    # Convert boxes to [x, y, z, dx, dy, dz, yaw]
    gt_boxes_np = np.array([
        b.center.tolist() + b.wlh.tolist() + [b.orientation.yaw_pitch_roll[0]]
        for b in gt_boxes
    ])  # (M, 7)

    # Use GPU to filter points inside boxes
    point_tensor = torch.from_numpy(points_xyz).unsqueeze(0).float().cuda()  # (1, N, 3)
    box_tensor = torch.from_numpy(gt_boxes_np[:, 0:7]).unsqueeze(0).float().cuda()  # (1, M, 7)

    box_idxs = roiaware_pool3d_utils.points_in_boxes_gpu(point_tensor, box_tensor).squeeze(0).cpu().numpy()
    mask = box_idxs >= 0  # True if point is inside any box

    # Zero out invalid depth pixels
    depth_flat = depth_data.flatten()
    keep_mask = np.zeros_like(depth_flat, dtype=bool)
    keep_mask[valid] = mask
    depth_flat[~keep_mask] = 0.0

    return depth_flat.reshape(H, W)
