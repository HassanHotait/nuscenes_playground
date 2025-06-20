#!/usr/bin/env python3
from __future__ import annotations

import argparse
from os import write
import os
import pathlib
import gzip
from typing import Any, Final

import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from nuscenes import nuscenes

# from .download_dataset import MINISPLIT_SCENES, download_minisplit
from export_gps import derive_latlon

import cv2

DESCRIPTION = """
# nuScenes

Visualize the [nuScenes dataset](https://www.nuscenes.org/) including lidar, radar, images, and bounding boxes data.

The full source code for this example is available
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/nuscenes_dataset).
"""

EXAMPLE_DIR: Final = pathlib.Path(__file__).parent.parent
DATASET_DIR: Final = pathlib.Path("/work3/s203211/datasets/nuscenes")
# currently need to calculate the color manually
# see https://github.com/rerun-io/rerun/issues/4409
cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(
    vmin=3.0,
    vmax=75.0,
)


def ensure_scene_available(root_dir: pathlib.Path, dataset_version: str, scene_name: str) -> None:
    """
    Ensure that the specified scene is available.

    Downloads minisplit into root_dir if scene_name is part of it and root_dir is empty.

    Raises ValueError if scene is not available and cannot be downloaded.
    """
    try:
        nusc = nuscenes.NuScenes(version=dataset_version, dataroot=root_dir, verbose=True)
    except AssertionError:  # dataset initialization failed
        if dataset_version == "v1.0-trainval":
            # download_minisplit(root_dir)
            nusc = nuscenes.NuScenes(version=dataset_version, dataroot=root_dir, verbose=True)
        else:
            print(f"Could not find dataset at {root_dir} and could not automatically download specified scene.")
            exit()

    scene_names = [s["name"] for s in nusc.scene]
    if scene_name not in scene_names:
        raise ValueError(f"{scene_name=} not found in dataset")


def nuscene_sensor_names(nusc: nuscenes.NuScenes, scene_name: str) -> list[str]:
    """Return all sensor names in the scene."""

    sensor_names = set()

    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    first_sample = nusc.get("sample", scene["first_sample_token"])
    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        if sample_data["sensor_modality"] == "camera":
            current_camera_token = sample_data_token
            while current_camera_token != "":
                sample_data = nusc.get("sample_data", current_camera_token)
                sensor_name = sample_data["channel"]
                sensor_names.add(sensor_name)
                current_camera_token = sample_data["next"]

    # For a known set of cameras, order the sensors in a circle.
    ordering = {
        "CAM_FRONT_LEFT": 0,
        "CAM_FRONT": 1,
        "CAM_FRONT_RIGHT": 2,
        "CAM_BACK_RIGHT": 3,
        "CAM_BACK": 4,
        "CAM_BACK_LEFT": 5,
    }
    return sorted(sensor_names, key=lambda sensor_name: ordering.get(sensor_name, float("inf")))


def log_nuscenes(nusc: nuscenes.NuScenes, scene_name: str, max_time_sec: float) -> None:
    """Log nuScenes scene."""

    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    location = nusc.get("log", scene["log_token"])["location"]

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", scene["first_sample_token"])

    first_lidar_token = ""
    first_radar_tokens = []
    first_camera_tokens = []
    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        log_sensor_calibration(sample_data, nusc)

        if sample_data["sensor_modality"] == "lidar":
            first_lidar_token = sample_data_token
        elif sample_data["sensor_modality"] == "radar":
            first_radar_tokens.append(sample_data_token)
        elif sample_data["sensor_modality"] == "camera":
            first_camera_tokens.append(sample_data_token)

    first_timestamp_us = nusc.get("sample_data", first_lidar_token)["timestamp"]
    max_timestamp_us = first_timestamp_us + 1e6 * max_time_sec

    log_lidar_and_ego_pose(location, first_lidar_token, nusc, max_timestamp_us)
    log_cameras(first_camera_tokens, nusc, max_timestamp_us)
    log_radars(first_radar_tokens, nusc, max_timestamp_us)
    log_annotations(location, first_sample_token, nusc, max_timestamp_us)


def log_lidar_and_ego_pose(
    location: str,
    first_lidar_token: str,
    nusc: nuscenes.NuScenes,
    max_timestamp_us: float,
) -> None:
    """Log lidar data and vehicle pose."""
    current_lidar_token = first_lidar_token

    ego_trajectory_lat_lon = []

    while current_lidar_token != "":
        sample_data = nusc.get("sample_data", current_lidar_token)
        sensor_name = sample_data["channel"]

        if max_timestamp_us < sample_data["timestamp"]:
            break

        # timestamps are in microseconds
        rr.set_time("timestamp", timestamp=sample_data["timestamp"] * 1e-6)

        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        rotation_xyzw = np.roll(ego_pose["rotation"], shift=-1)  # go from wxyz to xyzw
        position_lat_lon = derive_latlon(location, ego_pose)
        ego_trajectory_lat_lon.append(position_lat_lon)

        rr.log(
            "world/ego_vehicle",
            rr.Transform3D(
                translation=ego_pose["translation"],
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
                axis_length=10.0,  # The length of the visualized axis.
                relation=rr.TransformRelation.ParentFromChild,
            ),
            rr.GeoPoints(lat_lon=position_lat_lon, radii=rr.Radius.ui_points(8.0), colors=0xFF0000FF),
        )
        # TODO(#6889): We don't want the radius for the trajectory line to be the same as the radius of the points.
        # However, rr.GeoPoints uses the same `rr.components.Radius` for this, so these two archetypes would influence each other
        # if logged on the same entity. In the future, they will have different tags, which will allow them to live side by side.
        rr.log(
            "world/ego_vehicle/trajectory",
            rr.GeoLineStrings(lat_lon=ego_trajectory_lat_lon, radii=rr.Radius.ui_points(1.0), colors=0xFF0000FF),
        )

        current_lidar_token = sample_data["next"]

        data_file_path = nusc.dataroot / sample_data["filename"]
        pointcloud = nuscenes.LidarPointCloud.from_file(str(data_file_path))
        points = pointcloud.points[:3].T  # shape after transposing: (num_points, 3)
        point_distances = np.linalg.norm(points, axis=1)
        point_colors = cmap(norm(point_distances))
        rr.log(f"world/ego_vehicle/{sensor_name}", rr.Points3D(points, colors=point_colors))


# def log_cameras(first_camera_tokens: list[str], nusc: nuscenes.NuScenes, max_timestamp_us: float) -> None:
#     """Log camera data."""
#     logged_frames = 0
#     for first_camera_token in first_camera_tokens:
#         current_camera_token = first_camera_token
#         while current_camera_token != "":
#             sample_data = nusc.get("sample_data", current_camera_token)
#             if max_timestamp_us < sample_data["timestamp"]:
#                 break
#             sensor_name = sample_data["channel"]
#             rr.set_time("timestamp", timestamp=sample_data["timestamp"] * 1e-6)
#             data_file_path = nusc.dataroot / sample_data["filename"]
#             rr.log(f"world/ego_vehicle/{sensor_name}", rr.EncodedImage(path=data_file_path))
#             logged_frames += 1
#             current_camera_token = sample_data["next"]
#     print(f"Logged {logged_frames} camera frames.")

def log_cameras(first_camera_tokens: list[str], nusc: nuscenes.NuScenes, max_timestamp_us: float) -> None:
    """Log camera images and corresponding depth maps from rendered .npy.gz files."""
    
    render_root = pathlib.Path("/work3/s203211/datasets/nuscenes/unnamed/neurad/2025-06-13_183130/render")

    for first_camera_token in first_camera_tokens:
        current_camera_token = first_camera_token

        while current_camera_token != "":
            sample_data = nusc.get("sample_data", current_camera_token)
            # Skip if not CAM_FRONT
            if sample_data["channel"] != "CAM_FRONT":
                current_camera_token = sample_data["next"]
                continue

            # Stop if past the max allowed time
            if sample_data["timestamp"] > max_timestamp_us:
                break

            sensor_name = sample_data["channel"]
            timestamp_sec = sample_data["timestamp"] * 1e-6
            rr.set_time("timestamp", timestamp=timestamp_sec)

            # Log the RGB camera image
            image_path = nusc.dataroot / sample_data["filename"]
            img = cv2.imread(str(image_path))
            img_shape = img.shape
            rr.log(f"world/ego_vehicle/{sensor_name}", rr.EncodedImage(path=image_path))

            # Build the depth file path from the camera image filename
            path_parts = sample_data["filename"].split("/")  # e.g. ['samples', 'CAM_FRONT', '...jpg']
            assert len(path_parts) >= 3, f"Unexpected filename format: {sample_data['filename']}"
            rel_subpath = pathlib.Path(path_parts[0]) / path_parts[1] / path_parts[2]
            rel_subpath = rel_subpath.with_suffix(".npy.gz")

            # Try "train" first
            depth_file_path = render_root / "train" / "raw-depth" / rel_subpath
            if not depth_file_path.exists():
                print(f"Depth file {depth_file_path} does not exist, trying 'val' directory.")
                # Try "val" if not found
                depth_file_path = render_root / "val" / "raw-depth" / rel_subpath
                assert depth_file_path.exists(), f"Depth file {depth_file_path} does not exist."

            # Load and log the depth map
            with gzip.open(depth_file_path, "rb") as f:
                depth_data = np.load(f)
                depth_data =cv2.resize(depth_data, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
                # TODO: Only Consider depth values within radar FOV
                depth_data = filter_depth_by_radar_fov(nusc, depth_data, sample_data)


            rr.log(f"world/ego_vehicle/{sensor_name}/depth", rr.DepthImage(depth_data))

            current_camera_token = sample_data["next"]



def log_radars(first_radar_tokens: list[str], nusc: nuscenes.NuScenes, max_timestamp_us: float) -> None:
    """Log radar data."""
    for first_radar_token in first_radar_tokens:
        current_camera_token = first_radar_token
        while current_camera_token != "":
            sample_data = nusc.get("sample_data", current_camera_token)
            if max_timestamp_us < sample_data["timestamp"]:
                break
            sensor_name = sample_data["channel"]
            rr.set_time("timestamp", timestamp=sample_data["timestamp"] * 1e-6)
            data_file_path = nusc.dataroot / sample_data["filename"]
            pointcloud = nuscenes.RadarPointCloud.from_file(str(data_file_path))
            points = pointcloud.points[:3].T  # shape after transposing: (num_points, 3)
            point_distances = np.linalg.norm(points, axis=1)
            point_colors = cmap(norm(point_distances))
            rr.log(
                f"world/ego_vehicle/{sensor_name}",
                rr.Points3D(points, colors=point_colors),
            )
            current_camera_token = sample_data["next"]



def log_annotations(location: str, first_sample_token: str, nusc: nuscenes.NuScenes, max_timestamp_us: float) -> None:
    """Log 3D bounding boxes."""
    label2id: dict[str, int] = {}
    current_sample_token = first_sample_token
    while current_sample_token != "":
        sample_data = nusc.get("sample", current_sample_token)
        if max_timestamp_us < sample_data["timestamp"]:
            break
        rr.set_time("timestamp", timestamp=sample_data["timestamp"] * 1e-6)
        ann_tokens = sample_data["anns"]
        sizes = []
        centers = []
        quaternions = []
        class_ids = []
        lat_lon = []
        for ann_token in ann_tokens:
            ann = nusc.get("sample_annotation", ann_token)

            rotation_xyzw = np.roll(ann["rotation"], shift=-1)  # go from wxyz to xyzw
            width, length, height = ann["size"]
            sizes.append((length, width, height))  # x, y, z sizes
            centers.append(ann["translation"])
            quaternions.append(rr.Quaternion(xyzw=rotation_xyzw))
            if ann["category_name"] not in label2id:
                label2id[ann["category_name"]] = len(label2id)
            class_ids.append(label2id[ann["category_name"]])
            lat_lon.append(derive_latlon(location, ann))

        rr.log(
            "world/anns",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                class_ids=class_ids,
            ),
            rr.GeoPoints(lat_lon=lat_lon),
        )
        current_sample_token = sample_data["next"]

    annotation_context = [(i, label) for label, i in label2id.items()]
    rr.log("world/anns", rr.AnnotationContext(annotation_context), static=True)


def log_sensor_calibration(sample_data: dict[str, Any], nusc: nuscenes.NuScenes) -> None:
    """Log sensor calibration (pinhole camera, sensor poses, etc.)."""
    sensor_name = sample_data["channel"]
    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    rotation_xyzw = np.roll(calibrated_sensor["rotation"], shift=-1)  # go from wxyz to xyzw
    rr.log(
        f"world/ego_vehicle/{sensor_name}",
        rr.Transform3D(
            translation=calibrated_sensor["translation"],
            rotation=rr.Quaternion(xyzw=rotation_xyzw),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    if len(calibrated_sensor["camera_intrinsic"]) != 0:
        rr.log(
            f"world/ego_vehicle/{sensor_name}",
            rr.Pinhole(
                image_from_camera=calibrated_sensor["camera_intrinsic"],
                width=sample_data["width"],
                height=sample_data["height"],
            ),
            static=True,
        )

import gzip
import pathlib

def write_depth_filenames_to_txt(
    nusc: nuscenes.NuScenes,
    scene_name: str,
    split: str,
    render_dir: pathlib.Path,
    output_txt_path: pathlib.Path,
) -> None:
    """
    Writes paths of depth .npy.gz files that match nuScenes camera keyframes to a txt file.
    """
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    token = scene["first_sample_token"]

    lines = []

    while token:
        sample = nusc.get("sample", token)
        for sd_token in sample["data"].values():
            sd = nusc.get("sample_data", sd_token)
            if sd["sensor_modality"] != "camera":
                continue

            cam = sd["channel"]  # e.g. CAM_FRONT
            lines.append(str(sd['filename']))

        token = sample["next"]

    # Write to txt
    with open(output_txt_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"✅ Wrote {len(lines)} depth file paths to {output_txt_path}")

import numpy as np
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def filter_depth_by_radar_fov(nusc, depth_data, sample_data, radar_channel="RADAR_FRONT", 
                               azimuth_deg=120, elevation_deg=10):
    """
    Filters a depth image to only include pixels whose 3D projections fall within the radar FOV.

    Parameters:
    - nusc: The NuScenes instance.
    - depth_data: (H, W) numpy array of depth values.
    - sample_data: dict, the current camera sample_data from NuScenes.
    - radar_channel: Which radar sensor to compare FOV against.
    - azimuth_deg: Horizontal FOV in degrees (± this value).
    - elevation_deg: Vertical FOV in degrees (± this value).

    Returns:
    - A masked version of depth_data where out-of-FOV pixels are set to 0.
    """

    cs_camera = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    K = np.array(cs_camera["camera_intrinsic"])
    T_cam_to_ego = transform_matrix(cs_camera["translation"], Quaternion(cs_camera["rotation"]), inverse=False)

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

    H, W = depth_data.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # (3, N)
    z = depth_data.flatten()
    valid = z > 0
    uv1 = uv1[:, valid]
    z = z[valid]

    xyz_cam = np.linalg.inv(K) @ (uv1 * z)  # (3, N)
    xyz_cam_homo = np.vstack([xyz_cam, np.ones((1, xyz_cam.shape[1]))])
    xyz_radar = T_cam_to_radar @ xyz_cam_homo
    xyz_radar = xyz_radar[:3, :]

    x, y, z = xyz_radar
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))

    az_fov = np.radians(azimuth_deg)
    el_fov = np.radians(elevation_deg)
    fov_mask = (np.abs(azimuth) < az_fov) & (np.abs(elevation) < el_fov)

    depth_filtered = depth_data.flatten()
    keep_mask = np.zeros_like(depth_filtered, dtype=bool)
    keep_mask[valid] = fov_mask
    depth_filtered[~keep_mask] = 0.0
    return depth_filtered.reshape(H, W)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the nuScenes dataset using the Rerun SDK.")
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        default=DATASET_DIR,
        help="Root directory of nuScenes dataset",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="scene-0103",
        help="Scene name to visualize (typically of form 'scene-xxxx')",
    )
    parser.add_argument("--dataset-version", type=str, default="v1.0-trainval", help="Scene id to visualize")
    parser.add_argument(
        "--seconds",
        type=float,
        default=float("inf"),
        help="If specified, limits the number of seconds logged",
    )

    rr.serve_web(open_browser=False)
    rr.script_add_args(parser)
    args = parser.parse_args()

    # ensure_scene_available(args.root_dir, args.dataset_version, args.scene_name)

    nusc = nuscenes.NuScenes(version=args.dataset_version, dataroot=args.root_dir, verbose=True)
    write_depth_filenames_to_txt(
        nusc,
        scene_name="scene-0103",
        split="val",
        render_dir=pathlib.Path("/work3/s203211/datasets/nuscenes/unnamed/neurad/2025-06-13_183130/render"),
        output_txt_path=pathlib.Path("depth_files_scene-0103.txt"),
    )
    # print(f"Found {len(camera_samples)} camera sample data entries in scene {args.scene_name}.")

    # Set up the Rerun Blueprint (how the visualization is organized):
    sensor_views = [
        rrb.Spatial2DView(
            name=sensor_name,
            origin=f"world/ego_vehicle/{sensor_name}",
            contents=["$origin/**", "world/anns"],
            overrides={"world/anns": rr.Boxes3D.from_fields(fill_mode="majorwireframe")},
        )
        for sensor_name in nuscene_sensor_names(nusc, args.scene_name)
    ]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D",
                    origin="world",
                    # Set the image plane distance to 5m for all camera visualizations.
                    defaults=[rr.Pinhole.from_fields(image_plane_distance=5.0)],
                    overrides={"world/anns": rr.Boxes3D.from_fields(fill_mode="solid")},
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="description", name="Description"),
                    rrb.MapView(
                        origin="world",
                        name="MapView",
                        zoom=rrb.archetypes.MapZoom(18.0),
                        background=rrb.archetypes.MapBackground(rrb.components.MapProvider.OpenStreetMap),
                    ),
                    row_shares=[1, 1],
                ),
                column_shares=[3, 1],
            ),
            rrb.Grid(*sensor_views),
            row_shares=[4, 2],
        ),
        rrb.TimePanel(state="collapsed"),
    )

    
    

    rr.script_setup(args, "rerun_example_nuscenes", default_blueprint=blueprint)
    rr.save("example_mod.rrd",default_blueprint=blueprint)

    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )

    log_nuscenes(nusc, args.scene_name, max_time_sec=args.seconds)
    

    rr.script_teardown(args)


if __name__ == "__main__":
    main()