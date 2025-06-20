from nuscenes import NuScenes
import gzip
import cv2
import numpy as np
import pathlib

class NuScenesMod(NuScenes):
    def __init__(self, version, dataroot, render_path=None, verbose=True, **kwargs):
        super().__init__(version=version, dataroot=dataroot, verbose=verbose, **kwargs)
        self._version = version
        self._dataroot = dataroot
        self._render_path = pathlib.Path(render_path) if render_path else None


    @property
    def render_path(self):
        return self._render_path

    def get_depth_data(self, sample_data: dict) -> np.ndarray:
        """Fetch, resize, and filter depth map for given camera sample_data."""
        if self._render_path is None:
            raise ValueError("render_path was not provided to NuScenesMod.")

        # Infer depth file path
        path_parts = sample_data["filename"].split("/")
        rel_subpath = pathlib.Path(path_parts[0]) / path_parts[1] / path_parts[2]
        rel_subpath = rel_subpath.with_suffix(".npy.gz")

        # Try train/val fallback
        for split in ["train", "val"]:
            depth_file_path = self._render_path / split / "raw-depth" / rel_subpath
            if depth_file_path.exists():
                break
        else:
            raise FileNotFoundError(f"No depth file found for: {depth_file_path}")

        # Load and resize
        with gzip.open(depth_file_path, "rb") as f:
            depth_data = np.load(f)

        img_path = pathlib.Path(self.dataroot) / sample_data["filename"]
        img = cv2.imread(str(img_path))
        depth_data = cv2.resize(depth_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Optionally filter
        # from vizcopy import filter_depth_by_radar_fov
        # depth_data = filter_depth_by_radar_fov(self, depth_data, sample_data)

        return depth_data