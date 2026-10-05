"""Deniz data adapter based on the inspected release, with explicit repeat axes."""

from pathlib import Path
import numpy as np
import h5py

from .io import read_json


class DenizReader:
    def __init__(self, raw_root: Path, contract_path: Path):
        self.root = Path(raw_root)
        self.contract = read_json(Path(contract_path))
        if self.contract["format_version"] != 1:
            raise ValueError("Unsupported Deniz data contract.")

    def response(self, subject: str, story: str, voxels: slice = slice(None), *, trim: bool = True) -> np.ndarray:
        """Return repetitions x time x native cortical voxels; never average repeats."""
        spec = self.contract["subjects"][subject][story]
        path = self.root / "responses" / f"{subject}_reading_fmri_data_{spec['split']}.hdf"
        with h5py.File(path, "r") as file:
            values = file[story]
            if list(values.shape) != spec["shape"]:
                raise ValueError(f"Response layout changed since inspection: {path}/{story}")
            selection = [slice(None)] * values.ndim
            selection[spec["voxel_axis"]] = voxels
            selection[spec["time_axis"]] = slice(10, -10) if trim else slice(None)
            result = np.asarray(values[tuple(selection)], dtype=np.float32)
        if spec["repeat_axis"] is None:
            result = np.moveaxis(result, [spec["time_axis"], spec["voxel_axis"]], [0, 1])[None]
        else:
            result = np.moveaxis(result, [spec["repeat_axis"], spec["time_axis"], spec["voxel_axis"]], [0, 1, 2])
        return result

    def response_batches(self, subject: str, story: str, batch_voxels: int = 4096, *, trim: bool = True):
        if batch_voxels < 1:
            raise ValueError("batch_voxels must be positive.")
        n_voxels = self.contract["subjects"][subject][story]["voxels"]
        for first in range(0, n_voxels, batch_voxels):
            section = slice(first, min(first + batch_voxels, n_voxels))
            yield section, self.response(subject, story, section, trim=trim)

    def features(self, story: str, names: list[str] | None = None, *, trim: bool = True) -> dict[str, np.ndarray]:
        split = "val" if story == "story_11" else "trn"
        path = self.root / "features" / f"features_{split}_NEW.hdf"
        output = {}
        with h5py.File(path, "r") as file:
            for name in names if names is not None else file[story].keys():
                values = file[story][name]
                expected = self.contract["feature_shapes"][story][name]
                if list(values.shape) != expected:
                    raise ValueError(f"Feature layout changed: {path}/{story}/{name}")
                output[name] = np.asarray(values[10:-5] if trim else values[()], dtype=np.float32)
        return output

    def mapper_arrays(self, subject: str) -> dict:
        """Return CSR components, including the distinct subject04 naming convention."""
        mappings = self.contract["mappers"][subject]["matrices"]
        output = {}
        with h5py.File(self.root / "mappers" / f"{subject}_mappers.hdf", "r") as file:
            for name, metadata in mappings.items():
                output[name] = {field: file[f"{name}_{field}"][()] for field in ("data", "indices", "indptr")}
                output[name]["shape"] = tuple(metadata["shape"])
        return output

    def roi_masks(self, subject: str) -> dict[str, np.ndarray]:
        with h5py.File(self.root / "mappers" / f"{subject}_mappers.hdf", "r") as file:
            return {name.removeprefix("roi_mask_"): file[name][()] for name in file if name.startswith("roi_mask_")}
