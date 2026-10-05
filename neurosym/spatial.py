"""Atlas acquisition and native-voxel parcels from the released spatial mappings.

No neural response or semantic target selects a spatial assignment. Vertex labels
vote through the release's mapper; each retained native voxel belongs to one
parcel. This avoids counting interpolated surface vertices as new measurements.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

import h5py
import nibabel as nib
import numpy as np
from scipy import sparse

from .io import immutable_json, object_hash, read_json, save_json
from .semantics import digest_file


def _fetch(url, limit=8 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "neurosym-research/1"})
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Atlas response exceeds the declared size limit.")
    return body


def acquire_atlas(root, config, *, download=False):
    root = Path(root)
    directory = root / config["assets"]
    identity = {k: config[k] for k in ("repository", "revision", "atlas", "source_directory")}
    if len(identity["revision"]) != 40 or any(c not in "0123456789abcdef" for c in identity["revision"]):
        raise ValueError("Atlas source requires a pinned Git revision.")
    receipt_path = directory / "source.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["source"] != identity:
            raise ValueError("Existing atlas has another source identity.")
        for item in receipt["files"]:
            if digest_file(directory / item["name"]) != item["sha256"]:
                raise ValueError("Atlas file changed: " + item["name"])
        return receipt
    if not download:
        raise FileNotFoundError("Atlas not acquired. Run scripts/prepare_spatial.py --download.")
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for hemi in config["hemisphere_order"]:
        name = f"{hemi}.{config['atlas']}.annot"
        relative = config["source_directory"] + "/" + name
        meta = json.loads(_fetch(f"https://api.github.com/repos/{config['repository']}/contents/{relative}?ref={config['revision']}"))
        url = f"https://raw.githubusercontent.com/{config['repository']}/{config['revision']}/{relative}"
        body = _fetch(url)
        blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if len(body) != meta["size"] or blob != meta["sha"]:
            raise ValueError("Atlas bytes differ from the pinned Git blob.")
        path = directory / name
        if path.exists() and path.read_bytes() != body:
            raise ValueError("Refusing to overwrite a different atlas file.")
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(body)
        partial.replace(path)
        files.append({"name": name, "url": url, "bytes": len(body), "git_blob": blob,
                      "sha256": hashlib.sha256(body).hexdigest()})
        print("ATLAS ACQUIRED", name, len(body), "bytes", flush=True)
    receipt = {"format_version": 1, "source": identity, "files": files}
    immutable_json(receipt_path, receipt)
    return receipt


def atlas_labels(root, config):
    receipt = acquire_atlas(root, config)
    labels, names = [], []
    for hemi in config["hemisphere_order"]:
        path = Path(root) / config["assets"] / f"{hemi}.{config['atlas']}.annot"
        local, _, table = nib.freesurfer.read_annot(path, orig_ids=False)
        if len(local) != config["vertices_per_hemisphere"]:
            raise ValueError("Atlas vertex resolution differs from the release mapper.")
        translated = np.full(len(local), -1, dtype=np.int32)
        for value, raw_name in enumerate(table):
            name = raw_name.decode("utf-8")
            if not name.startswith("17Networks_"):
                if "Medial_Wall" not in name and name.lower() not in {"unknown", "background"}:
                    raise ValueError("Unrecognized atlas background label: " + name)
                continue
            if f"_{hemi.upper()}_" not in name or name in names:
                raise ValueError("Atlas hemisphere/name identity is inconsistent.")
            translated[local == value] = len(names)
            names.append(name)
        labels.append(translated)
    if len(names) != config["parcels"] or any(not np.any(np.concatenate(labels) == i) for i in range(len(names))):
        raise ValueError("Atlas does not contain all configured parcels.")
    return np.concatenate(labels), names, receipt


def surface_mapper(path, spec, n_vertices):
    def load(file, name):
        shape = tuple(int(x) for x in file[name + "_shape"][()])
        if list(shape) != spec["matrices"][name]["shape"]:
            raise ValueError("Mapper shape changed since release inspection.")
        result = sparse.csr_matrix((file[name + "_data"][()], file[name + "_indices"][()],
                                    file[name + "_indptr"][()]), shape=shape, dtype=np.float64)
        result.check_format(full_check=True)
        if not np.isfinite(result.data).all():
            raise ValueError("Surface mapper must have finite weights.")
        return result
    with h5py.File(path, "r") as file:
        if "voxel_to_fsaverage" in spec["matrices"]:
            mapper = load(file, "voxel_to_fsaverage")
            layout = "released concatenated fsaverage; left then right"
        elif {"vox_to_fsavg_left", "vox_to_fsavg_right"} <= set(spec["matrices"]):
            mapper = sparse.vstack([load(file, "vox_to_fsavg_left"), load(file, "vox_to_fsavg_right")], format="csr")
            layout = "explicit left/right release matrices stacked left then right"
        else:
            raise ValueError("No verified fsaverage mapper is available.")
    if mapper.shape[0] != n_vertices or mapper.shape[1] not in spec["voxel_dimensions"]:
        raise ValueError("Surface/voxel dimensions do not agree with atlas and release.")
    # Released interpolation operators contain signed coefficients. Atlas
    # membership uses their absolute spatial influence, not signed signal
    # interpolation or an assumption that mapper coefficients are probabilities.
    negative_coefficients = int(np.count_nonzero(mapper.data < 0))
    mapper.data = np.abs(mapper.data)
    total = np.asarray(mapper.sum(axis=1)).ravel()
    inverse = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
    return sparse.diags(inverse) @ mapper, total > 0, layout, negative_coefficients


def prepare_spatial(root, config_path, *, download=False):
    root = Path(root).resolve()
    config = read_json(Path(config_path))
    if (config["assignment"] != "maximum_surface_label_vote_per_native_voxel" or
            config["vote_weights"] != "absolute_mapper_coefficients_normalized_per_vertex" or
            config["ties"] != "exclude" or config["hemisphere_order"] != ["lh", "rh"]):
        raise ValueError("Unsupported spatial assignment policy.")
    acquire_atlas(root, config, download=download)
    labels, names, receipt = atlas_labels(root, config)
    contract = read_json(root / config["contract"])
    identity = {"config": config, "atlas_source": receipt, "contract_hash": object_hash(contract),
                "code_sha256": digest_file(Path(__file__)), "parcel_names": names,
                "scope": "Native cortical voxel assignment; not concept localization or new neural measurements."}
    output = root / config["output"] / object_hash(identity)
    if (output / "complete.json").is_file():
        view = SpatialView(output)
        for subject in contract["mappers"]:
            view.assignments(subject)
        save_json(root / config["output"] / "latest.json", {"path": object_hash(identity)})
        return output
    immutable_json(output / "identity.json", identity)
    cortex = labels >= 0
    grouping = sparse.csr_matrix((np.ones(cortex.sum()), (labels[cortex], np.flatnonzero(cortex))),
                                shape=(len(names), len(labels)))
    subjects = []
    for subject, spec in sorted(contract["mappers"].items()):
        source = root / config["raw"] / "mappers" / (subject + "_mappers.hdf")
        expected = next(f["sha256"] for f in contract["source_files"] if f["path"] == "mappers/" + source.name)
        if digest_file(source) != expected:
            raise ValueError("Actual mapper differs from inspected release: " + subject)
        mapper, observed, layout, negative_coefficients = surface_mapper(source, spec, len(labels))
        votes = (grouping @ mapper).toarray()
        total = votes.sum(axis=0)
        maximum = votes.max(axis=0)
        tied = (np.isclose(votes, maximum[None], rtol=0, atol=1e-12).sum(axis=0) != 1)
        assignment = votes.argmax(axis=0).astype(np.int32)
        assignment[(total == 0) | tied] = -1
        confidence = np.divide(maximum, total, out=np.zeros_like(total), where=total > 0)
        counts = np.bincount(assignment[assignment >= 0], minlength=len(names))
        np.savez_compressed(output / (subject + ".npz"), parcel=assignment, vote_fraction=confidence,
                            parcel_counts=counts, surface_observed=observed)
        item = {"subject": subject, "mapper_sha256": expected, "layout": layout,
                "signed_mapper_coefficients": negative_coefficients,
                "voxels": len(assignment), "assigned_voxels": int((assignment >= 0).sum()),
                "unmapped_or_medial_voxels": int((total == 0).sum()),
                "ambiguous_boundary_voxels": int(((total > 0) & tied).sum()),
                "observed_surface_vertices": int(observed.sum()),
                "parcel_counts": counts.tolist(), "missing_parcels": [names[i] for i in np.flatnonzero(counts == 0)],
                "assignment_sha256": digest_file(output / (subject + ".npz"))}
        subjects.append(item)
        print("SPATIAL", subject, item["assigned_voxels"], "/", len(assignment), "native voxels;",
              int((counts > 0).sum()), "parcels", flush=True)
    save_json(output / "complete.json", {"identity_hash": object_hash(identity), "subjects": subjects,
              "hemisphere_convention": "fsaverage concatenated left/right; subject04 explicit hemisphere matrices",
              "anatomical_registration_reestimated": False, "neural_responses_used": False})
    save_json(root / config["output"] / "latest.json", {"path": object_hash(identity)})
    return output


class SpatialView:
    def __init__(self, path):
        self.path = Path(path)
        self.identity = read_json(self.path / "identity.json")
        self.receipt = read_json(self.path / "complete.json")
        self.identity_hash = object_hash(self.identity)
        if self.receipt["identity_hash"] != self.identity_hash:
            raise ValueError("Spatial identity mismatch.")
        self.names = self.identity["parcel_names"]

    def assignments(self, subject):
        entry = next((s for s in self.receipt["subjects"] if s["subject"] == subject), None)
        if entry is None or digest_file(self.path / (subject + ".npz")) != entry["assignment_sha256"]:
            raise ValueError("Missing or changed spatial assignment.")
        with np.load(self.path / (subject + ".npz"), allow_pickle=False) as file:
            return file["parcel"].copy()
