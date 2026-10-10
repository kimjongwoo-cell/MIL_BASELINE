#!/usr/bin/env python3
"""Resumable paired 5x/20x CONCH extraction for fully downloaded BRACS SVS files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import openslide
import torch
from PIL import Image

from clam_base_utils.conch.open_clip_custom import create_model_from_pretrained
from clam_base_utils.wsi_core.WholeSlideImage import WholeSlideImage
from clam_base_utils.wsi_core.util_classes import isInContourV3_Easy
from clam_base_utils.wsi_core.wsi_utils import isBlackPatch, isWhitePatch


SOURCE = Path("/media/super/16TB/Slide/data/BRACS/BRACS_WSI")
INVENTORY = Path("/home/super/Jong/research/data/BRACS/source/official/download_inventory.json")
WEIGHTS = Path("/home/super/Jong/research/data/_models/CONCH/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin")
DEFAULT_OUTPUT = Path("/media/super/16TB/Slide/data/BRACS/conch_x5_x20_v1")
DEFAULT_STATUS = Path("/home/super/Jong/research/data/BRACS/preparation_v1/extraction_status.json")
DEFAULT_WORKBOOK = Path("/media/super/16TB/Slide/data/BRACS/BRACS.xlsx")
MODEL_NAME = "conch_ViT-B-16"
PROTOCOL = "bracs_paired_5x_20x_conch_v1"
EXPECTED_WEIGHTS_SHA256 = "40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c"
FEATURE_DIM = 512
ENCODE_BATCH_SIZE = 32
PARENT_SIZE = 224
CHILD_SIZE = 224


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_slides(path: Path, source: Path) -> dict[Path, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = {}
    for row in payload["files"]:
        remote = row["remote_path"]
        marker = "/BRACS_WSI/"
        if marker not in remote or not remote.lower().endswith(".svs"):
            continue
        rel = remote.split(marker, 1)[1]
        files[source / rel] = int(row["bytes"])
    return files


def workbook_provenance(path: Path) -> tuple[dict[str, dict], dict]:
    import pandas as pd

    frame = pd.read_excel(path, sheet_name="WSI_Information")
    rows = {}
    patients_by_split: dict[str, set[str]] = {}
    split_names = {"training": "train", "validation": "val", "testing": "test"}
    for _, row in frame.iterrows():
        split = str(row["Set"]).strip().lower()
        split = split_names.get(split, split)
        patient = str(row["Patient Id"])
        rows[str(row["WSI Filename"]).strip()] = {
            "patient_id": patient,
            "label": str(row["WSI label"]),
            "workbook_split": split,
        }
        patients_by_split.setdefault(split, set()).add(patient)
    overlaps = {}
    splits = sorted(patients_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            common = sorted(patients_by_split[left] & patients_by_split[right])
            if common:
                overlaps[f"{left}__{right}"] = common
    summary = {
        "workbook": str(path),
        "workbook_sha256": sha256_file(path),
        "sheet": "WSI_Information",
        "rows": len(rows),
        "patient_overlap_by_workbook_split": overlaps,
        "split_counts": {key: len(value) for key, value in sorted(patients_by_split.items())},
    }
    return rows, summary


def slide_geometry(slide: openslide.OpenSlide) -> dict:
    props = slide.properties
    objectives = []
    for key in (openslide.PROPERTY_NAME_OBJECTIVE_POWER, "aperio.AppMag"):
        raw = props.get(key)
        match = re.search(r"\d+(?:\.\d+)?", str(raw)) if raw else None
        if match:
            objectives.append((key, float(match.group())))
    if not objectives:
        raise ValueError("missing objective-power metadata; cannot verify 5x/20x geometry")
    if any(abs(value - objectives[0][1]) > max(1.0, objectives[0][1] * 0.05) for _, value in objectives[1:]):
        raise ValueError(f"objective-power properties disagree: {objectives}")
    objective = objectives[0][1]

    mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)
    try:
        mpp_x = float(mpp_x) if mpp_x else None
        mpp_y = float(mpp_y) if mpp_y else None
    except (TypeError, ValueError):
        mpp_x = mpp_y = None
    if not math.isfinite(objective) or objective <= 0:
        raise ValueError("invalid objective-power metadata")
    if mpp_x is None or mpp_y is None:
        raise ValueError("missing OpenSlide MPP metadata; cannot verify physical footprint")
    if not all(math.isfinite(v) and v > 0 for v in (mpp_x, mpp_y)):
        raise ValueError("invalid MPP metadata")
    if abs(mpp_x - mpp_y) / max(mpp_x, mpp_y) > 0.02:
        raise ValueError(f"anisotropic MPP values x={mpp_x}, y={mpp_y}")
    inferred_20x_mpp = 10.0 / objective
    if abs((mpp_x + mpp_y) / 2.0 - inferred_20x_mpp) / inferred_20x_mpp > 0.25:
        raise ValueError(
            f"objective/MPP disagreement: objective={objective:g}x, mpp=({mpp_x:g},{mpp_y:g})"
        )

    # Target MPP at a requested magnification is measured_mpp*objective/target_mag.
    # Convert the physical crop to level 0, preserving exact 4:1 parent/child spans.
    target5_x = mpp_x * objective / 5.0
    target5_y = mpp_y * objective / 5.0
    target20_x = mpp_x * objective / 20.0
    target20_y = mpp_y * objective / 20.0
    parent_w = int(round(PARENT_SIZE * target5_x / mpp_x))
    parent_h = int(round(PARENT_SIZE * target5_y / mpp_y))
    child_w = int(round(CHILD_SIZE * target20_x / mpp_x))
    child_h = int(round(CHILD_SIZE * target20_y / mpp_y))
    if parent_w != child_w * 4 or parent_h != child_h * 4:
        raise ValueError("rounded 5x parent footprint does not divide into 4x4 20x children")
    return {
        "objective_power": objective,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "mpp_source": "openslide_properties",
        "parent_level0_wh": [parent_w, parent_h],
        "child_level0_wh": [child_w, child_h],
        "physical_parent_um": [PARENT_SIZE * target5_x, PARENT_SIZE * target5_y],
        "physical_child_um": [CHILD_SIZE * target20_x, CHILD_SIZE * target20_y],
    }


def existing_clam_coordinates(wsi: WholeSlideImage, slide_id: str, scratch: Path, parent_wh: list[int]) -> np.ndarray:
    # Reuse CLAM's tissue segmentation, contour/hole filtering and four-point
    # parent selector; this only saves coordinates, never image patches.
    seg_level = wsi.wsi.get_best_level_for_downsample(64)
    seg_w, seg_h = wsi.level_dim[seg_level]
    if seg_w * seg_h > 100_000_000:
        raise ValueError(f"CLAM segmentation level too large: {seg_w}x{seg_h}")
    wsi.segmentTissue(
        seg_level=seg_level,
        sthresh=8,
        mthresh=7,
        close=4,
        use_otsu=False,
        filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8},
    )
    if not wsi.contours_tissue:
        return np.empty((0, 2), dtype=np.int64)
    wsi.process_contours(
        save_path=str(scratch),
        real_patch_size=PARENT_SIZE,
        patch_level=0,
        patch_size=parent_wh[0],
        step_size=parent_wh[0],
        strict_control=True,
        contour_fn="four_pt",
        use_padding=True,
    )
    coord_path = scratch / f"{slide_id}.h5"
    if not coord_path.is_file():
        return np.empty((0, 2), dtype=np.int64)
    with h5py.File(coord_path, "r") as f:
        coords = np.asarray(f["coords"], dtype=np.int64)
    width, height = wsi.level_dim[0]
    keep = (
        (coords[:, 0] >= 0)
        & (coords[:, 1] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] < height)
    )
    return coords[keep]


def child_tissue_valid(wsi: WholeSlideImage, x: int, y: int, child_wh: list[int]) -> bool:
    cw, ch = child_wh
    sw, sh = wsi.level_dim[0]
    if x < 0 or y < 0 or x + cw > sw or y + ch > sh:
        return False
    for contour, holes in zip(wsi.contours_tissue or [], wsi.holes_tissue or []):
        checker = isInContourV3_Easy(contour=contour, patch_size=cw, center_shift=0.5)
        if wsi.isInContours(checker, (x, y), holes, cw):
            return True
    return False


def image_and_tensor(slide: openslide.OpenSlide, xy: tuple[int, int], wh: tuple[int, int], preprocess):
    def read_at_finest_adequate_level(location, level0_wh):
        target_downsample = level0_wh[0] / PARENT_SIZE
        downsamples = slide.level_downsamples
        candidates = [i for i, ds in enumerate(downsamples) if float(ds) <= target_downsample + 1e-6]
        level = 0
        size = tuple(level0_wh)
        for candidate in sorted(candidates, key=lambda i: float(downsamples[i]), reverse=True):
            ds = float(downsamples[candidate])
            native_size = (max(1, int(round(level0_wh[0] / ds))), max(1, int(round(level0_wh[1] / ds))))
            if all(abs(ds * native_size[axis] - level0_wh[axis]) < 1e-6 for axis in (0, 1)):
                level, size = candidate, native_size
                break
        return slide.read_region(location, level, size)

    rgba = read_at_finest_adequate_level(xy, wh)
    image = Image.new("RGB", rgba.size, (255, 255, 255))
    image.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    if image.size != (PARENT_SIZE, PARENT_SIZE):
        image = image.resize((PARENT_SIZE, PARENT_SIZE), Image.Resampling.BICUBIC)
    return image, preprocess(image)


def prepare_parent(
    slide: openslide.OpenSlide,
    wsi: WholeSlideImage,
    parent_index: int,
    coords: np.ndarray,
    parent_wh: list[int],
    child_wh: list[int],
    preprocess,
) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[int, int]]]:
    """Prepare one parent and its valid, row-major children without encoding."""
    x, y = map(int, coords[parent_index])
    _, parent_tensor = image_and_tensor(slide, (x, y), tuple(parent_wh), preprocess)
    child_tensors: list[torch.Tensor] = []
    child_slots: list[tuple[int, int]] = []
    for gy in range(4):
        for gx in range(4):
            child_index = gy * 4 + gx
            cx, cy = x + gx * child_wh[0], y + gy * child_wh[1]
            if not child_tissue_valid(wsi, cx, cy, child_wh):
                continue
            child_image, child_tensor = image_and_tensor(slide, (cx, cy), tuple(child_wh), preprocess)
            rgb = np.asarray(child_image)
            if isBlackPatch(rgb, rgbThresh=50) or isWhitePatch(rgb, satThresh=15):
                continue
            child_slots.append((parent_index, child_index))
            child_tensors.append(child_tensor)
    return parent_tensor, child_tensors, child_slots


def prepare_batch(prepared_parents) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
    """Flatten parent results in their supplied order, preserving child slots."""
    parent_tensors: list[torch.Tensor] = []
    child_tensors: list[torch.Tensor] = []
    child_slots: list[tuple[int, int]] = []
    for parent_tensor, children, slots in prepared_parents:
        parent_tensors.append(parent_tensor)
        child_tensors.extend(children)
        child_slots.extend(slots)
    return parent_tensors, child_tensors, child_slots


def encode(model, tensors: list[torch.Tensor], device: torch.device) -> np.ndarray:
    outputs = []
    for start in range(0, len(tensors), ENCODE_BATCH_SIZE):
        batch = torch.stack(tensors[start : start + ENCODE_BATCH_SIZE]).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.encode_image(batch, proj_contrast=False, normalize=False)
        result = result.float()
        if result.ndim != 2 or result.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"unexpected CONCH embedding shape {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise RuntimeError("CONCH returned non-finite embeddings")
        half = result.cpu().numpy().astype(np.float16, copy=False)
        if not np.isfinite(half).all():
            raise RuntimeError("CONCH embeddings overflowed float16 storage")
        outputs.append(half)
        del batch, result
    if not outputs:
        return np.empty((0, FEATURE_DIM), dtype=np.float16)
    return np.concatenate(outputs, axis=0)


def extract_one(
    path: Path,
    expected_bytes: int,
    output_path: Path,
    model,
    preprocess,
    device: torch.device,
    identity: str,
    provenance: dict,
    progress: Callable[[str, int, int], None],
    patch_workers: int = 4,
) -> dict:
    if patch_workers < 0:
        raise ValueError("patch_workers must be zero or greater")
    extraction_started = time.perf_counter()
    source_before = path.stat()
    actual_bytes = source_before.st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"source is not complete: got {actual_bytes} bytes, expected {expected_bytes}")
    slide = openslide.OpenSlide(str(path))
    wsi = None
    executor = None
    worker_slides: list[openslide.OpenSlide] = []
    worker_slides_lock = threading.Lock()
    worker_local = threading.local()
    try:
        geometry = slide_geometry(slide)
        parent_wh = geometry["parent_level0_wh"]
        child_wh = geometry["child_level0_wh"]
        progress("segment_and_select", 0, 0)
        wsi = WholeSlideImage(str(path))
        wsi.contours_tissue = None
        wsi.holes_tissue = None
        try:
            with tempfile.TemporaryDirectory(prefix="bracs-clam-coords-") as temp_dir:
                coords = existing_clam_coordinates(wsi, path.stem, Path(temp_dir), parent_wh)
        except Exception as exc:
            if not wsi.contours_tissue:
                source_after = path.stat()
                if (source_after.st_size, source_after.st_mtime_ns, source_after.st_ino) != (
                    source_before.st_size, source_before.st_mtime_ns, source_before.st_ino
                ):
                    raise RuntimeError("source slide changed during feature extraction")
                return {"state": "no_tissue", "reason": f"no tissue candidates: {type(exc).__name__}: {exc}", "geometry": geometry}
            raise
        p = len(coords)
        if p == 0:
            source_after = path.stat()
            if (source_after.st_size, source_after.st_mtime_ns, source_after.st_ino) != (
                source_before.st_size, source_before.st_mtime_ns, source_before.st_ino
            ):
                raise RuntimeError("source slide changed during feature extraction")
            return {"state": "no_tissue", "reason": "CLAM tissue segmentation selected zero complete-origin parent tiles", "geometry": geometry}
        segmentation_seconds = time.perf_counter() - extraction_started
        patch_wait_seconds = 0.0
        encode_seconds = 0.0
        x5 = np.zeros((p, FEATURE_DIM), dtype=np.float16)
        x20 = np.zeros((p, 16, FEATURE_DIM), dtype=np.float16)
        valid = np.zeros((p, 16), dtype=np.bool_)
        progress("encoding", 0, p)
        if patch_workers:
            executor = ThreadPoolExecutor(max_workers=patch_workers)

            def prepare_in_worker(parent_index: int):
                worker_slide = getattr(worker_local, "slide", None)
                if worker_slide is None:
                    worker_slide = openslide.OpenSlide(str(path))
                    worker_local.slide = worker_slide
                    with worker_slides_lock:
                        worker_slides.append(worker_slide)
                return prepare_parent(worker_slide, wsi, parent_index, coords, parent_wh, child_wh, preprocess)

        else:
            def prepare_in_worker(parent_index: int):
                return prepare_parent(slide, wsi, parent_index, coords, parent_wh, child_wh, preprocess)

        next_start = 0
        next_end = min(ENCODE_BATCH_SIZE, p)
        pending = (
            [executor.submit(prepare_in_worker, i) for i in range(next_start, next_end)]
            if executor is not None
            else None
        )
        for start in range(0, p, ENCODE_BATCH_SIZE):
            end = min(start + ENCODE_BATCH_SIZE, p)
            progress("patch_prepare", start, p)
            preparation_started = time.perf_counter()
            if pending is None:
                prepared = [prepare_in_worker(i) for i in range(start, end)]
            else:
                prepared = [future.result() for future in pending]
            # Keep at most one following parent chunk in flight while this
            # chunk is encoded; retrieving futures in submission order keeps
            # the original parent and row-major child order.
            next_start, next_end = end, min(end + ENCODE_BATCH_SIZE, p)
            pending = (
                [executor.submit(prepare_in_worker, i) for i in range(next_start, next_end)]
                if executor is not None and next_start < p
                else None
            )
            parent_tensors, child_tensors, child_slots = prepare_batch(prepared)
            patch_wait_seconds += time.perf_counter() - preparation_started
            progress("encoding", start, p)
            encoding_started = time.perf_counter()
            x5[start:end] = encode(model, parent_tensors, device)
            if child_tensors:
                child_features = encode(model, child_tensors, device)
                for (parent_index, child_index), features in zip(child_slots, child_features):
                    x20[parent_index, child_index] = features
                    valid[parent_index, child_index] = True
            encode_seconds += time.perf_counter() - encoding_started
            progress("encoding", end, p)

        source_after = path.stat()
        if (source_after.st_size, source_after.st_mtime_ns, source_after.st_ino) != (
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ino,
        ):
            raise RuntimeError("source slide changed during feature extraction")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for worker_slide in worker_slides:
            worker_slide.close()
        slide.close()
        if wsi is not None:
            wsi.wsi.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_output = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    try:
        with h5py.File(temp_output, "w") as f:
            f.create_dataset("x5_conch", data=x5, dtype="float16", chunks=True, compression="lzf")
            f.create_dataset("x20_conch", data=x20, dtype="float16", chunks=True, compression="lzf")
            f.create_dataset("child_valid_mask", data=valid, dtype="bool", chunks=True, compression="lzf")
            f.create_dataset("parent_bbox_lv0", data=np.column_stack((coords, np.tile(parent_wh, (p, 1)))), dtype="int64")
            f.attrs["protocol"] = PROTOCOL
            f.attrs["identity"] = identity
            f.attrs["source_path"] = str(path)
            f.attrs["source_expected_bytes"] = expected_bytes
            f.attrs["source_observed_bytes"] = actual_bytes
            f.attrs["source_mtime_ns_before"] = source_before.st_mtime_ns
            f.attrs["source_mtime_ns_after"] = source_after.st_mtime_ns
            f.attrs["source_inode"] = source_before.st_ino
            f.attrs["feature_model"] = MODEL_NAME
            f.attrs["feature_dim"] = FEATURE_DIM
            f.attrs["feature_dtype"] = "float16"
            f.attrs["coordinate_convention"] = "level0_xywh; origin is pixel top-left; half-open extent"
            f.attrs["geometry_json"] = json.dumps(geometry, sort_keys=True)
            f.attrs["parent_count"] = p
            f.attrs["valid_child_count"] = int(valid.sum())
            f.attrs["child_validity"] = "inside slide; CLAM tissue contour/holes four-point selector; not all-black/white"
            f.attrs["workbook_provenance_json"] = json.dumps(provenance, sort_keys=True)
            f.flush()
        os.link(temp_output, output_path)
    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)
    return {
        "state": "complete",
        "output": str(output_path),
        "parents": p,
        "valid_children": int(valid.sum()),
        "source_bytes": actual_bytes,
        "source_mtime_ns_before": source_before.st_mtime_ns,
        "source_mtime_ns_after": source_after.st_mtime_ns,
        "source_inode": source_before.st_ino,
        "geometry": geometry,
        "dataset_provenance": provenance,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timing_seconds": {
            "total": time.perf_counter() - extraction_started,
            "segmentation": segmentation_seconds,
            "patch_wait": patch_wait_seconds,
            "encoding": encode_seconds,
        },
    }


def validated_resume_identity(audit_path: Path | None, config: dict) -> str | None:
    if audit_path is None:
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    baseline = audit["baseline_config"]
    baseline_identity = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    if audit.get("compatible_identity") != baseline_identity:
        raise ValueError("resume audit baseline identity does not match its config")
    if audit.get("optimized_source_sha256") != config["implementation_sha256"]:
        raise ValueError("resume audit does not cover the current implementation")
    snapshot = Path(audit["baseline_source"])
    if sha256_file(snapshot) != baseline.get("implementation_sha256"):
        raise ValueError("resume audit baseline source hash mismatch")
    performance_keys = {"implementation_sha256", "patch_workers", "prefetch_batches"}
    scientific_config = lambda value: {k: v for k, v in value.items() if k not in performance_keys}
    if scientific_config(baseline) != scientific_config(config):
        raise ValueError("resume audit cannot authorize a changed feature protocol")
    checks = audit.get("checks", {})
    required = {"preprocessed_tensors_exact", "child_slots_exact", "features_exact", "coordinates_exact", "masks_exact"}
    if audit.get("state") != "passed" or not all(checks.get(key) is True for key in required):
        raise ValueError("resume requires a passed patch and feature equivalence audit")
    return baseline_identity


def main() -> None:
    global ENCODE_BATCH_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--watch", action="store_true", help="poll until all inventory slides are extracted")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-slides", type=int, default=0, help="stop after this many new completed slides; 0 means unlimited")
    parser.add_argument("--batch-size", type=int, default=ENCODE_BATCH_SIZE)
    parser.add_argument("--patch-workers", type=int, default=4, help="ordered CPU patch preparation workers; 0 keeps serial preparation")
    parser.add_argument("--resume-audit", type=Path, help="passed equivalence receipt permitting reuse of the original implementation's outputs")
    args = parser.parse_args()
    args.status.parent.mkdir(parents=True, exist_ok=True)
    lock_file = args.status.with_suffix(args.status.suffix + ".lock")
    lock_handle = lock_file.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another BRACS extraction process holds {lock_file}") from exc
    if args.batch_size < 1 or args.batch_size > 64:
        parser.error("--batch-size must be between 1 and 64")
    if not 0 <= args.patch_workers <= 16:
        parser.error("--patch-workers must be between 0 and 16")
    ENCODE_BATCH_SIZE = args.batch_size

    if not args.weights.is_file() or not args.inventory.is_file() or not args.workbook.is_file():
        raise FileNotFoundError("CONCH weights, BRACS download inventory, or BRACS workbook not found")
    inventory = inventory_slides(args.inventory, args.source)
    weights_sha = sha256_file(args.weights)
    if weights_sha != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(f"CONCH weights SHA256 mismatch: {weights_sha}")
    workbook_rows, workbook_summary = workbook_provenance(args.workbook)
    config = {
        "protocol": PROTOCOL,
        "implementation_sha256": sha256_file(Path(__file__)),
        "patch_workers": args.patch_workers,
        "prefetch_batches": 1 if args.patch_workers else 0,
        "encoder_sha256": weights_sha,
        "model": MODEL_NAME,
        "preprocess": "official create_model_from_pretrained transform",
        "encode_image": {"proj_contrast": False, "normalize": False},
        "precision": "cuda_bfloat16_autocast_required",
        "encode_batch_size": ENCODE_BATCH_SIZE,
        "parent_pixels_at_5x": PARENT_SIZE,
        "children_per_parent": 16,
        "child_pixels_at_20x": CHILD_SIZE,
        "child_order": "row_major_y_x",
        "segmentation": {"seg_level": "best_level_for_downsample_64", "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False},
        "filter": {"a_t": 100, "a_h": 16, "max_n_holes": 8},
        "parent_selector": "CLAM process_contours strict_control=True contour_fn=four_pt use_padding=True",
        "child_selector": "CLAM four_pt contour and holes plus black/white patch filters",
        "boundary_policy": "retain in-bounds origin; white pad parent crop; mask out-of-bounds children",
        "workbook_sha256": workbook_summary["workbook_sha256"],
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    compatible_identity = validated_resume_identity(args.resume_audit, config)
    accepted_identities = {identity}
    if compatible_identity is not None:
        accepted_identities.add(compatible_identity)
    status = {"schema_version": 1, "config": config, "identity": identity, "slides": {}}
    if args.status.exists():
        prior = json.loads(args.status.read_text(encoding="utf-8"))
        if prior.get("identity") == identity:
            status = prior
            status.setdefault("slides", {})
        else:
            if prior.get("identity") in accepted_identities:
                status["slides"] = json.loads(json.dumps(prior.get("slides", {})))
            status["history"] = prior.get("history", []) + [{
                "identity": prior.get("identity"),
                "config": prior.get("config"),
                "slides": prior.get("slides", {}),
            }]
    status["dataset_provenance"] = workbook_summary
    status["execution"] = {"patch_workers": args.patch_workers, "prefetch_batches": config["prefetch_batches"], "resume_audit": str(args.resume_audit) if args.resume_audit else None}
    args.output.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; refusing the authorized GPU-only extraction")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible CUDA device does not support required BF16 inference")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print(f"CONCH device={device}; CUDA BF16=True; batch={ENCODE_BATCH_SIZE}; patch_workers={args.patch_workers}", flush=True)
    model, preprocess = create_model_from_pretrained(MODEL_NAME, checkpoint_path=str(args.weights))
    model = model.to(device).eval()
    done_this_run = 0
    while True:
        pending = []
        for path, expected in sorted(inventory.items()):
            if not path.is_file() or path.stat().st_size != expected:
                continue
            output_path = args.output / path.relative_to(args.source).with_suffix(".h5")
            entry = status["slides"].get(str(path), {})
            if output_path.is_file():
                source_stat = path.stat()
                if (entry.get("state") == "complete" and entry.get("identity") in accepted_identities
                        and entry.get("output") == str(output_path)
                        and entry.get("source_bytes") == source_stat.st_size
                        and entry.get("source_mtime_ns_after") == source_stat.st_mtime_ns
                        and entry.get("source_inode") == source_stat.st_ino):
                    continue
                try:
                    with h5py.File(output_path, "r") as f:
                        identity_ok = (
                            f.attrs.get("identity", "") in accepted_identities
                            and f.attrs.get("source_path", "") == str(path)
                            and int(f.attrs.get("source_expected_bytes", -1)) == expected
                            and int(f.attrs.get("source_mtime_ns_after", -1)) == source_stat.st_mtime_ns
                            and int(f.attrs.get("source_inode", -1)) == source_stat.st_ino
                        )
                        n = f["x5_conch"].shape[0]
                        valid_output = identity_ok and (
                            f["x5_conch"].shape == (n, FEATURE_DIM)
                            and f["x20_conch"].shape == (n, 16, FEATURE_DIM)
                            and f["child_valid_mask"].shape == (n, 16)
                            and f["parent_bbox_lv0"].shape == (n, 4)
                            and f["x5_conch"].dtype == np.dtype("float16")
                            and f["x20_conch"].dtype == np.dtype("float16")
                            and f["child_valid_mask"].dtype == np.dtype("bool")
                            and f["parent_bbox_lv0"].dtype == np.dtype("int64")
                        )
                        if valid_output and n:
                            boxes = f["parent_bbox_lv0"][:]
                            valid_output = bool(np.all(boxes[:, 2:] > 0) and len(np.unique(boxes[:, :2], axis=0)) == n)
                            valid_output &= bool(np.isfinite(f["x5_conch"][:]).all())
                            valid_output &= bool(np.isfinite(f["x20_conch"][:]).all())
                        parents = n
                        output_identity = f.attrs.get("identity", "")
                except Exception:
                    valid_output = False
                if valid_output:
                    status["slides"][str(path)] = {
                        "state": "complete",
                        "identity": output_identity,
                        "output": str(output_path),
                        "parents": parents,
                        "source_bytes": source_stat.st_size,
                        "source_mtime_ns_after": source_stat.st_mtime_ns,
                        "source_inode": source_stat.st_ino,
                        "recovered_from_output": True,
                    }
                else:
                    status["slides"][str(path)] = {
                        "state": "identity_conflict",
                        "identity": identity,
                        "output": str(output_path),
                        "error": "existing output failed source/identity/schema validation; refusing overwrite",
                    }
                continue
            if entry.get("state") == "complete" and entry.get("identity") in accepted_identities:
                status["slides"].pop(str(path), None)
            if entry.get("state") in {"error", "no_tissue", "identity_conflict"} and entry.get("identity") in accepted_identities:
                continue
            pending.append((path, expected, output_path))
        if not pending:
            if not args.watch:
                break
            all_downloaded = all(path.is_file() and path.stat().st_size == size for path, size in inventory.items())
            terminal = {"complete", "no_tissue", "error", "identity_conflict"}
            all_settled = all(status["slides"].get(str(path), {}).get("state") in terminal
                              and status["slides"].get(str(path), {}).get("identity") in accepted_identities for path in inventory)
            if all_downloaded and all_settled:
                failures = sum(status["slides"].get(str(path), {}).get("state") != "complete" for path in inventory)
                print(f"All BRACS inventory slides settled; {failures} did not produce features.", flush=True)
                if failures:
                    raise SystemExit(2)
                break
            print("No fully downloaded pending slides; watching for finalized SVS files.", flush=True)
            time.sleep(max(5, args.poll_seconds))
            continue
        for path, expected, output_path in pending:
            key = str(path)
            def update_progress(stage: str, done: int, total: int) -> None:
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                status["active"] = {
                    "slide": key,
                    "stage": stage,
                    "parents_done": done,
                    "parents_total": total,
                    "updated_at": now,
                }
                status["slides"][key] = {
                    **status["slides"].get(key, {}),
                    "state": "running",
                    "identity": identity,
                    "expected_bytes": expected,
                    "stage": stage,
                    "parents_done": done,
                    "parents_total": total,
                    "updated_at": now,
                }
                atomic_json(args.status, status)

            update_progress("starting", 0, 0)
            print(f"Starting {path}", flush=True)
            try:
                provenance = dict(workbook_rows.get(path.stem, {}))
                provenance.update({
                    "workbook": workbook_summary["workbook"],
                    "workbook_sha256": workbook_summary["workbook_sha256"],
                    "raw_partition": path.relative_to(args.source).parts[0],
                })
                result = extract_one(path, expected, output_path, model, preprocess, device, identity, provenance, update_progress,
                                     patch_workers=args.patch_workers)
                result["identity"] = identity
                status["slides"][key] = result
                if result["state"] == "complete":
                    done_this_run += 1
                    print(f"Complete {path}: parents={result['parents']} valid_children={result['valid_children']}", flush=True)
                else:
                    print(f"No tissue features for {path}: {result['reason']}", flush=True)
            except Exception as exc:
                status["slides"][key] = {
                    "state": "error",
                    "identity": identity,
                    "expected_bytes": expected,
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(f"Failed {path}: {type(exc).__name__}: {exc}", flush=True)
            atomic_json(args.status, status)
            status.pop("active", None)
            atomic_json(args.status, status)
            if args.max_slides and done_this_run >= args.max_slides:
                return
        if not args.watch:
            break
    atomic_json(args.status, status)


if __name__ == "__main__":
    main()
