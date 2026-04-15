"""
Script version of the latest yadisk_dicom_to_png.ipynb.
Adds CLI params for offset/limit/max-files/metadata-path while preserving notebook defaults.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time
import datetime
import tempfile
import posixpath
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any, List, Tuple
import shutil

import yadisk
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image

# --------- Settings (match notebook) ---------
YA_TOKEN = os.getenv("YA_TOKEN")

# source
SOURCE_DISK_ROOT = None  # '/Users/oovamoyo/Downloads/MRT'
SOURCE_PUBLIC_URL = os.getenv("SOURCE_PUBLIC_URL")

# destination
DEST_DISK_ROOT = "Загрузки/MRT_PNGs"
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
DEST_LOCAL_ROOT = os.path.join(ARTIFACTS_DIR, DEST_DISK_ROOT)
UPLOAD_ARTIFACTS_FROM_LOCAL = False     # False = do NOT upload after conversion
DELETE_LOCAL_AFTER_UPLOAD = False       # True = cleanup local artifacts after upload

# script settings
SKIP_IF_PNG_EXISTS = True
PATIENTS_OFFSET = 0
PATIENTS_LIMIT = 2  # set None for full run
UPLOAD_METADATA_EVERY = None  # upload only at the end
LOG_TIMING = False

# Image conversion settings
CLIP_PERCENTILES = (1, 99)
OUTPUT_MODE = "L"
PNG_COMPRESS_LEVEL = 3

# --------- Clients ---------
y_auth = yadisk.Client(token=YA_TOKEN)
y_public = yadisk.Client()

if y_auth is None:
    raise ValueError("set YA_TOKEN in env variable to upload PNGs to Disk")

# --------- Data structures ---------
@dataclass
class SourceFile:
    src_mode: str  # disk or public
    src_path: str  # for disk, absolute disk path, for public, rel path within public folder
    rel_path: str  # rel path under source root
    name: str

# --------- Helpers ---------
def maybe_dicom(fname: str, skipdicomdir: bool = True) -> bool:
    name_lower = fname.lower()
    if name_lower == "dicomdir" and skipdicomdir:
        return False
    return fname.startswith("IM") or name_lower.endswith(".dcm") or ("." not in fname)


def ensure_remote_dir(client: yadisk.Client, remote_dir: str):
    remote_dir = remote_dir.rstrip("/") or "/"
    if remote_dir == "/":
        return

    try:
        client.makedirs(remote_dir)
        return
    except Exception:
        pass

    parts = [p for p in remote_dir.split("/") if p]
    cur = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            if not client.exists(cur):
                client.mkdir(cur)
        except yadisk.exceptions.PathExistsError:
            print("cannot make directories for", remote_dir)
            pass


def list_patient_dirs() -> List[yadisk.objects.SyncPublicResourceObject]:
    out = []
    for item in y_public.public_listdir(SOURCE_PUBLIC_URL, sort="name"):
        if item.type == "dir":
            out.append(item)
    return out


def iter_disk_files(local_root: str) -> Iterator[Tuple[str, List[SourceFile]]]:
    """Yield (patient_folder, [SourceFile, ...]) from a local folder where top-level dirs are patients."""
    root_norm = os.path.abspath(local_root)
    dirs = [d for d in os.listdir(root_norm)]
    dirs.sort()
    if PATIENTS_OFFSET:
        dirs = dirs[PATIENTS_OFFSET:]
    if PATIENTS_LIMIT:
        dirs = dirs[:PATIENTS_LIMIT]
    for patient in dirs:
        patient_path = os.path.join(root_norm, patient)
        if not os.path.isdir(patient_path):
            continue
        files: List[SourceFile] = []
        for dirpath, _, filenames in os.walk(patient_path):
            for filename in filenames:
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, root_norm).replace(os.sep, "/")
                if not maybe_dicom(filename):
                    continue
                files.append(SourceFile(src_mode="disk", src_path=abs_path, rel_path=rel_path, name=filename))
        files.sort(key=lambda s: s.rel_path)
        yield patient, files


def iter_public_files(client: yadisk.Client, public_url: str) -> Iterator[Tuple[str, List[SourceFile]]]:
    """Yield (patient_folder, [SourceFile, ...]) from a public folder where top-level dirs are patients."""

    for item in client.public_listdir(public_url, path=None, sort="name", max_items=PATIENTS_LIMIT, offset=PATIENTS_OFFSET):
        if item.type != "dir":
            continue
        patient = item.name
        files: List[SourceFile] = []
        stack = [item.path]
        while stack:
            cur_rel = stack.pop()
            for child in client.public_listdir(public_url, path=cur_rel):
                if child.type == "dir":
                    stack.append(child.path)
                elif not maybe_dicom(child.name):
                    continue
                else:
                    files.append(SourceFile(src_mode="public", src_path=child.path, rel_path=child.path, name=child.name))
        yield patient, files


def iter_source_files() -> Iterator[Tuple[str, List[SourceFile]]]:
    """Yield (patient_folder, [SourceFile, ...]) using disk or public source."""
    if SOURCE_DISK_ROOT:
        yield from iter_disk_files(SOURCE_DISK_ROOT)
    elif SOURCE_PUBLIC_URL:
        yield from iter_public_files(y_public, SOURCE_PUBLIC_URL)
    else:
        raise ValueError("Set either SOURCE_DISK_ROOT or SOURCE_PUBLIC_URL")


def _get_first_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return float(x[0])
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def dicom_to_png_bytes(dicom_bytes: bytes) -> Tuple[List[Tuple[str, bytes]], Dict[str, Any]]:
    """Returns ([(suffix, png_bytes), ...], meta). suffix is '' or '_f000' etc."""
    bio = io.BytesIO(dicom_bytes)
    ds = pydicom.dcmread(bio, force=True)

    meta: Dict[str, Any] = {}
    meta["study_instance_uid"] = getattr(ds, "StudyInstanceUID", None)
    meta["series_instance_uid"] = getattr(ds, "SeriesInstanceUID", None)
    meta["sop_instance_uid"] = getattr(ds, "SOPInstanceUID", None)
    meta["modality"] = getattr(ds, "Modality", None)
    meta["series_description"] = getattr(ds, "SeriesDescription", None)
    meta["instance_number"] = getattr(ds, "InstanceNumber", None)
    meta["acquisition_number"] = getattr(ds, "AcquisitionNumber", None)

    meta["rows"] = getattr(ds, "Rows", None)
    meta["cols"] = getattr(ds, "Columns", None)
    meta["pixel_spacing"] = list(getattr(ds, "PixelSpacing", [])) if hasattr(ds, "PixelSpacing") else None
    meta["slice_thickness"] = _get_first_number(getattr(ds, "SliceThickness", None))
    meta["slice_location"] = _get_first_number(getattr(ds, "SliceLocation", None))
    meta["image_position_patient"] = list(getattr(ds, "ImagePositionPatient", [])) if hasattr(ds, "ImagePositionPatient") else None

    meta["bits_allocated"] = getattr(ds, "BitsAllocated", None)
    meta["photometric_interpretation"] = getattr(ds, "PhotometricInterpretation", None)
    meta["transfer_syntax_uid"] = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)

    if not hasattr(ds, "PixelData"):
        raise ValueError("No PixelData in DICOM")

    arr = ds.pixel_array.astype(np.float32)

    slope = _get_first_number(getattr(ds, "RescaleSlope", 1.0)) or 1.0
    intercept = _get_first_number(getattr(ds, "RescaleIntercept", 0.0)) or 0.0
    arr = arr * slope + intercept

    wc = _get_first_number(getattr(ds, "WindowCenter", None))
    ww = _get_first_number(getattr(ds, "WindowWidth", None))

    def normalize_to_u8(x: np.ndarray) -> np.ndarray:
        if wc is not None and ww is not None and ww > 0:
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
            x = np.clip(x, lo, hi)
        else:
            p_lo, p_hi = np.percentile(x, CLIP_PERCENTILES)
            if p_hi <= p_lo:
                p_lo, p_hi = float(np.min(x)), float(np.max(x))
            x = np.clip(x, p_lo, p_hi)
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        return (x * 255.0).round().astype(np.uint8)

    pngs: List[Tuple[str, bytes]] = []
    if arr.ndim == 2:
        u8 = normalize_to_u8(arr)
        im = Image.fromarray(u8, mode=OUTPUT_MODE)
        out = io.BytesIO()
        im.save(out, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        pngs.append(("", out.getvalue()))
    elif arr.ndim == 3:
        for i in range(arr.shape[0]):
            u8 = normalize_to_u8(arr[i])
            im = Image.fromarray(u8, mode=OUTPUT_MODE)
            out = io.BytesIO()
            im.save(out, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
            pngs.append((f"_f{i:03d}", out.getvalue()))
        meta["number_of_frames"] = arr.shape[0]
    else:
        raise ValueError(f"Unsupported pixel_array ndim={arr.ndim}")

    return pngs, meta


# Download + upload helpers
def download_source_bytes(sf: SourceFile, pc: yadisk.Client = y_public) -> bytes:
    if sf.src_mode == "disk":
        with open(sf.src_path, "rb") as f:
            return f.read()
    elif sf.src_mode == "public":
        out = io.BytesIO()
        pc.download_public(SOURCE_PUBLIC_URL, out, path=sf.src_path)
        return out.getvalue()
    else:
        raise ValueError(f"Unsupported src_mode: {sf.src_mode}")


def save_png_locally(png_bytes: bytes, dest_path: str) -> str:
    # dest_path is a posix path under DEST_LOCAL_ROOT that ends with .png
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(png_bytes)
    return dest_path


def upload_artifacts_from_local() -> None:
    """
    Upload everything under DEST_LOCAL_ROOT to DEST_DISK_ROOT in Yandex.Disk.
    This includes metadata.csv and all patient PNG subfolders.
    """
    print("uploading artifacts from local root")
    for dirpath, _, filenames in os.walk(DEST_LOCAL_ROOT):
        rel_dir = os.path.relpath(dirpath, DEST_LOCAL_ROOT)
        # map local dir -> remote dir
        remote_dir = DEST_DISK_ROOT.rstrip("/")
        if rel_dir not in (".", ""):
            remote_dir = posixpath.join(remote_dir, rel_dir.replace(os.sep, "/"))
        ensure_remote_dir(y_auth, remote_dir)

        for filename in filenames:
            local_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(local_path, DEST_LOCAL_ROOT).replace(os.sep, "/")
            remote_path = posixpath.join(DEST_DISK_ROOT.rstrip("/"), rel_path)
            y_auth.upload(local_path, remote_path, overwrite=True)

    print("uploading artifacts from local root complete!")


def delete_local_artifacts() -> None:
    print("Deleting local artifacts...")
    for patient in os.listdir(DEST_LOCAL_ROOT):
        patient_path = os.path.join(DEST_LOCAL_ROOT, patient)
        if not os.path.isdir(patient_path):
            continue
        shutil.rmtree(patient_path, ignore_errors=True)
    print("Local artifacts deleted.")


# Metadata CSV writer
os.makedirs(DEST_LOCAL_ROOT, exist_ok=True)

METADATA_LOCAL_PATH = os.path.join(DEST_LOCAL_ROOT, "metadata.csv")
METADATA_REMOTE_PATH = posixpath.join(DEST_DISK_ROOT.rstrip("/"), "metadata.csv")

CSV_FIELDS = [
    "processed_at",
    "status",
    "error",
    "src_mode",
    "src_rel_path",
    "src_path",
    "dest_png_path",
    "frame_index",
    "patient_folder",
    "study_instance_uid",
    "series_instance_uid",
    "sop_instance_uid",
    "modality",
    "series_description",
    "instance_number",
    "acquisition_number",
    "rows",
    "cols",
    "pixel_spacing",
    "slice_thickness",
    "slice_location",
    "image_position_patient",
    "bits_allocated",
    "photometric_interpretation",
    "transfer_syntax_uid",
    "number_of_frames",
]


def init_metadata_csv(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
    print(f'initialized metadata csv at {path}')


def append_metadata_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            safe_row = {k: row.get(k, None) for k in CSV_FIELDS}
            w.writerow(safe_row)

def dest_png_path_from_rel(rel_path: str, frame_suffix: str = "") -> Tuple[str, str]:
    """Returns full dest path for local and remote."""
    rel_dir = posixpath.dirname(rel_path)
    name = posixpath.basename(rel_path)
    base, ext = os.path.splitext(name)
    out_name = f"{base}{frame_suffix}.png" if ext else f"{name}{frame_suffix}.png"
    if rel_dir:
        return (
            os.path.join(DEST_LOCAL_ROOT, rel_dir.lstrip("/"), out_name),
            posixpath.join(DEST_DISK_ROOT.rstrip("/"), rel_dir.lstrip("/"), out_name),
        )
    return (
        os.path.join(DEST_LOCAL_ROOT, out_name),
        posixpath.join(DEST_DISK_ROOT.rstrip("/"), out_name),
    )


def get_patient_folder_from_rel(rel_path: str) -> str:
    return rel_path.split("/", 1)[0] if "/" in rel_path else rel_path


def process_all(max_files: Optional[int] = None):
    processed_rows = 0
    processed_files = 0
    processed_patients = 0
    skipped_files = 0
    errors = 0
    stop_processing = False

    for patient_folder, files in tqdm(iter_source_files(), desc="Patients"):
        print(f"Processing patient folder: {patient_folder} with {len(files)} files")
        rows = []
        for sf in tqdm(files, desc=f"Files ({patient_folder})", total=len(files), leave=False, position=1):
            if max_files is not None and processed_files >= max_files:
                stop_processing = True
                break
            try:
                if SKIP_IF_PNG_EXISTS:
                    dest_path_local, _ = dest_png_path_from_rel(sf.rel_path, frame_suffix="")
                    if os.path.exists(dest_path_local):
                        skipped_files += 1
                        continue
                t0 = time.time()
                dicom_bytes = download_source_bytes(sf)
                t1 = time.time()
                pngs, dicom_meta = dicom_to_png_bytes(dicom_bytes)
                t2 = time.time()
                save_time = 0.0
                for frame_i, (suffix, png_bytes) in enumerate(pngs):
                    dest_path_local, _ = dest_png_path_from_rel(sf.rel_path, frame_suffix=suffix)
                    s = time.time()
                    save_png_locally(png_bytes, dest_path_local)
                    save_time += time.time() - s
                    rows.append(
                        {
                            "processed_at": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                            "status": "OK",
                            "error": None,
                            "src_mode": sf.src_mode,
                            "src_rel_path": sf.rel_path,
                            "src_path": sf.src_path,
                            "dest_png_path": dest_path_local,
                            "frame_index": frame_i if len(pngs) > 1 else 0,
                            "patient_folder": patient_folder,
                            **dicom_meta,
                        }
                    )

                    processed_rows += 1
                processed_files += 1
                if LOG_TIMING:
                    t3 = time.time()
                    print(
                        "[timing] %s: download=%.3fs convert=%.3fs save=%.3fs total=%.3fs"
                        % (sf.rel_path, t1 - t0, t2 - t1, save_time, t3 - t0)
                    )
            except InvalidDicomError:
                continue
            except Exception as e:
                errors += 1
                rows.append(
                    {
                        "processed_at": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                        "status": "ERROR",
                        "error": repr(e)[:2000],
                        "src_mode": sf.src_mode,
                        "src_rel_path": sf.rel_path,
                        "src_path": sf.src_path,
                        "dest_png_path": "",
                        "frame_index": "",
                        "patient_folder": patient_folder,
                    }
                )
        processed_patients += 1
        append_metadata_rows(METADATA_LOCAL_PATH, rows)
        if stop_processing:
            break
    print("Done.")
    print(
        {
            "processed_patients": processed_patients,
            "processed_files": processed_files,
            "processed_rows": processed_rows,
            "skipped_files": skipped_files,
            "errors": errors,
            "dest_root": DEST_DISK_ROOT,
            "metadata_remote": METADATA_REMOTE_PATH,
        }
    )


def main():
    global PATIENTS_OFFSET, PATIENTS_LIMIT, METADATA_LOCAL_PATH

    parser = argparse.ArgumentParser(description="Run yadisk_dicom_to_png (script version of notebook).")
    parser.add_argument("--offset", type=int, default=PATIENTS_OFFSET, help="Patient offset")
    parser.add_argument("--limit", type=int, default=PATIENTS_LIMIT if PATIENTS_LIMIT is not None else -1, help="Patient limit (-1 for all)")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most this many files")
    parser.add_argument("--metadata-path", default=METADATA_LOCAL_PATH, help="Metadata CSV path (single file)")
    args = parser.parse_args()

    PATIENTS_OFFSET = args.offset
    PATIENTS_LIMIT = None if args.limit is not None and args.limit < 0 else args.limit
    METADATA_LOCAL_PATH = args.metadata_path

    if not YA_TOKEN:
        raise SystemExit("Set YA_TOKEN in environment")

    os.makedirs(os.path.dirname(METADATA_LOCAL_PATH), exist_ok=True)
    init_metadata_csv(METADATA_LOCAL_PATH)
    process_all(max_files=args.max_files)


if __name__ == "__main__":
    main()
