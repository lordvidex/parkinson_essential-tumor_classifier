"""
model_downloader.py — Downloads model weights from Google Drive.

HOW TO GET YOUR FILE ID:
  1. Upload your .pth to Google Drive
  2. Right-click → Share → Anyone with the link → Copy link
  3. The link looks like:
       https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9/view?usp=sharing
                                        ^^^^^^^^^^^^^^^^^ this is FILE_ID
  4. Paste FILE_ID into .streamlit/secrets.toml  (see README.md)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests
import streamlit as st

# ── Cache path inside the container / local ────────────────────────────────
MODEL_CACHE_DIR  = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/neuroscan_models"))
MODEL_FILENAME   = "swin_classifier.pth"
MODEL_LOCAL_PATH = MODEL_CACHE_DIR / MODEL_FILENAME

CHUNK_SIZE_MB = 8   # download in 8 MB chunks


# ── Google Drive direct-download URL builder ───────────────────────────────

def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _confirm_token(session: requests.Session, url: str, file_id: str) -> str:
    """
    Google Drive redirects large files through a virus-scan confirmation page.
    This extracts the confirmation token so we get the actual file.
    """
    r = session.get(url, stream=True, timeout=30)
    # Look for confirmation token in cookies
    for key, val in r.cookies.items():
        if key.startswith("download_warning"):
            return f"{url}&confirm={val}"
    # Fallback: search HTML for token
    content = b""
    for chunk in r.iter_content(chunk_size=32768):
        content += chunk
        if len(content) > 1_000_000:
            break
    match = re.search(rb'confirm=([0-9A-Za-z_\-]+)', content)
    if match:
        token = match.group(1).decode()
        return f"{url}&confirm={token}"
    # Newer Google Drive: uses uuid confirmation
    match = re.search(rb'"downloadUrl":"(https://[^"]+)"', content)
    if match:
        return match.group(1).decode().replace(r"\u003d", "=").replace(r"\u0026", "&")
    return url   # hope for the best


import gdown

def download_from_gdrive(file_id: str, save_path: Path, progress_bar=None):
    url = f"https://drive.google.com/uc?id={file_id}"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and save_path.stat().st_size > 1_000_000:
        return save_path

    gdown.download(url, str(save_path), quiet=False)

    # validate
    if save_path.stat().st_size < 1_000_000:
        raise ValueError("Downloaded file too small — likely failed")

    return save_path


# ── Streamlit-aware loader (call this from app.py) ─────────────────────────

def ensure_model_downloaded() -> Path:
    """
    Called once at app startup via @st.cache_resource.

    Reads GDRIVE_FILE_ID from st.secrets (set in .streamlit/secrets.toml).
    Shows a progress bar in the sidebar while downloading.
    Returns the local path to the .pth file.
    """
    if MODEL_LOCAL_PATH.exists() and MODEL_LOCAL_PATH.stat().st_size > 1_000_000:
        return MODEL_LOCAL_PATH

    try:
        file_id = st.secrets["GDRIVE_FILE_ID"]
    except (KeyError, FileNotFoundError):
        st.error(
            "**Missing `GDRIVE_FILE_ID` in secrets.**\n\n"
            "Add it to `.streamlit/secrets.toml`:\n"
            "```toml\nGDRIVE_FILE_ID = \"your_file_id_here\"\n```\n\n"
            "See README.md for how to get the file ID from your Drive share link."
        )
        st.stop()

    st.info("⬇️ Downloading model weights from Google Drive (one-time ~300 MB)…")
    bar = st.progress(0.0)

    try:
        path = download_from_gdrive(file_id, MODEL_LOCAL_PATH, progress_bar=bar)
        bar.progress(1.0)
        return path
    except Exception as e:
        st.error(f"Download failed: `{e}`\n\nCheck that your Drive file is shared with **Anyone with the link**.")
        st.stop()