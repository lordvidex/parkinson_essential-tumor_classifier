# 🧠 NeuroScan AI — 3-Class Brain MRI Classifier

> **Research prototype · Not for clinical use**

Streamlit deployment of a Swin Transformer (Swin-B or Swin-S) fine-tuned to classify brain MRI slices into three classes: **ET** (Essential Tremor), **PD** (Parkinson's Disease), and **Healthy**. Includes GradCAM and Attention Rollout explainability maps, brain masking, and optional TTA.

---

## 📁 File Structure

```
├── app.py                        # Streamlit UI
├── inference.py                  # Model definition, prediction, GradCAM, Rollout
├── model_downloader.py           # Auto-downloads weights from Google Drive
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── secrets.toml              # ← YOU CREATE THIS (never commit to git)
└── README.md
```

---

## 🚀 Deploying to Streamlit Community Cloud (free tier)

### Step 1 — Host your model on Google Drive

Because the model checkpoint is ~300 MB, it cannot be committed to GitHub.  
The app downloads it automatically at startup from Google Drive.

1. Upload your `.pt` checkpoint to Google Drive.
2. Right-click the file → **Share** → **Anyone with the link** → **Viewer** → **Copy link**.
3. The share link looks like:
   ```
   https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing
                                    ^^^^^^^^^^^^^^^^^^^^
                                    this is your FILE_ID
   ```
4. Copy that `FILE_ID` — you'll need it in Step 3.

> **Important:** the file must be shared as *"Anyone with the link"*. A private file will cause the download to fail.

---

### Step 2 — Push the code to GitHub

```bash
# Create a new repo (private is fine)
git init
git add app.py inference.py model_downloader.py requirements.txt README.md
# Do NOT add .streamlit/secrets.toml — it contains your file ID
echo ".streamlit/secrets.toml" >> .gitignore
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

### Step 3 — Add your secret on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
2. Connect your GitHub repo, set the main file to `app.py`.
3. Before deploying, click **Advanced settings → Secrets** and paste:

```toml
GDRIVE_FILE_ID = "1A2B3C4D5E6F7G8H9I0J"
```

Replace the value with the `FILE_ID` you copied in Step 1.

4. Click **Deploy**. On the very first boot the app will download the weights (~300 MB) into `/tmp/neuroscan_models/` and cache them for the rest of the session.

---

### Step 4 — Choose the right variant

| Variant | Parameters | Checkpoint size | Speed (CPU) |
|---------|-----------|-----------------|-------------|
| `swin_b` | 88 M | ~340 MB | ~1.2 s/slice |
| `swin_s` | 49 M | ~195 MB | ~0.7 s/slice |

Select the variant in the sidebar. Make sure it matches the checkpoint you uploaded — loading a Swin-S checkpoint with the `swin_b` variant (or vice versa) will raise a `RuntimeError`.

---

## 💻 Running Locally

```bash
# 1. Clone / navigate to the project folder
cd neuroscan_app

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the secrets file
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE"
EOF

# 5. Run
streamlit run app.py
```

The model will be downloaded to `/tmp/neuroscan_models/swin_classifier.pth` on first run and reused on subsequent runs (as long as `/tmp` is not cleared).

---

## 🔑 Alternative: Skip Google Drive (local checkpoint)

If you are running locally and already have the `.pt` file on disk, you can bypass the downloader entirely. Open `model_downloader.py` and replace `ensure_model_downloaded()` with a hard-coded path:

```python
# model_downloader.py — local override
from pathlib import Path

def ensure_model_downloaded() -> Path:
    return Path("/path/to/your/best_swin_b.pt")
```

---

## 🌐 Alternative: Host on Hugging Face Hub

Hugging Face Hub is another good option for large model files and has no download quotas.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_HF_USERNAME/neuroscan-mri best_swin_b.pt
```

Then update `model_downloader.py`:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path

def ensure_model_downloaded() -> Path:
    local = hf_hub_download(
        repo_id="YOUR_HF_USERNAME/neuroscan-mri",
        filename="best_swin_b.pt",
        cache_dir="/tmp/neuroscan_models",
    )
    return Path(local)
```

Add your HF token to `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

And pass it as `token=st.secrets["HF_TOKEN"]` in the `hf_hub_download` call.

---

## ⚙️ App Features

| Feature | Description |
|---------|-------------|
| **3-class output** | ET · PD · Healthy with per-class probability bars |
| **TTA** | Averages original + horizontal-flip predictions (~+1% AUC, +20 ms) |
| **GradCAM** | Class activation map anchored to the Swin `norm` layer |
| **Attention Rollout** | Aggregated activation magnitude across all SwinTransformerBlocks |
| **Brain masking** | Otsu-threshold mask suppresses background noise in heatmaps |
| **Swin-B / Swin-S** | Switchable backbone via sidebar — no code change needed |
| **Auto-download** | Weights fetched from Google Drive at startup, cached in `/tmp` |

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: Checkpoint not found` | Check `GDRIVE_FILE_ID` in secrets and that the Drive file is public |
| `RuntimeError: Error(s) in loading state_dict` | Variant mismatch — ensure sidebar variant matches the uploaded checkpoint |
| `Download failed: 403` | Google Drive file is not shared with "Anyone with the link" |
| `CUDA out of memory` | App runs on CPU by default; GPU not required |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python-headless` |
| Slow first load | Normal — model downloads once (~300 MB). Subsequent loads use the cache. |

---

## ⚠️ Disclaimer

This tool is a **research prototype** developed as part of a thesis project. It is **not** a validated clinical diagnostic device and must **not** be used as a substitute for professional medical evaluation. Always consult a qualified neurologist for diagnosis and treatment decisions.
