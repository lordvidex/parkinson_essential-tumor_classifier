# NeuroScan AI — MRI Classifier · Streamlit Deployment

Binary classification of MRI PNG slices: **Essential Tremor (ET) vs Parkinson's Disease (PD)**  
Built with MONAI · EfficientNet-B0 · Streamlit

---

## File Structure

```
streamlit_app/
├── app.py                  # Streamlit UI
├── inference.py            # Model loading, prediction, GradCAM
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme config
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your model checkpoint

Copy your trained checkpoint into the same folder as `app.py`:

```bash
cp /path/to/monai_efficientnet_b0_final.pth ./
```

> The app looks for this filename by default. You can change it in the sidebar.

### 3. Run locally

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, set `app.py` as the entry point
4. Add your model file via **Secrets** or upload it to the repo

> ⚠️ The `.pth` file may be large (>25 MB). Use [Git LFS](https://git-lfs.com/) or host it on
> Hugging Face Hub and download at startup (see below).

### Optional: Download model from Hugging Face Hub at startup

Add this to the top of `app.py` if you host the checkpoint on HF Hub:

```python
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id  = "your-username/your-model-repo",
    filename = "monai_efficientnet_b0_final.pth",
)
```

Then install: `pip install huggingface_hub`

---

## Checkpoint Formats Supported

The `load_model()` function in `inference.py` handles both formats from the training notebook:

| Format | Description |
|--------|-------------|
| **Full dict** | `torch.save({"model_state_dict": ..., "test_auc": ..., ...})` — from the training notebook |
| **State dict** | `torch.save(model.state_dict(), ...)` — raw weights only |

---

## Sidebar Options

| Option | Default | Description |
|--------|---------|-------------|
| Model checkpoint | `monai_efficientnet_b0_final.pth` | Path to `.pth` file |
| Classification threshold | `0.5` | PD probability cutoff |
| Show GradCAM | `On` | Attention heatmap overlay |
| Show raw probabilities | `On` | Confidence bars for both classes |

---

## GradCAM Notes

GradCAM targets the **last convolutional block** of EfficientNet-B0 (`model._blocks[-1]`).
It uses a zero-dependency hook implementation — no `grad-cam` library required in deployment.

---

## Disclaimer

> This tool is a **research prototype** for a master's thesis.  
> It is **not** a validated clinical diagnostic device.  
> Results must not substitute professional neurological evaluation.
