"""
NeuroScan AI — 3-Class Brain MRI Classifier
ET · PD · Healthy  |  Swin-B / Swin-S  |  GradCAM + Attention Rollout
"""

import io
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from inference import (
    CLASS_NAMES, DEVICE, IMG_SIZE,
    generate_gradcam, generate_attention_rollout,
    load_model, predict,
)
from model_downloader import ensure_model_downloaded

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI · 3-Class MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg:       #0a0f1e;
    --surface:  #111827;
    --border:   #1e2d45;
    --accent:   #00c8a0;
    --accent2:  #3b82f6;
    --warning:  #f59e0b;
    --danger:   #f87171;
    --healthy:  #34d399;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --card-bg:  #131c2e;
}
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Navbar */
.navbar {
    display:flex; align-items:center; gap:14px;
    padding:0 0 2rem; border-bottom:1px solid var(--border);
    margin-bottom:2.5rem;
}
.navbar-icon {
    width:44px; height:44px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    border-radius:12px; display:flex; align-items:center;
    justify-content:center; font-size:22px;
}
.navbar-title { font-size:1.4rem; font-weight:700; letter-spacing:-0.02em; }
.navbar-sub   { font-size:0.78rem; color:var(--muted); font-family:'DM Mono',monospace; }

/* Cards */
.card {
    background:var(--card-bg); border:1px solid var(--border);
    border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;
    transition:border-color .2s;
}
.card:hover { border-color:var(--accent); }
.card-title {
    font-size:0.7rem; font-weight:600; letter-spacing:.12em;
    text-transform:uppercase; color:var(--muted);
    font-family:'DM Mono',monospace; margin-bottom:.8rem;
}

/* Result badges — one per class */
.result-badge {
    display:inline-block; padding:6px 18px; border-radius:999px;
    font-size:.85rem; font-weight:600; letter-spacing:.05em;
    font-family:'DM Mono',monospace; margin-bottom:.5rem;
}
.badge-et      { background:rgba(0,200,160,.12);  color:var(--accent);  border:1px solid var(--accent); }
.badge-pd      { background:rgba(248,113,113,.12); color:var(--danger);  border:1px solid var(--danger); }
.badge-healthy { background:rgba(52,211,153,.12);  color:var(--healthy); border:1px solid var(--healthy); }

/* Probability bars */
.conf-bar-wrap { margin:.6rem 0 .3rem; }
.conf-label {
    display:flex; justify-content:space-between;
    font-size:.76rem; color:var(--muted); margin-bottom:4px;
    font-family:'DM Mono',monospace;
}
.conf-track { height:8px; border-radius:99px; background:var(--border); overflow:hidden; }
.conf-fill  { height:100%; border-radius:99px; transition:width .6s cubic-bezier(.4,0,.2,1); }
.fill-et      { background:linear-gradient(90deg,var(--accent),#00e5b8); }
.fill-pd      { background:linear-gradient(90deg,var(--danger),#ff9a9a); }
.fill-healthy { background:linear-gradient(90deg,var(--healthy),#6ee7b7); }

/* Metric tiles */
.metric-row { display:flex; gap:1rem; margin:1rem 0; flex-wrap:wrap; }
.metric-tile {
    flex:1; min-width:90px; background:var(--surface);
    border:1px solid var(--border); border-radius:12px;
    padding:1rem 1.2rem; text-align:center;
}
.metric-val { font-size:1.4rem; font-weight:700; color:var(--accent); font-family:'DM Mono',monospace; }
.metric-key { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; margin-top:.2rem; }

/* Upload zone */
[data-testid="stFileUploader"] {
    border:2px dashed var(--border) !important;
    border-radius:16px !important; background:var(--card-bg) !important;
    transition:border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color:var(--accent) !important; }

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;
    color:#000 !important; font-weight:700 !important; border:none !important;
    border-radius:10px !important; padding:.6rem 1.8rem !important;
    font-family:'Sora',sans-serif !important; letter-spacing:.02em !important;
    transition:opacity .2s !important;
}
.stButton > button:hover { opacity:.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] {
    background:var(--surface) !important; border-radius:8px !important;
    border:1px solid var(--border) !important; color:var(--muted) !important;
    font-size:.8rem !important; font-family:'DM Mono',monospace !important;
}
.stTabs [aria-selected="true"] {
    background:var(--card-bg) !important; color:var(--accent) !important;
    border-color:var(--accent) !important;
}

/* Spinner */
.stSpinner > div { border-top-color:var(--accent) !important; }

/* Disclaimer */
.disclaimer {
    font-size:.72rem; color:var(--muted);
    border:1px solid var(--border); border-radius:10px;
    padding:.8rem 1rem; margin-top:2rem; line-height:1.6;
}
</style>
""", unsafe_allow_html=True)


# ─── Navbar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-icon">🧠</div>
  <div>
    <div class="navbar-title">NeuroScan AI</div>
    <div class="navbar-sub">MRI · 3-Class Classifier · Swin Transformer · GradCAM + Attention Rollout</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
CLASS_COLORS = {"ET": "et", "PD": "pd", "Healthy": "healthy"}
CLASS_FULL   = {
    "ET"     : "Essential Tremor",
    "PD"     : "Parkinson's Disease (tremor-dominant)",
    "Healthy": "Healthy / Control",
}
FILL_COLORS  = {"ET": "fill-et", "PD": "fill-pd", "Healthy": "fill-healthy"}


def prob_bars_html(probs: list[float]) -> str:
    rows = ""
    for name, prob in zip(CLASS_NAMES, probs):
        fill = FILL_COLORS[name]
        rows += f"""
        <div class='conf-bar-wrap'>
          <div class='conf-label'><span>{name} — {CLASS_FULL[name]}</span><span>{prob*100:.1f}%</span></div>
          <div class='conf-track'>
            <div class='conf-fill {fill}' style='width:{prob*100:.1f}%'></div>
          </div>
        </div>"""
    return rows


def make_heatmap_figure(img_array, heatmap, overlay, title: str) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#131c2e")

    img_norm = img_array.astype(np.float32) / 255.0

    panels = [
        ("Original",  img_norm,  "gray"),
        (title,       heatmap,   "jet"),
        ("Overlay",   overlay,   None),
    ]
    for ax, (t, data, cmap) in zip(axes, panels):
        ax.set_facecolor("#131c2e")
        if cmap == "gray":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        elif cmap == "jet":
            ax.imshow(data, cmap="jet",  vmin=0, vmax=1)
        else:
            ax.imshow(data)
        ax.set_title(t, color="#94a3b8", fontsize=10, pad=8)
        ax.axis("off")

    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131c2e")
    buf.seek(0); plt.close()
    return buf


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_variant = st.selectbox(
        "Model variant",
        options=["swin_b", "swin_s"],
        index=0,
        help="swin_b = 88M params (higher accuracy) · swin_s = 49M params (faster, ~170 MB)"
    )

    use_tta = st.toggle("Test-Time Augmentation (TTA)", value=True,
                        help="Averages original + horizontal-flip predictions. Adds ~20ms.")

    show_probs    = st.toggle("Show class probabilities",   value=True)
    show_gradcam  = st.toggle("Show GradCAM heatmap",       value=True)
    show_rollout  = st.toggle("Show Attention Rollout",      value=False)
    apply_mask    = st.toggle("Apply brain mask to heatmap", value=True)

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.markdown(f"""
    <div style='font-size:.78rem; color:#64748b; line-height:1.9; font-family:"DM Mono",monospace;'>
    Backbone &nbsp;&nbsp; {model_variant}<br>
    Input &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 224 × 224 px<br>
    Classes &nbsp;&nbsp;&nbsp; ET · PD · Healthy<br>
    Training &nbsp;&nbsp; 3-phase fine-tune<br>
    XAI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GradCAM + Attn Rollout<br>
    Device &nbsp;&nbsp;&nbsp;&nbsp; {str(DEVICE).upper()}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ Class Legend")
    st.markdown("""
    <span class='result-badge badge-et'>ET</span> &nbsp; Essential Tremor<br><br>
    <span class='result-badge badge-pd'>PD</span> &nbsp; Parkinson's Disease<br><br>
    <span class='result-badge badge-healthy'>Healthy</span> &nbsp; Control / No pathology
    """, unsafe_allow_html=True)


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(variant: str):
    model_path = ensure_model_downloaded()
    return load_model(str(model_path), variant=variant)


with st.spinner("Loading model…"):
    try:
        model, meta = get_model(model_variant)
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"⚠️ Could not load model: `{e}`")


# ─── Main layout ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.1], gap="large")

# ── Upload panel ──────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="card-title">Upload MRI Slice</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("L")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Uploaded Slice</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, clamp=True)
        st.markdown(f"""
        <div style='font-size:.72rem; color:#64748b; font-family:"DM Mono",monospace; margin-top:.5rem;'>
        {uploaded.name} &nbsp;·&nbsp; {image.size[0]}×{image.size[1]} px &nbsp;·&nbsp;
        {uploaded.size/1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("🔍 Run Classification", use_container_width=True, disabled=not model_loaded)
    else:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 1rem; color:#64748b;'>
            <div style='font-size:2.5rem; margin-bottom:1rem;'>🫧</div>
            <div style='font-weight:600; margin-bottom:.4rem;'>Drop an MRI slice here</div>
            <div style='font-size:.78rem;'>PNG · JPG · TIFF · Single 2D axial slice</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False


# ── Results panel ─────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-title">Classification Result</div>', unsafe_allow_html=True)

    if uploaded and run_btn and model_loaded:
        img_array = np.array(image).astype(np.float32)

        # ── Inference ────────────────────────────────────────────────────────
        with st.spinner("Running inference…"):
            t0    = time.perf_counter()
            probs = predict(model, image, use_tta=use_tta)
            latency_ms = (time.perf_counter() - t0) * 1000

        pred_idx   = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        badge_cls  = CLASS_COLORS[pred_label]

        # ── Result badge ─────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Prediction</div>
          <span class='result-badge badge-{badge_cls}'>{pred_label}</span>
          <div style='font-size:1rem; font-weight:600; margin-top:.6rem;'>
              {CLASS_FULL[pred_label]}
          </div>
          <div style='font-size:.8rem; color:#64748b; margin-top:.2rem;'>
              Confidence: {confidence*100:.1f}%
              {'&nbsp;·&nbsp; TTA active' if use_tta else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──────────────────────────────────────────────────
        if show_probs:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Class Probabilities</div>', unsafe_allow_html=True)
            st.markdown(prob_bars_html(probs.tolist()), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────────────────────────────────
        et_p, pd_p, h_p = probs[0]*100, probs[1]*100, probs[2]*100
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-tile'>
            <div class='metric-val'>{et_p:.1f}%</div>
            <div class='metric-key'>ET prob</div>
          </div>
          <div class='metric-tile'>
            <div class='metric-val'>{pd_p:.1f}%</div>
            <div class='metric-key'>PD prob</div>
          </div>
          <div class='metric-tile'>
            <div class='metric-val'>{h_p:.1f}%</div>
            <div class='metric-key'>Healthy prob</div>
          </div>
          <div class='metric-tile'>
            <div class='metric-val'>{latency_ms:.0f}ms</div>
            <div class='metric-key'>Inference</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── XAI tabs ─────────────────────────────────────────────────────────
        xai_tabs = []
        if show_gradcam:  xai_tabs.append("🔥 GradCAM")
        if show_rollout:  xai_tabs.append("🔵 Attention Rollout")

        if xai_tabs:
            tabs = st.tabs(xai_tabs)
            tab_idx = 0

            if show_gradcam:
                with tabs[tab_idx]:
                    with st.spinner("Generating GradCAM…"):
                        try:
                            heatmap, overlay = generate_gradcam(
                                model, img_array, class_idx=pred_idx,
                                apply_brain_mask=apply_mask,
                            )
                            buf = make_heatmap_figure(img_array, heatmap, overlay, "GradCAM")
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<div class="card-title">GradCAM — Predicted Class Activation</div>', unsafe_allow_html=True)
                            st.image(buf, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"GradCAM skipped: {e}")
                tab_idx += 1

            if show_rollout:
                with tabs[tab_idx]:
                    with st.spinner("Computing Attention Rollout…"):
                        try:
                            rollout = generate_attention_rollout(model, img_array)
                            img_rgb = np.stack([img_array.astype(np.uint8)] * 3, axis=-1)
                            from inference import overlay_heatmap
                            ov = overlay_heatmap(img_rgb, rollout)
                            buf = make_heatmap_figure(img_array, rollout, ov, "Attention Rollout")
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<div class="card-title">Attention Rollout — Swin Layer Importance</div>', unsafe_allow_html=True)
                            st.image(buf, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Attention Rollout skipped: {e}")

    elif not uploaded:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 1rem; color:#64748b;'>
          <div style='font-size:2rem; margin-bottom:1rem;'>⬅️</div>
          <div style='font-weight:600;'>Upload an MRI slice to begin</div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded and not run_btn:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 1rem; color:#64748b;'>
          <div style='font-size:2rem; margin-bottom:1rem;'>🔍</div>
          <div style='font-weight:600;'>Click "Run Classification" to analyse</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='disclaimer'>
⚠️ <strong>Research use only.</strong> This tool is a thesis prototype and is not a validated
clinical diagnostic device. Results must not be used as a substitute for professional medical
evaluation. Always consult a qualified neurologist for diagnosis and treatment decisions.
</div>
""", unsafe_allow_html=True)