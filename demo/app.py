"""
MRI Classifier — ET vs Parkinson's Disease
Streamlit deployment for MONAI EfficientNet-B0
"""

import streamlit as st
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

from inference import load_model, predict, generate_gradcam

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI · MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0f1e;
    --surface:   #111827;
    --border:    #1e2d45;
    --accent:    #00c8a0;
    --accent2:   #3b82f6;
    --danger:    #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --card-bg:   #131c2e;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Navbar / header ── */
.navbar {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 2rem 0; border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.navbar-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.navbar-title  { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
.navbar-sub    { font-size: 0.78rem; color: var(--muted); font-family: 'DM Mono', monospace; }

/* ── Cards ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--accent); }
.card-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
    font-family: 'DM Mono', monospace; margin-bottom: 0.8rem;
}

/* ── Result badge ── */
.result-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.5rem;
}
.badge-et { background: rgba(0, 200, 160, 0.12); color: var(--accent); border: 1px solid var(--accent); }
.badge-pd { background: rgba(248, 113, 113, 0.12); color: var(--danger);  border: 1px solid var(--danger); }

/* ── Confidence bar ── */
.conf-bar-wrap { margin: 0.6rem 0 0.3rem; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.76rem; color: var(--muted); margin-bottom: 4px;
    font-family: 'DM Mono', monospace;
}
.conf-track {
    height: 8px; border-radius: 99px;
    background: var(--border);
    overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.fill-et { background: linear-gradient(90deg, var(--accent), #00e5b8); }
.fill-pd { background: linear-gradient(90deg, var(--danger), #ff9a9a); }

/* ── Metric tile ── */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
.metric-tile {
    flex: 1; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700; color: var(--accent); font-family: 'DM Mono', monospace; }
.metric-key { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    background: var(--card-bg) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    font-family: 'Sora', sans-serif !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px !important; border: 1px solid var(--border) !important; }

/* ── Disclaimer ── */
.disclaimer {
    font-size: 0.72rem; color: var(--muted);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 0.8rem 1rem; margin-top: 2rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─── Navbar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-icon">🧠</div>
  <div>
    <div class="navbar-title">NeuroScan AI</div>
    <div class="navbar-sub">MRI · Binary Classifier · EfficientNet-B0 · MONAI</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_path = st.text_input(
        "Model checkpoint (.pth)",
        value="models/monai_efficientnet_b0_final.pth",
        help="Path to the saved MONAI checkpoint"
    )

    threshold = st.slider(
        "Classification threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.01,
        help="Probability threshold for PD classification. Default 0.5."
    )

    show_gradcam = st.toggle("Show GradCAM heatmap", value=True)
    show_raw     = st.toggle("Show raw probabilities", value=True)

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.markdown("""
    <div style='font-size:0.78rem; color:#64748b; line-height:1.9; font-family:"DM Mono",monospace;'>
    Backbone &nbsp;&nbsp; EfficientNet-B0<br>
    Framework &nbsp; MONAI 1.x<br>
    Input &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 224 × 224 px<br>
    Classes &nbsp;&nbsp;&nbsp; ET · PD<br>
    Training &nbsp;&nbsp; Two-phase FT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ Class Legend")
    st.markdown("""
    <span class='result-badge badge-et'>ET</span> &nbsp; Essential Tremor<br><br>
    <span class='result-badge badge-pd'>PD</span> &nbsp; Tremor-dominant Parkinson's
    """, unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def get_model(path):
    return load_model(path)

try:
    model, meta = get_model(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model: `{e}`\n\nMake sure `{model_path}` is in the working directory.")


# ─── Main layout: two columns ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.05], gap="large")

with col_left:
    st.markdown('<div class="card-title">Upload MRI Slice</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("L")  # grayscale

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Uploaded Slice</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, clamp=True)
        st.markdown(f"""
        <div style='font-size:0.72rem; color:#64748b; font-family:"DM Mono",monospace; margin-top:0.5rem;'>
        {uploaded.name} &nbsp;·&nbsp; {image.size[0]}×{image.size[1]} px &nbsp;·&nbsp;
        {uploaded.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("🔍 Run Classification", use_container_width=True, disabled=not model_loaded)
    else:
        st.markdown("""
        <div class='card' style='text-align:center; padding: 3rem 1rem; color:#64748b;'>
            <div style='font-size:2.5rem; margin-bottom:1rem;'>🫧</div>
            <div style='font-weight:600; margin-bottom:0.4rem;'>Drop an MRI slice here</div>
            <div style='font-size:0.78rem;'>PNG, JPG, TIFF · Single 2D slice</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False


# ─── Results column ───────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-title">Classification Result</div>', unsafe_allow_html=True)

    if uploaded and run_btn and model_loaded:
        with st.spinner("Running inference…"):
            img_array = np.array(image).astype(np.float32)
            t0        = time.perf_counter()
            probs     = predict(model, img_array, meta)
            latency   = (time.perf_counter() - t0) * 1000

        et_prob = float(probs[0])
        pd_prob = float(probs[1])
        pred    = 1 if pd_prob >= threshold else 0
        label   = "PD" if pred == 1 else "ET"
        conf    = pd_prob if pred == 1 else et_prob

        # ── Result badge ──
        badge_cls = "badge-pd" if pred == 1 else "badge-et"
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Prediction</div>
          <span class='result-badge {badge_cls}'>{label}</span>
          <div style='font-size:1rem; font-weight:600; margin-top:0.6rem;'>
              {"Parkinson's Disease (tremor-dominant)" if pred == 1 else "Essential Tremor"}
          </div>
          <div style='font-size:0.8rem; color:#64748b; margin-top:0.2rem;'>
              Confidence: {conf*100:.1f}% &nbsp;·&nbsp; Threshold: {threshold:.2f}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars ──
        if show_raw:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Class Probabilities</div>', unsafe_allow_html=True)
            for name, prob, cls in [("ET — Essential Tremor", et_prob, "et"),
                                     ("PD — Parkinson's Disease", pd_prob, "pd")]:
                st.markdown(f"""
                <div class='conf-bar-wrap'>
                  <div class='conf-label'><span>{name}</span><span>{prob*100:.1f}%</span></div>
                  <div class='conf-track'>
                    <div class='conf-fill fill-{cls}' style='width:{prob*100:.1f}%'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Metrics row ──
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-tile'>
            <div class='metric-val'>{pd_prob*100:.1f}%</div>
            <div class='metric-key'>PD Probability</div>
          </div>
          <div class='metric-tile'>
            <div class='metric-val'>{et_prob*100:.1f}%</div>
            <div class='metric-key'>ET Probability</div>
          </div>
          <div class='metric-tile'>
            <div class='metric-val'>{latency:.0f}ms</div>
            <div class='metric-key'>Inference</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── GradCAM ──
        if show_gradcam:
            with st.spinner("Generating GradCAM…"):
                try:
                    heatmap = generate_gradcam(model, img_array, meta, pred)
                    h, w = img_array.shape[:2]
                    heatmap = np.array(Image.fromarray(heatmap).resize((w, h), Image.Resampling.BILINEAR))

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    fig.patch.set_facecolor("#131c2e")

                    titles = ["Original", "GradCAM", "Overlay"]
                    imgs_to_show = [
                        img_array / 255.0,
                        heatmap,
                        None  # overlay handled separately
                    ]

                    # Overlay
                    img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
                    colored  = cm.jet(heatmap)[:, :, :3]
                    overlay  = 0.55 * np.stack([img_norm]*3, axis=-1) + 0.45 * colored

                    for ax, title, img_data in zip(axes, titles, [img_array/255.0, heatmap, overlay]):
                        ax.set_facecolor("#131c2e")
                        if title == "GradCAM":
                            ax.imshow(img_data, cmap="jet", vmin=0, vmax=1)
                        elif title == "Original":
                            ax.imshow(img_data, cmap="gray", vmin=0, vmax=1)
                        else:
                            ax.imshow(img_data, vmin=0, vmax=1)
                        ax.set_title(title, color="#94a3b8", fontsize=10, pad=8)
                        ax.axis("off")

                    plt.tight_layout(pad=1.2)

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                                facecolor="#131c2e")
                    buf.seek(0)
                    plt.close()

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">GradCAM Attention Map</div>', unsafe_allow_html=True)
                    st.image(buf, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"GradCAM skipped: {e}")

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
⚠️ <strong>Research use only.</strong> This tool is a thesis prototype and is not a validated clinical
diagnostic device. Results must not be used as a substitute for professional medical evaluation.
Always consult a qualified neurologist for diagnosis.
</div>
""", unsafe_allow_html=True)
