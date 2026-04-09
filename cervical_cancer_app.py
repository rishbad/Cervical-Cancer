"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CERVICAL CANCER DETECTION — Streamlit Dashboard                           ║
║   Supports: SIPaKMeD (5 classes) & PapSmear (8 Bethesda classes)             ║
║   Run: streamlit run cervical_cancer_app.py                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json, warnings
from pathlib import Path
from io import BytesIO
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

try:
    import timm; TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ── Dataset Configurations ───────────────────────────────────────────────────
DATASET_CONFIG = {
    "SIPaKMeD": {
        "output_dir": Path("./cervical_output"),
        "name": "SIPaKMeD",
        "subtitle": "Single-cell cytology · 5 cell types",
        "classes": [
            "Dyskeratotic",
            "Koilocytotic",
            "Metaplastic",
            "Parabasal",
            "Superficial_Intermediate"
        ],
        "class_meta": {
            "Dyskeratotic"            : {"label":"Cancerous",  "color":"#e74c3c", "icon":"🔴"},
            "Koilocytotic"            : {"label":"Cancerous",  "color":"#e67e22", "icon":"🔴"},
            "Metaplastic"             : {"label":"Benign",     "color":"#f1c40f", "icon":"🟡"},
            "Parabasal"               : {"label":"Normal",     "color":"#2ecc71", "icon":"🟢"},
            "Superficial_Intermediate": {"label":"Normal",     "color":"#27ae60", "icon":"🟢"},
        },
        "descriptions": {
            "Dyskeratotic"            : "Abnormal keratinization — indicator of malignancy",
            "Koilocytotic"            : "HPV-infected cells — pre-cancerous changes",
            "Metaplastic"             : "Benign cellular changes — monitor closely",
            "Parabasal"               : "Immature normal cells from deep layers",
            "Superficial_Intermediate": "Mature healthy epithelial cells",
        }
    },
    "PapSmear": {
        "output_dir": Path("./cervical_output_papsmear"),
        "name": "PapSmear",
        "subtitle": "Bethesda System · 8 categories · ORCI Tanzania",
        "classes": ["ASC-US", "ASC-H", "AGC", "HSIL", "LSIL", "NILM", "SC", "A"],
        "class_meta": {
            "ASC-US": {"label":"Atypical",   "color":"#e67e22", "icon":"🟠"},
            "ASC-H" : {"label":"Atypical",   "color":"#e67e22", "icon":"🟠"},
            "AGC"   : {"label":"Atypical",   "color":"#e67e22", "icon":"🟠"},
            "HSIL"  : {"label":"Lesion",     "color":"#e74c3c", "icon":"🔴"},
            "LSIL"  : {"label":"Lesion",     "color":"#f1c40f", "icon":"🟡"},
            "NILM"  : {"label":"Normal",     "color":"#2ecc71", "icon":"🟢"},
            "SC"    : {"label":"Carcinoma",  "color":"#9b59b6", "icon":"🔴"},
            "A"     : {"label":"Adenocarcinoma", "color":"#9b59b6", "icon":"🔴"},
        },
        "descriptions": {
            "ASC-US": "Atypical Squamous Cells of Undetermined Significance",
            "ASC-H" : "Atypical Squamous Cells, cannot exclude HSIL",
            "AGC"   : "Atypical Glandular Cells",
            "HSIL"  : "High-grade Squamous Intraepithelial Lesion",
            "LSIL"  : "Low-grade Squamous Intraepithelial Lesion",
            "NILM"  : "Negative for Intraepithelial Lesion or Malignancy",
            "SC"    : "Squamous Cell Carcinoma",
            "A"     : "Adenocarcinoma",
        }
    },
}

IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def get_active_config():
    """Get config for currently selected dataset."""
    ds = st.session_state.get("selected_dataset", "SIPaKMeD")
    return DATASET_CONFIG[ds]


def get_paths():
    cfg = get_active_config()
    od = cfg["output_dir"]
    return {
        "output_dir": od,
        "results_dir": od / "results",
        "plots_dir": od / "plots",
        "gradcam_dir": od / "gradcam",
        "checkpoint_dir": od / "checkpoints",
    }

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cervical Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }

/* Header banner */
.app-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 60%, #1a0a2e 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(88,130,238,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(174,67,238,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-blue  { background:#1d3557; color:#58a6ff; border:1px solid #1f6feb; }
.badge-green { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.badge-purple{ background:#2d1a47; color:#bc8cff; border:1px solid #6e40c9; }

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Result cards */
.result-cancer {
    background: #2d0f0f;
    border: 2px solid #da3633;
    border-radius: 10px;
    padding: 1rem 1.4rem;
}
.result-benign {
    background: #2d2200;
    border: 2px solid #d29922;
    border-radius: 10px;
    padding: 1rem 1.4rem;
}
.result-normal {
    background: #0d2818;
    border: 2px solid #238636;
    border-radius: 10px;
    padding: 1rem 1.4rem;
}
.result-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.result-sub {
    font-size: 0.8rem;
    opacity: 0.75;
    font-family: 'IBM Plex Mono', monospace;
}

/* Progress bars */
.prob-row {
    display: flex;
    align-items: center;
    margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    gap: 10px;
}
.prob-label { width: 170px; color: #c9d1d9; flex-shrink: 0; }
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #21262d;
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.prob-pct { width: 50px; text-align: right; color: #8b949e; }

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #8b949e;
    font-weight: 500;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1f2937 !important;
    color: #e6edf3 !important;
}

/* Model selector */
.model-chip {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    cursor: pointer;
    border: 1px solid #30363d;
    background: #161b22;
    color: #8b949e;
    margin: 3px;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding-left: 10px;
    margin: 1.2rem 0 0.8rem;
}

/* Checkpoint table */
.ckpt-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}
.ckpt-table th {
    background: #161b22;
    color: #8b949e;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #30363d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.ckpt-table td {
    padding: 7px 12px;
    color: #c9d1d9;
    border-bottom: 1px solid #21262d;
}
.ckpt-table tr:hover td { background: #161b22; }
.ckpt-best td { color: #3fb950; font-weight: 600; }

div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* Dataset selector highlight */
.stSelectbox > div > div {
    border-radius: 8px;
}
[data-baseweb="select"] {
    border-radius: 8px !important;
    border: 1px solid #30363d !important;
}

/* Improved empty state */
.empty-state {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    margin: 2rem 0;
}
.empty-state:hover { border-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_all_results():
    paths = get_paths()
    results = {}
    if not paths["results_dir"].exists():
        return results
    for f in paths["results_dir"].glob("*_results.json"):
        try:
            with open(f) as fp:
                d = json.load(fp)
            results[d["model_name"]] = d
        except Exception:
            pass
    return results


def load_summary():
    paths = get_paths()
    sp = paths["output_dir"] / "summary.json"
    if sp.exists():
        with open(sp) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model_weights(model_name, dataset_name, num_classes):
    """Load model for selected dataset (num_classes must match: 5 for SIPaKMeD, 8 for PapSmear)."""
    od = DATASET_CONFIG[dataset_name]["output_dir"]
    paths = {"checkpoint_dir": od / "checkpoints"}
    ckpt_path = paths["checkpoint_dir"] / model_name / "best.pth"
    if not ckpt_path.exists():
        return None
    state = torch.load(ckpt_path, map_location=DEVICE)
    if model_name == "EfficientNet_B4":
        if TIMM_AVAILABLE:
            m = timm.create_model("efficientnet_b4", pretrained=False,
                                  num_classes=num_classes)
        else:
            m = models.efficientnet_b4(weights=None)
            m.classifier = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, num_classes))
    elif model_name == "ResNet50":
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes))
    elif model_name == "DenseNet201":
        m = models.densenet201(weights=None)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes))
    else:
        return None
    m.load_state_dict(state["model"])
    m.to(DEVICE).eval()
    return m


def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self._acts = self._grads = None
        self._h = [
            layer.register_forward_hook(lambda _, __, o: setattr(self,'_acts',o.detach())),
            layer.register_full_backward_hook(lambda _,__,g: setattr(self,'_grads',g[0].detach())),
        ]
    def generate(self, t, cls=None):
        self.model.eval()
        out = self.model(t.unsqueeze(0).to(DEVICE))
        if cls is None: cls = out.argmax(1).item()
        self.model.zero_grad()
        out[0, cls].backward()
        w   = self._grads.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls
    def remove(self):
        for h in self._h: h.remove()


def get_gradcam_layer(model, name):
    if "EfficientNet" in name:
        return model.conv_head if TIMM_AVAILABLE else model.features[-1][0]
    elif "ResNet" in name:
        l = model.layer4[-1]
        return getattr(l, "conv3", l.conv2)
    elif "DenseNet" in name:
        blk = model.features.denseblock4
        last = list(blk.children())[-1]
        return getattr(last, "conv2", None)
    return None


def fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor="#0d1117")
    buf.seek(0)
    return buf


def plot_prob_bars(probs, pred_idx):
    cfg = get_active_config()
    classes = cfg["classes"]
    class_meta = cfg["class_meta"]
    html = ""
    for i, (cls, p) in enumerate(zip(classes, probs)):
        meta  = class_meta.get(cls, {"color":"#58a6ff", "icon":"•"})
        color = meta["color"]
        bold  = "font-weight:700; color:#e6edf3;" if i == pred_idx else ""
        html += f"""
        <div class="prob-row">
          <div class="prob-label" style="{bold}">{meta.get('icon','•')} {str(cls)[:18]}</div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{color}"></div>
          </div>
          <div class="prob-pct" style="{bold}">{p*100:.1f}%</div>
        </div>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
# Initialize session state for dataset
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = "SIPaKMeD"

with st.sidebar:
    st.markdown("### 📂 Dataset")
    dataset_choice = st.selectbox(
        "Select results to view",
        options=list(DATASET_CONFIG.keys()),
        format_func=lambda x: f"{x} — {len(DATASET_CONFIG[x]['classes'])} classes",
        key="dataset_select",
        label_visibility="collapsed"
    )
    st.session_state.selected_dataset = dataset_choice

    st.markdown("---")
    st.markdown("### 🔬 Navigation")
    page = st.radio("", [
        "🏠  Overview",
        "📊  Model Results",
        "🔍  Predict Image",
        "🎨  Grad-CAM Gallery",
        "💾  Checkpoints",
        "📈  Training Curves",
    ], label_visibility="collapsed")

    st.markdown("---")
    all_results = load_all_results()
    summary     = load_summary()
    paths       = get_paths()
    cfg         = get_active_config()

    if all_results:
        best_model = max(all_results.values(), key=lambda x: x.get("accuracy", 0))
        st.markdown(f"""
        <div style="background:#0d2818;border:1px solid #238636;border-radius:8px;padding:12px 14px;">
          <div style="font-size:0.65rem;color:#3fb950;font-family:'IBM Plex Mono',monospace;
                      text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">{dataset_choice} · Best</div>
          <div style="font-size:0.95rem;font-weight:700;color:#e6edf3;">
            {best_model['model_name']}</div>
          <div style="font-size:1.5rem;font-weight:700;color:#3fb950;
                      font-family:'IBM Plex Mono',monospace;">
            {best_model['accuracy']*100:.2f}%</div>
          <div style="font-size:0.7rem;color:#8b949e;">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No results for **{dataset_choice}**. Run the corresponding training script.")

    st.markdown("---")
    n_results = len(list(paths["results_dir"].glob("*_results.json"))) if paths["results_dir"].exists() else 0
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#8b949e;font-family:'IBM Plex Mono',monospace;">
    Dataset: <span style="color:#58a6ff;">{dataset_choice}</span><br>
    Device: <span style="color:#58a6ff;">{DEVICE.upper()}</span><br>
    Results: <span style="color:#58a6ff;">{n_results} saved</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
cfg = get_active_config()
st.markdown(f"""
<div class="app-header">
  <div class="app-title">🔬 Cervical Cancer Detection</div>
  <div class="app-subtitle">{cfg['name']} · {cfg['subtitle']}</div>
  <div style="margin-top:10px;">
    <span class="badge badge-blue">{cfg['name']}</span>
    <span class="badge badge-blue">EfficientNet-B4</span>
    <span class="badge badge-blue">ResNet50</span>
    <span class="badge badge-blue">DenseNet201</span>
    <span class="badge badge-green">Ensemble</span>
    <span class="badge badge-purple">Grad-CAM</span>
    <span class="badge badge-purple">TTA</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "Overview" in page:
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1,c2,c3,c4],
            ["Total Images","Train Set","Val Set","Test Set"],
            [summary.get("dataset_total","-"), summary.get("train_size","-"),
             summary.get("val_size","-"), summary.get("test_size","-")]
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

    # Cross-dataset comparison (when both exist)
    sipakmed_summary = None
    papsmear_summary = None
    if (DATASET_CONFIG["SIPaKMeD"]["output_dir"] / "summary.json").exists():
        with open(DATASET_CONFIG["SIPaKMeD"]["output_dir"] / "summary.json") as f:
            sipakmed_summary = json.load(f)
    if (DATASET_CONFIG["PapSmear"]["output_dir"] / "summary.json").exists():
        with open(DATASET_CONFIG["PapSmear"]["output_dir"] / "summary.json") as f:
            papsmear_summary = json.load(f)

    if sipakmed_summary and papsmear_summary:
        st.markdown('<div class="section-header">Dataset Comparison</div>',
                    unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            best_s = max(sipakmed_summary.get("results", []), key=lambda x: x.get("accuracy", 0), default={})
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;padding:1rem;">
              <div style="font-size:0.7rem;color:#8b949e;margin-bottom:4px;">SIPaKMeD</div>
              <div style="font-size:1.2rem;font-weight:700;color:#e6edf3;">{sipakmed_summary.get('dataset_total','—')} images</div>
              <div style="font-size:0.85rem;color:#3fb950;">Best: {best_s.get('model_name','—')} ({best_s.get('accuracy',0)*100:.1f}%)</div>
            </div>""", unsafe_allow_html=True)
        with dc2:
            best_p = max(papsmear_summary.get("results", []), key=lambda x: x.get("accuracy", 0), default={})
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;padding:1rem;">
              <div style="font-size:0.7rem;color:#8b949e;margin-bottom:4px;">PapSmear</div>
              <div style="font-size:1.2rem;font-weight:700;color:#e6edf3;">{papsmear_summary.get('dataset_total','—')} images</div>
              <div style="font-size:0.85rem;color:#3fb950;">Best: {best_p.get('model_name','—')} ({best_p.get('accuracy',0)*100:.1f}%)</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

    if all_results:
        st.markdown('<div class="section-header">Model Leaderboard</div>',
                    unsafe_allow_html=True)

        rows = sorted(all_results.values(),
                      key=lambda x: x.get("accuracy",0), reverse=True)
        header_cols = st.columns([2.5,1.3,1.3,1.3,1.3,1.3])
        for col, h in zip(header_cols,
                          ["Model","Accuracy","F1 Macro","Precision","Recall","AUC-ROC"]):
            col.markdown(f"**{h}**")

        for i, r in enumerate(rows):
            cols = st.columns([2.5,1.3,1.3,1.3,1.3,1.3])
            icon = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else "  "
            cols[0].markdown(f"{icon} **{r['model_name']}**")
            cols[1].markdown(f"`{r.get('accuracy',0)*100:.2f}%`")
            cols[2].markdown(f"`{r.get('f1_macro',0):.4f}`")
            cols[3].markdown(f"`{r.get('precision_macro',0):.4f}`")
            cols[4].markdown(f"`{r.get('recall_macro',0):.4f}`")
            cols[5].markdown(f"`{r.get('auc_macro',0):.4f}`")

        # Comparison plot
        paths = get_paths()
        cmp_plot = paths["plots_dir"] / "model_comparison.png"
        if cmp_plot.exists():
            st.markdown('<div class="section-header">Performance Comparison</div>',
                        unsafe_allow_html=True)
            st.image(str(cmp_plot), use_column_width=True)
    else:
        ds = st.session_state.get("selected_dataset", "SIPaKMeD")
        script = "cervical_cancer_papsmear.py" if ds == "PapSmear" else "cervical_cancer_train.py"
        st.info(f"No training results found for **{ds}**. Run `python {script}` first.")

    # Class reference
    st.markdown('<div class="section-header">Class Reference</div>',
                unsafe_allow_html=True)
    cfg = get_active_config()
    classes = cfg["classes"]
    class_meta = cfg["class_meta"]
    descriptions = cfg.get("descriptions", {})
    n_cols = 4
    cols = st.columns(n_cols)
    for i, cls in enumerate(classes):
        meta = class_meta.get(cls, {"icon":"•", "color":"#58a6ff", "label":"—"})
        desc = descriptions.get(cls, "")
        with cols[i % n_cols]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                        padding:12px;text-align:center;min-height:130px;margin-bottom:8px;">
              <div style="font-size:1.5rem">{meta['icon']}</div>
              <div style="font-size:0.75rem;font-weight:700;color:#e6edf3;margin-top:4px">{cls}</div>
              <div style="font-size:0.6rem;color:{meta['color']};font-weight:600;margin:3px 0">{meta['label']}</div>
              <div style="font-size:0.58rem;color:#8b949e;line-height:1.2">{desc[:50]}{'...' if len(desc)>50 else ''}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Model Results" in page:
    if not all_results:
        st.warning("No results saved yet. Run training first.")
    else:
        selected = st.selectbox("Select Model", list(all_results.keys()))
        r        = all_results[selected]

        # Top metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, label, val in zip(
            [c1,c2,c3,c4,c5],
            ["Test Accuracy","F1 Macro","Precision","Recall","AUC-ROC"],
            [f"{r.get('accuracy',0)*100:.2f}%",
             f"{r.get('f1_macro',0):.4f}",
             f"{r.get('precision_macro',0):.4f}",
             f"{r.get('recall_macro',0):.4f}",
             f"{r.get('auc_macro',0):.4f}"]
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.6rem">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

        # Per-class report
        st.markdown('<div class="section-header">Per-Class Report</div>',
                    unsafe_allow_html=True)
        report = r.get("classification_report", {})
        if report:
            per_cls = {k: v for k, v in report.items()
                       if k not in ["accuracy","macro avg","weighted avg"]
                       and isinstance(v, dict)}
            hcols = st.columns([2.5,1.2,1.2,1.2,1.2])
            for col, h in zip(hcols, ["Class","Precision","Recall","F1","Support"]):
                col.markdown(f"**{h}**")
            cfg = get_active_config()
            class_meta = cfg["class_meta"]
            for cls_name, metrics in per_cls.items():
                meta = class_meta.get(cls_name, {})
                icon = meta.get("icon","•")
                vcols = st.columns([2.5,1.2,1.2,1.2,1.2])
                vcols[0].markdown(f"{icon} {cls_name}")
                vcols[1].markdown(f"`{metrics.get('precision',0):.4f}`")
                vcols[2].markdown(f"`{metrics.get('recall',0):.4f}`")
                vcols[3].markdown(f"`{metrics.get('f1-score',0):.4f}`")
                vcols[4].markdown(f"`{int(metrics.get('support',0))}`")

        # Plots
        paths = get_paths()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Confusion Matrix</div>',
                        unsafe_allow_html=True)
            cm_img = paths["plots_dir"] / f"{selected}_confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), use_column_width=True)
        with col_b:
            st.markdown('<div class="section-header">ROC Curves</div>',
                        unsafe_allow_html=True)
            roc_img = paths["plots_dir"] / f"{selected}_roc.png"
            if roc_img.exists():
                st.image(str(roc_img), use_column_width=True)

        # Per-class AUC table
        roc_data = r.get("roc_data", {})
        if roc_data:
            st.markdown('<div class="section-header">Per-Class AUC-ROC</div>',
                        unsafe_allow_html=True)
            cfg = get_active_config()
            class_meta = cfg["class_meta"]
            acols = st.columns(len(roc_data))
            for col, (cls, d) in zip(acols, roc_data.items()):
                meta  = class_meta.get(cls, {})
                color = meta.get("color","#58a6ff")
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value" style="font-size:1.4rem;color:{color}">
                    {d['auc']:.4f}</div>
                  <div class="metric-label">{cls[:14]}</div>
                </div>""", unsafe_allow_html=True)

        # Download results
        st.markdown("")
        st.download_button(
            "⬇ Download Full Results JSON",
            data=json.dumps(r, indent=2),
            file_name=f"{selected}_results.json",
            mime="application/json"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT IMAGE
# ─────────────────────────────────────────────────────────────────────────────
elif "Predict" in page:
    paths = get_paths()
    cfg = get_active_config()
    classes = cfg["classes"]
    class_meta = cfg["class_meta"]
    num_classes = len(classes)

    available_models = [
        n for n in ["EfficientNet_B4","ResNet50","DenseNet201"]
        if (paths["checkpoint_dir"] / n / "best.pth").exists()
    ]

    if not available_models:
        ds = st.session_state.get("selected_dataset", "SIPaKMeD")
        train_script = "cervical_cancer_papsmear.py" if ds == "PapSmear" else "cervical_cancer_train.py"
        st.error(f"No trained model checkpoints found for **{ds}**. Run `python {train_script}` first.")
        st.stop()

    c_left, c_right = st.columns([1, 2])

    with c_left:
        st.markdown('<div class="section-header">Settings</div>',
                    unsafe_allow_html=True)
        st.caption(f"Dataset: **{cfg['name']}** · {num_classes} classes")
        model_choice = st.selectbox("Model", available_models)
        show_gc      = st.checkbox("Show Grad-CAM heatmap", value=True)
        tta_n        = st.slider("TTA passes", 0, 8, 3,
                                 help="More passes = more accurate but slower")

        st.markdown('<div class="section-header">Upload Image</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("",
                                    type=["jpg","jpeg","png","bmp","tiff"],
                                    label_visibility="collapsed")

    with c_right:
        if uploaded is None:
            st.markdown("""
            <div style="background:#161b22;border:2px dashed #30363d;border-radius:12px;
                        padding:3rem;text-align:center;margin-top:2rem;">
              <div style="font-size:3rem">🔬</div>
              <div style="color:#8b949e;margin-top:1rem;">
                Upload a Pap smear cell image to begin analysis</div>
              <div style="color:#30363d;font-size:0.8rem;margin-top:0.5rem;">
                Supported: JPG, PNG, BMP, TIFF</div>
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Loading model and running inference..."):
                img    = Image.open(uploaded).convert("RGB")
                ds_name = st.session_state.get("selected_dataset", "SIPaKMeD")
                model  = load_model_weights(model_choice, ds_name, num_classes)

                if model is None:
                    st.error(f"Could not load {model_choice}.")
                    st.stop()

                tf     = get_transform()
                img_t  = tf(img)

                # TTA inference
                tta_tf = T.Compose([
                    T.Resize((IMG_SIZE, IMG_SIZE)),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomRotation(10),
                    T.ToTensor(),
                    T.Normalize(MEAN, STD),
                ])

                with torch.no_grad():
                    base = F.softmax(model(img_t.unsqueeze(0).to(DEVICE)), 1).cpu().numpy()[0]
                    tta_acc = np.zeros(num_classes)
                    for _ in range(tta_n):
                        aug_t = tta_tf(img)
                        p = F.softmax(model(aug_t.unsqueeze(0).to(DEVICE)), 1).cpu().numpy()[0]
                        tta_acc += p
                    if tta_n > 0:
                        final_probs = (base + tta_acc) / (1 + tta_n)
                    else:
                        final_probs = base

                pred      = final_probs.argmax()
                pred_cls  = classes[pred]
                meta      = class_meta.get(pred_cls, {"label":"—", "color":"#58a6ff", "icon":"•"})
                conf      = final_probs[pred]

            # Display result
            col_img, col_res = st.columns([1, 1])

            with col_img:
                st.markdown('<div class="section-header">Input Image</div>',
                            unsafe_allow_html=True)
                st.image(img, use_column_width=True)

            with col_res:
                st.markdown('<div class="section-header">Prediction</div>',
                            unsafe_allow_html=True)

                css_class = ("result-cancer" if meta["label"] == "Cancerous"
                             else "result-benign" if meta["label"] == "Benign"
                             else "result-normal")
                color = meta["color"]
                st.markdown(f"""
                <div class="{css_class}">
                  <div class="result-title" style="color:{color}">
                    {meta['icon']} {pred_cls}</div>
                  <div class="result-sub">{meta['label']} · Confidence: {conf*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("")

                st.markdown("**Class Probabilities**")
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;'
                    f'border-radius:8px;padding:12px 16px;">'
                    f'{plot_prob_bars(final_probs, pred)}</div>',
                    unsafe_allow_html=True)

                info_map = {
                    "Cancerous"     : "⚠️ Abnormal cells detected. Immediate clinical evaluation recommended.",
                    "Carcinoma"     : "⚠️ Carcinoma detected. Urgent clinical evaluation required.",
                    "Adenocarcinoma": "⚠️ Adenocarcinoma detected. Urgent clinical evaluation required.",
                    "Lesion"        : "⚠️ Squamous lesion detected. Further evaluation recommended.",
                    "Atypical"      : "ℹ️ Atypical cells detected. Follow-up monitoring advised.",
                    "Benign"        : "ℹ️ Benign changes detected. Follow-up monitoring advised.",
                    "Normal"        : "✅ No abnormal cells detected in this sample.",
                }
                st.markdown("")
                st.info(info_map.get(meta["label"], "ℹ️ Classification complete. Consult clinical guidelines."))

                col_m1, col_m2 = st.columns(2)
                col_m1.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value" style="font-size:1.4rem">{conf*100:.1f}%</div>
                  <div class="metric-label">Confidence</div>
                </div>""", unsafe_allow_html=True)
                col_m2.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value" style="font-size:1.4rem">{tta_n+1}×</div>
                  <div class="metric-label">TTA Passes</div>
                </div>""", unsafe_allow_html=True)

            # Grad-CAM
            if show_gc:
                st.markdown('<div class="section-header">Grad-CAM Heatmap</div>',
                            unsafe_allow_html=True)
                try:
                    layer = get_gradcam_layer(model, model_choice)
                    if layer:
                        gc = GradCAM(model, layer)
                        cam, _ = gc.generate(img_t, cls=int(pred))
                        gc.remove()

                        inv = T.Normalize([-m/s for m,s in zip(MEAN,STD)],
                                          [1/s for s in STD])
                        raw = inv(img_t).permute(1,2,0).numpy()
                        raw = np.clip(raw, 0, 1)

                        fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                                                 facecolor="#0d1117")
                        titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
                        imgs_show = [raw,
                                     plt.cm.jet(cam)[:,:,:3],
                                     raw * 0.55 + plt.cm.jet(cam)[:,:,:3] * 0.45]
                        for ax, im, tt in zip(axes, imgs_show, titles):
                            ax.imshow(im)
                            ax.set_title(tt, color="#e6edf3", fontsize=10)
                            ax.axis("off")
                            for spine in ax.spines.values():
                                spine.set_color("#30363d")
                        plt.tight_layout(pad=0.5)
                        st.image(fig_to_buf(fig), use_column_width=True)
                        plt.close(fig)
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GRAD-CAM GALLERY
# ─────────────────────────────────────────────────────────────────────────────
elif "Grad-CAM" in page:
    paths = get_paths()
    cfg = get_active_config()
    classes = cfg["classes"]

    st.markdown('<div class="section-header">Grad-CAM Gallery — Saved from Training</div>',
                unsafe_allow_html=True)
    st.caption(f"Dataset: **{cfg['name']}**")

    if not paths["gradcam_dir"].exists():
        st.info("Grad-CAM images will appear here after training completes.")
    else:
        gradcam_dir = paths["gradcam_dir"]
        model_dirs = [d for d in gradcam_dir.iterdir() if d.is_dir()]
        gallery_imgs = list(gradcam_dir.glob("*_gradcam_gallery.png"))

        if gallery_imgs:
            tabs = st.tabs([img.stem.replace("_gradcam_gallery","") for img in gallery_imgs])
            for tab, img_path in zip(tabs, gallery_imgs):
                with tab:
                    st.image(str(img_path), use_column_width=True)
        else:
            st.info("No gallery images found yet.")

        # Individual per-class images
        if model_dirs:
            st.markdown('<div class="section-header">Individual Cell Analyses</div>',
                        unsafe_allow_html=True)
            sel_model = st.selectbox("Model", [d.name for d in model_dirs],
                                     key="gc_model")
            sel_dir   = gradcam_dir / sel_model
            all_gc_imgs = list(sel_dir.glob("*.png"))

            # Filter by class
            cls_filter = st.multiselect("Filter by class", classes, default=classes)
            filtered   = [p for p in all_gc_imgs
                          if any(c in p.name for c in cls_filter)]

            if filtered:
                page_size = 12
                n_pages   = max(1, (len(filtered) + page_size - 1) // page_size)
                if n_pages > 1:
                    pg = st.slider("Page", 1, n_pages, 1)
                else:
                    pg = 1
                batch = filtered[(pg-1)*page_size: pg*page_size]
                cols  = st.columns(3)
                for i, img_path in enumerate(batch):
                    with cols[i % 3]:
                        st.image(str(img_path), use_column_width=True,
                                 caption=img_path.stem[:30])
            else:
                st.info("No images match the selected filters.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────
elif "Checkpoints" in page:
    paths = get_paths()
    cfg = get_active_config()

    st.markdown('<div class="section-header">Checkpoint Browser</div>',
                unsafe_allow_html=True)
    st.caption(f"Dataset: **{cfg['name']}**")

    if not paths["checkpoint_dir"].exists():
        st.info("No checkpoints found yet. Run training first.")
    else:
        model_ckpt_dirs = [d for d in paths["checkpoint_dir"].iterdir() if d.is_dir()]
        if not model_ckpt_dirs:
            st.info("No checkpoints found.")
        else:
            for mdir in sorted(model_ckpt_dirs):
                log_file = mdir / "checkpoint_log.json"
                if not log_file.exists():
                    continue
                with open(log_file) as f:
                    log = json.load(f)

                best_ep  = log.get("best_epoch", -1)
                best_acc = log.get("best_val_acc", 0.0)
                ckpts    = log.get("checkpoints", [])

                with st.expander(
                    f"📁 {mdir.name}  —  Best: epoch {best_ep}  ({best_acc*100:.2f}%)",
                    expanded=True
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="font-size:1.4rem">
                        {best_acc*100:.2f}%</div>
                      <div class="metric-label">Best Val Accuracy</div>
                    </div>""", unsafe_allow_html=True)
                    c2.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="font-size:1.4rem">
                        {best_ep}</div>
                      <div class="metric-label">Best Epoch</div>
                    </div>""", unsafe_allow_html=True)
                    c3.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="font-size:1.4rem">
                        {len(ckpts)}</div>
                      <div class="metric-label">Saved Checkpoints</div>
                    </div>""", unsafe_allow_html=True)

                    if ckpts:
                        st.markdown("")
                        rows_html = ""
                        for ck in sorted(ckpts, key=lambda x: x["epoch"]):
                            is_best = ck["epoch"] == best_ep
                            cls_str = 'class="ckpt-best"' if is_best else ""
                            star    = " ★" if is_best else ""
                            rows_html += f"""<tr {cls_str}>
                              <td>Epoch {ck['epoch']}{star}</td>
                              <td>{ck['val_acc']*100:.2f}%</td>
                              <td>{ck['val_loss']:.4f}</td>
                              <td style="color:#8b949e;font-size:0.7rem">
                                {Path(ck['file']).name}</td>
                            </tr>"""
                        st.markdown(f"""
                        <table class="ckpt-table">
                          <thead><tr>
                            <th>Epoch</th><th>Val Accuracy</th>
                            <th>Val Loss</th><th>File</th>
                          </tr></thead>
                          <tbody>{rows_html}</tbody>
                        </table>""", unsafe_allow_html=True)

                    # File list
                    all_files = list(mdir.glob("*.pth"))
                    st.markdown(f"""
                    <div style="margin-top:12px;font-size:0.75rem;color:#8b949e;
                                font-family:'IBM Plex Mono',monospace;">
                    Checkpoint files: {', '.join(f.name for f in sorted(all_files))}
                    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
elif "Training Curves" in page:
    if not all_results:
        st.info("No training results found.")
    else:
        # Overlay all models on same axes
        st.markdown('<div class="section-header">All Models — Accuracy Comparison</div>',
                    unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#0d1117")
        colors    = ["#58a6ff","#3fb950","#f78166","#d2a8ff"]

        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.grid(alpha=0.2, color="#30363d")
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.tick_params(colors="#8b949e")

        for (mname, r), col in zip(all_results.items(), colors):
            hist = r.get("history", {})
            if not hist or "train_acc" not in hist:
                continue
            ep = range(1, len(hist["train_acc"]) + 1)
            axes[0].plot(ep, hist["train_acc"], color=col, lw=1.5,
                         alpha=0.5, linestyle="--")
            axes[0].plot(ep, hist["val_acc"], color=col, lw=2,
                         label=mname)
            if "train_loss" in hist:
                axes[1].plot(ep, hist["train_loss"], color=col, lw=1.5,
                             alpha=0.5, linestyle="--")
                axes[1].plot(ep, hist["val_loss"], color=col, lw=2,
                             label=mname)

        for ax, title, ylabel in zip(axes, ["Validation Accuracy","Validation Loss"],
                                     ["Accuracy","Loss"]):
            ax.set_title(title, color="#e6edf3", fontsize=11)
            ax.set_xlabel("Epoch", color="#8b949e")
            ax.set_ylabel(ylabel, color="#8b949e")
            leg = ax.legend(facecolor="#161b22", edgecolor="#30363d",
                            labelcolor="#e6edf3")

        plt.suptitle("Training History — All Models  (dashed=train, solid=val)",
                     color="#e6edf3", fontsize=12)
        plt.tight_layout()
        st.image(fig_to_buf(fig), use_column_width=True)
        plt.close(fig)

        # Individual history plots from saved PNGs
        paths = get_paths()
        st.markdown('<div class="section-header">Individual Training Histories</div>',
                    unsafe_allow_html=True)
        hist_plots = list(paths["plots_dir"].glob("*_history.png"))
        if hist_plots:
            tabs = st.tabs([p.stem.replace("_history","") for p in hist_plots])
            for tab, img_path in zip(tabs, hist_plots):
                with tab:
                    st.image(str(img_path), use_column_width=True)
        else:
            st.info("History plot images not found in plots directory.")
