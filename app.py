import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PestID · Insect Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — Dark Cinematic Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@300;400;500;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Tokens ── */
:root {
    --bg-deep:      #080C10;
    --bg-base:      #0D1117;
    --bg-panel:     #111820;
    --bg-card:      #141C26;
    --bg-card-hover:#182030;
    --bg-lift:      #1C2535;
    --border-sub:   rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.08);
    --border-mid:   rgba(255,255,255,0.13);
    --border-hi:    rgba(255,255,255,0.22);
    --teal:         #00E5C8;
    --teal-dim:     rgba(0,229,200,0.12);
    --teal-border:  rgba(0,229,200,0.28);
    --teal-glow:    rgba(0,229,200,0.06);
    --teal-dark:    #00A090;
    --blue:         #4D9FFF;
    --blue-dim:     rgba(77,159,255,0.12);
    --amber:        #F5A623;
    --amber-dim:    rgba(245,166,35,0.12);
    --red:          #FF5C6A;
    --red-dim:      rgba(255,92,106,0.12);
    --green:        #34D058;
    --green-dim:    rgba(52,208,88,0.12);
    --text-primary: #E8EDF5;
    --text-sec:     #8898AA;
    --text-muted:   #4D5F73;
    --text-hint:    #2E3D50;
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    16px;
    --radius-xl:    22px;
}

/* ── Reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: var(--bg-base) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

.block-container {
    padding: 0 2.75rem 3rem !important;
    max-width: 1280px !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"] {
    padding: 2.25rem 1.75rem !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--bg-lift); border-radius: 4px; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem 1.5rem !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--teal-border) !important;
}
[data-testid="stFileUploadDropzone"] { background: transparent !important; }
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: var(--text-sec) !important;
    font-size: 13px !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-mid) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: var(--teal) !important;
    color: #04100E !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    height: 3rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--teal) !important; }

/* ── HR ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 0 !important; }

/* ── Markdown ── */
[data-testid="stMarkdownContainer"] p {
    color: var(--text-sec) !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}

/* ─────────────────────────────
   COMPONENTS
───────────────────────────── */

/* Topbar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}
.logo-mark {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-icon {
    width: 32px; height: 32px;
    background: var(--teal-dim);
    border: 1px solid var(--teal-border);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.logo-text {
    font-size: 18px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}
.logo-dot { color: var(--teal); }
.nav-pills {
    display: flex; align-items: center; gap: 8px;
}
.nav-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid var(--border-mid);
    border-radius: 100px;
    padding: 4px 12px;
    color: var(--text-muted);
    background: var(--bg-card);
}
.nav-pill.active {
    border-color: var(--teal-border);
    color: var(--teal);
    background: var(--teal-dim);
}

/* Section label */
.slabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.7rem;
    display: flex; align-items: center; gap: 8px;
}
.slabel::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-sub);
}

/* Empty state */
.empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 10px;
    padding: 4rem 2rem;
    background: var(--bg-card);
    border: 1px dashed var(--border-mid);
    border-radius: var(--radius-lg);
    text-align: center;
}
.empty-icon { font-size: 28px; opacity: 0.25; }
.empty-title { font-size: 14px; font-weight: 500; color: var(--text-sec); }
.empty-sub { font-size: 12px; color: var(--text-muted); max-width: 200px; line-height: 1.6; }

/* Result card */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--teal-border) 30%,
        var(--teal-border) 70%,
        transparent 100%);
}
.result-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.4rem;
}
.result-name {
    font-size: 26px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 1.25rem;
}
.conf-meta {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 7px;
}
.conf-label { font-size: 12px; color: var(--text-muted); }
.conf-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px; font-weight: 500;
    color: var(--text-primary);
}
.conf-track {
    height: 6px;
    background: var(--bg-lift);
    border-radius: 100px;
    overflow: hidden;
}
.conf-fill { height: 6px; border-radius: 100px; }
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.02em;
    border-radius: 100px;
    padding: 5px 14px;
    margin-top: 1.1rem;
}

/* Stats row */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 1.25rem;
}
.stat-cell {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
}
.stat-cell-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.stat-cell-value {
    font-size: 20px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.stat-cell-value.teal { color: var(--teal); }

/* Ranking table */
.rank-table {
    background: var(--bg-card);
    border: 1px solid var(--border-mid);
    border-radius: var(--radius-lg);
    overflow: hidden;
}
.rank-thead {
    display: grid;
    grid-template-columns: 42px 1fr 110px 58px;
    padding: 0.55rem 1.25rem;
    background: var(--bg-lift);
    border-bottom: 1px solid var(--border);
}
.th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.rank-row {
    display: grid;
    grid-template-columns: 42px 1fr 110px 58px;
    padding: 0.75rem 1.25rem;
    align-items: center;
    border-bottom: 1px solid var(--border-sub);
    transition: background 0.15s;
}
.rank-row:last-child { border-bottom: none; }
.rank-row:hover { background: var(--bg-card-hover); }
.rank-row.r1 {
    background: linear-gradient(90deg, rgba(0,229,200,0.06) 0%, transparent 60%);
    border-bottom: 1px solid var(--border);
}
.rank-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-muted);
}
.rank-num.r1 { color: var(--teal); font-weight: 500; }
.rank-name { font-size: 13px; color: var(--text-sec); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 12px; }
.rank-name.r1 { color: var(--text-primary); font-weight: 700; }
.bar-wrap { height: 4px; background: var(--bg-lift); border-radius: 100px; overflow: hidden; }
.bar-fill { height: 4px; border-radius: 100px; background: var(--border-hi); }
.bar-fill.r1 { background: var(--teal); }
.rank-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-muted); text-align: right;
}
.rank-pct.r1 { color: var(--teal); }

/* Chip */
.chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 3px 10px;
    color: var(--text-muted);
    background: var(--bg-card);
}
.chip.live {
    border-color: var(--teal-border);
    color: var(--teal);
    background: var(--teal-dim);
}

/* Live pulse dot */
.live-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: pulse 1.8s ease-in-out infinite;
    vertical-align: middle;
    margin-right: 5px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.35; transform: scale(0.7); }
}

/* Sidebar */
.sb-wordmark {
    font-size: 20px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}
.sb-wordmark span { color: var(--teal); }
.sb-sub { font-size: 11px; color: var(--text-muted); margin-top: 3px; font-family: 'JetBrains Mono', monospace; }
.sb-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.5rem 0 0.65rem;
}
.sb-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-sub);
    font-size: 12px;
}
.sb-row-label { color: var(--text-muted); }
.sb-row-val { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-sec); }
.sb-tag {
    display: inline-block;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    background: var(--bg-lift);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 2px 8px;
    color: var(--text-muted);
    margin: 2px 2px 2px 0;
}
.sb-step {
    display: flex; align-items: flex-start; gap: 10px;
    font-size: 12px; color: var(--text-sec);
    padding: 0.4rem 0;
    line-height: 1.5;
}
.sb-step-num {
    width: 18px; height: 18px; min-width: 18px;
    border-radius: 50%;
    background: var(--bg-lift);
    border: 1px solid var(--border-mid);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text-muted);
    display: flex; align-items: center; justify-content: center;
    margin-top: 1px;
}

/* Page heading */
.page-hero {
    margin-bottom: 2.25rem;
}
.page-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.5rem;
}
.page-title {
    font-size: 36px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 0 0 0.6rem;
}
.page-desc {
    font-size: 14px; color: var(--text-sec);
    max-width: 500px; line-height: 1.7;
    margin: 0;
}

/* File name row */
.file-meta {
    display: flex; justify-content: space-between;
    margin-top: 0.6rem; padding: 0 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE & MODEL
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
NUM_CLASSES = len(class_names)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    m = models.resnet50(pretrained=False)
    m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    m.to(device).eval()
    return m

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
<div style="margin-bottom:1.75rem;">
  <div class="sb-wordmark">Pest<span>ID</span></div>
  <div class="sb-sub">v2.0 · LIVE INFERENCE</div>
</div>
<div style="height:1px;background:var(--border);"></div>

<div class="sb-section">System</div>
<div class="sb-row">
  <span class="sb-row-label">Status</span>
  <span class="sb-row-val"><span class="live-dot"></span>Online</span>
</div>
<div class="sb-row">
  <span class="sb-row-label">Device</span>
  <span class="sb-row-val">{str(device).upper()}</span>
</div>
<div class="sb-row">
  <span class="sb-row-label">Classes</span>
  <span class="sb-row-val">{NUM_CLASSES} species</span>
</div>
<div class="sb-row" style="border:none;">
  <span class="sb-row-label">Precision</span>
  <span class="sb-row-val">FP32</span>
</div>

<div class="sb-section">Model</div>
<div class="sb-row">
  <span class="sb-row-label">Architecture</span>
  <span class="sb-row-val">ResNet-50</span>
</div>
<div class="sb-row">
  <span class="sb-row-label">Training</span>
  <span class="sb-row-val">Transfer</span>
</div>
<div class="sb-row" style="border:none;">
  <span class="sb-row-label">Dataset</span>
  <span class="sb-row-val">IP102</span>
</div>

<div class="sb-section">Stack</div>
<div style="margin-bottom:1.25rem;">
  <span class="sb-tag">PyTorch</span>
  <span class="sb-tag">torchvision</span>
  <span class="sb-tag">Streamlit</span>
  <span class="sb-tag">CUDA</span>
</div>

<div style="height:1px;background:var(--border);"></div>

<div class="sb-section">Workflow</div>
<div class="sb-step">
  <div class="sb-step-num">1</div>
  <div>Upload a JPG or PNG insect photo</div>
</div>
<div class="sb-step">
  <div class="sb-step-num">2</div>
  <div>Click <strong style="color:var(--teal);">Run Analysis</strong></div>
</div>
<div class="sb-step">
  <div class="sb-step-num">3</div>
  <div>Review species ID &amp; confidence scores</div>
</div>

<div class="sb-section">Image tips</div>
<div style="font-size:11px;color:var(--text-muted);line-height:2.1;
            font-family:'JetBrains Mono',monospace;">
  · Clear, well-lit close-up<br>
  · Insect fills the frame<br>
  · Avoid blur &amp; overexposure
</div>

<div style="margin-top:2.5rem;padding-top:1.25rem;border-top:1px solid var(--border);">
  <div style="font-size:10px;color:var(--text-muted);
              font-family:'JetBrains Mono',monospace;">
    Built with PyTorch &amp; Streamlit
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="topbar">
  <div class="logo-mark">
    <div class="logo-icon">🔬</div>
    <div class="logo-text">Pest<span class="logo-dot">ID</span></div>
  </div>
  <div class="nav-pills">
    <div class="nav-pill active">
      <span class="live-dot" style="width:5px;height:5px;margin-right:4px;vertical-align:middle;"></span>
      Live
    </div>
    <div class="nav-pill">ResNet-50</div>
    <div class="nav-pill">IP102 · {NUM_CLASSES} classes</div>
    <div class="nav-pill">{str(device).upper()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-hero">
  <div class="page-tag">· Deep Learning · Entomology · Real-time</div>
  <h1 class="page-title">Insect Pest Classifier</h1>
  <p class="page-desc">
    Upload a photograph and the model identifies the pest species in seconds —
    powered by ResNet-50 transfer learning on the IP102 benchmark.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="stats-row" style="margin-bottom:2rem;">
  <div class="stat-cell">
    <div class="stat-cell-label">Model</div>
    <div class="stat-cell-value" style="font-size:15px;letter-spacing:0;">ResNet-50</div>
  </div>
  <div class="stat-cell">
    <div class="stat-cell-label">Species</div>
    <div class="stat-cell-value teal">{NUM_CLASSES}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-cell-label">Backend</div>
    <div class="stat-cell-value" style="font-size:15px;letter-spacing:0;">{str(device).upper()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([9, 11], gap="large")

# ── LEFT — Upload ─────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="slabel">Image Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop image here",
        type=["jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "tiff"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown(f"""
<div class="file-meta">
  <span>{uploaded_file.name}</span>
  <span>{image.width} × {image.height} px &nbsp;·&nbsp; {uploaded_file.size // 1024} KB</span>
</div>
""", unsafe_allow_html=True)

        input_tensor = transform(image).unsqueeze(0).to(device)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡  Run Analysis")
    else:
        st.markdown("""
<div class="empty" style="min-height:260px;">
  <div class="empty-icon">🖼️</div>
  <div class="empty-title">No image selected</div>
  <div class="empty-sub">Drag & drop or click to upload a JPG or PNG</div>
</div>
""", unsafe_allow_html=True)
        run_btn = False

# ── RIGHT — Results ────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="slabel">Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file and run_btn:
        # ── Live inference with progress ──────────────────────────────────────
        progress_placeholder = st.empty()
        status_placeholder   = st.empty()

        steps = [
            ("Preprocessing image…",   25),
            ("Running forward pass…",   60),
            ("Computing softmax…",      85),
            ("Ranking predictions…",   100),
        ]
        bar = progress_placeholder.progress(0)
        for label, pct in steps:
            status_placeholder.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:var(--teal);margin-bottom:0.5rem;">{label}</div>',
                unsafe_allow_html=True,
            )
            bar.progress(pct)
            time.sleep(0.25)

        progress_placeholder.empty()
        status_placeholder.empty()

        # ── Actual inference ──────────────────────────────────────────────────
        with torch.no_grad():
            output = model(input_tensor)
            probs  = F.softmax(output, dim=1)
            conf, pred  = torch.max(probs, 1)
            predicted_class = class_names[pred.item()]
            conf_score      = conf.item() * 100

        top5_prob, top5_catid = torch.topk(probs, 5)
        max_prob = top5_prob[0][0].item() * 100

        # Confidence theming
        if conf_score >= 75:
            fill    = "#00E5C8"; chip_bg = "rgba(0,229,200,0.10)"
            chip_fg = "#00E5C8"; dot_c   = "#00E5C8"; label = "High Confidence"
        elif conf_score >= 45:
            fill    = "#F5A623"; chip_bg = "rgba(245,166,35,0.10)"
            chip_fg = "#F5A623"; dot_c   = "#F5A623"; label = "Moderate Confidence"
        else:
            fill    = "#FF5C6A"; chip_bg = "rgba(255,92,106,0.10)"
            chip_fg = "#FF5C6A"; dot_c   = "#FF5C6A"; label = "Low Confidence"

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown(f"""
<div class="result-card">
  <div class="result-eyebrow">Top Prediction</div>
  <div class="result-name">{predicted_class}</div>

  <div class="conf-meta">
    <span class="conf-label">Confidence score</span>
    <span class="conf-value">{conf_score:.1f}%</span>
  </div>
  <div class="conf-track">
    <div class="conf-fill" style="width:{conf_score:.1f}%;background:{fill};"></div>
  </div>
  <div>
    <span class="status-badge" style="background:{chip_bg};color:{chip_fg};">
      <svg width="6" height="6" viewBox="0 0 6 6" aria-hidden="true">
        <circle cx="3" cy="3" r="3" fill="{dot_c}"/>
      </svg>
      {label}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Ranking table ─────────────────────────────────────────────────────
        st.markdown('<div class="slabel" style="margin-top:1.5rem;">Top 5 Candidates</div>', unsafe_allow_html=True)

        rows_html = ""
        for i in range(5):
            cls  = class_names[top5_catid[0][i]]
            prob = top5_prob[0][i].item() * 100
            bw   = round((prob / max_prob) * 100)
            r1   = "r1" if i == 0 else ""
            label_rank = f"#{i+1}"
            rows_html += f"""
<div class="rank-row {r1}">
  <div class="rank-num {r1}">{label_rank}</div>
  <div class="rank-name {r1}">{cls}</div>
  <div class="bar-wrap">
    <div class="bar-fill {r1}" style="width:{bw}%;"></div>
  </div>
  <div class="rank-pct {r1}">{prob:.1f}%</div>
</div>"""

        st.markdown(f"""
<div class="rank-table">
  <div class="rank-thead">
    <div class="th">Rank</div>
    <div class="th">Species</div>
    <div class="th">Confidence</div>
    <div class="th" style="text-align:right;">Score</div>
  </div>
  {rows_html}
</div>
""", unsafe_allow_html=True)

        # ── Meta row ──────────────────────────────────────────────────────────
        st.markdown(f"""
<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:1rem;align-items:center;">
  <span class="chip live"><span class="live-dot" style="width:5px;height:5px;margin-right:3px;vertical-align:middle;"></span>Inference complete</span>
  <span class="chip">ResNet-50</span>
  <span class="chip">{str(device).upper()}</span>
  <span class="chip">Top-5 softmax</span>
</div>
""", unsafe_allow_html=True)

    elif uploaded_file and not run_btn:
        st.markdown("""
<div class="empty" style="min-height:340px;">
  <div class="empty-icon">⚡</div>
  <div class="empty-title">Ready to analyse</div>
  <div class="empty-sub">Click <strong style="color:var(--teal);">Run Analysis</strong> to identify the species.</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div class="empty" style="min-height:340px;">
  <div class="empty-icon">📡</div>
  <div class="empty-title">Awaiting input</div>
  <div class="empty-sub">Upload an insect photograph on the left to begin.</div>
</div>
""", unsafe_allow_html=True)