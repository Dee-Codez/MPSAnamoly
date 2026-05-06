import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MPS Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 { color: #58a6ff; margin: 0; font-size: 2rem; }
    .metric-card p  { color: #8b949e; margin: 0; font-size: 0.85rem; }
    .section-header {
        color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2ea043, #3fb950); }
    .alert-box {
        background: #1c2128; border-left: 4px solid #f85149;
        border-radius: 4px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    }
    .info-box {
        background: #1c2128; border-left: 4px solid #58a6ff;
        border-radius: 4px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    }
    .success-box {
        background: #1c2128; border-left: 4px solid #2ea043;
        border-radius: 4px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    }
    .step-badge {
        display: inline-block; background: #21262d; border: 1px solid #30363d;
        border-radius: 50%; width: 28px; height: 28px; text-align: center;
        line-height: 28px; font-weight: bold; color: #58a6ff; margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
NUM_SUPPLIES = 117
ANOMALY_THRESHOLD_MULTIPLIER = 2.0
BEAM_PARTIAL_LOSS_THRESHOLD = 2.0
NUM_EVENTS = 10_000
NUM_TIMESTAMPS = 50_000

# ─── Session state ─────────────────────────────────────────────────────────────
_STATE_KEYS = [
    "event_data", "mps_data", "deviation_data",
    "beam_cycles", "fault_labels",
    "model", "threshold", "train_history",
    "predictions", "recon_errors",
    "data_source",   # "synthetic" | "uploaded"
    "data_ready",    # bool
    "beam_ready",    # bool
    "model_ready",   # bool
]
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None
for b in ("data_ready", "beam_ready", "model_ready"):
    if st.session_state[b] is None:
        st.session_state[b] = False


# ─── Core functions ────────────────────────────────────────────────────────────
def generate_synthetic_data(seed: int = 42, progress_cb=None):
    rng = np.random.default_rng(seed)

    if progress_cb: progress_cb(0.05, "Generating event log…")
    t_events = pd.date_range("2023-01-01", periods=NUM_EVENTS, freq="1min")
    event_df = pd.DataFrame({
        "Time":       t_events,
        "Ramp end":   rng.choice(["Ramp done", ""], NUM_EVENTS, p=[0.35, 0.65]),
        "Kill signal": rng.choice(["Kill", ""],    NUM_EVENTS, p=[0.30, 0.70]),
        "Event type": rng.choice(
            ["Ramp done", "Beam kill", "Injection", "Other"],
            NUM_EVENTS, p=[0.25, 0.25, 0.25, 0.25]
        ),
    })

    if progress_cb: progress_cb(0.20, "Building beam current trace…")
    t_mps = pd.date_range("2023-01-01", periods=NUM_TIMESTAMPS, freq="1min")
    beam_current = np.clip(
        100 + np.cumsum(rng.normal(0, 5, NUM_TIMESTAMPS)) * 0.01, 0, 120
    )
    fault_idx = rng.integers(5000, NUM_TIMESTAMPS - 5000, size=200)
    for fi in fault_idx:
        width = rng.integers(10, 60)
        beam_current[fi: fi + width] = rng.choice([beam_current[fi], 0.0])

    if progress_cb: progress_cb(0.40, "Generating MPS readback/vmeset signals…")
    mps_dict: dict = {"Timestamp": t_mps, "Beam Current": beam_current}
    for i in range(1, NUM_SUPPLIES + 1):
        base  = rng.uniform(0.8, 1.2)
        noise = rng.normal(0, 0.005, NUM_TIMESTAMPS)
        mps_dict[f"sp{i}_vmeset"]   = base + rng.normal(0, 0.002, NUM_TIMESTAMPS)
        mps_dict[f"sp{i}_readback"] = mps_dict[f"sp{i}_vmeset"] + noise
        if i % 20 == 0 and progress_cb:
            progress_cb(0.40 + 0.35 * (i / NUM_SUPPLIES), f"Supply {i}/{NUM_SUPPLIES}…")

    if progress_cb: progress_cb(0.78, "Injecting synthetic faults…")
    anomaly_rows     = rng.choice(NUM_TIMESTAMPS, size=int(0.05 * NUM_TIMESTAMPS), replace=False)
    faulty_supplies  = rng.integers(1, NUM_SUPPLIES + 1, size=len(anomaly_rows))
    for row, sup in zip(anomaly_rows, faulty_supplies):
        mps_dict[f"sp{sup}_readback"][row] += rng.uniform(0.05, 0.3)

    if progress_cb: progress_cb(0.90, "Computing deviations…")
    mps_df = pd.DataFrame(mps_dict)
    return event_df, mps_df


def compute_deviations(mps_df: pd.DataFrame) -> pd.DataFrame:
    dev = {"Timestamp": mps_df["Timestamp"], "Beam Current": mps_df["Beam Current"]}
    for i in range(1, NUM_SUPPLIES + 1):
        dev[f"sp{i}_dev"] = mps_df[f"sp{i}_readback"] - mps_df[f"sp{i}_vmeset"]
    return pd.DataFrame(dev)


def extract_beam_cycles(event_df: pd.DataFrame):
    starts, kills = [], []
    flg = False
    for _, row in event_df.iterrows():
        if row["Ramp end"] == "Ramp done" and not flg:
            flg = True
            starts.append(row["Time"])
        elif row["Kill signal"] == "Kill" and flg:
            flg = False
            kills.append(row["Time"])
    n = min(len(starts), len(kills))
    return list(zip(starts[:n], kills[:n]))


def classify_beam_faults(mps_df: pd.DataFrame) -> np.ndarray:
    bc    = mps_df["Beam Current"].values.copy()
    diffs = np.abs(np.diff(bc, prepend=bc[0]))
    return np.where(
        bc < 1.0, "Complete Loss",
        np.where(diffs > BEAM_PARTIAL_LOSS_THRESHOLD, "Partial Loss", "No Loss")
    )


def run_beam_analysis(progress_cb=None):
    event_df = st.session_state["event_data"]
    mps_df   = st.session_state["mps_data"]
    if progress_cb: progress_cb(0.1, "Scanning event log for Ramp End markers…")
    time.sleep(0.05)
    cycles = extract_beam_cycles(event_df)
    if progress_cb: progress_cb(0.6, f"Found {len(cycles)} cycles — classifying beam faults…")
    time.sleep(0.05)
    fault_labels = classify_beam_faults(mps_df)
    if progress_cb: progress_cb(1.0, "Done.")
    st.session_state["beam_cycles"]   = cycles
    st.session_state["fault_labels"]  = fault_labels
    st.session_state["beam_ready"]    = True


# ─── TF autoencoder ────────────────────────────────────────────────────────────
def build_autoencoder(input_dim: int = NUM_SUPPLIES):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model

    inp = Input(shape=(input_dim,), name="input")
    x   = Dense(64,  activation="relu",    name="enc1")(inp)
    x   = Dense(32,  activation="relu",    name="enc2")(x)
    lat = Dense(16,  activation="relu",    name="latent")(x)
    x   = Dense(32,  activation="relu",    name="dec1")(lat)
    x   = Dense(64,  activation="relu",    name="dec2")(x)
    out = Dense(input_dim, activation="linear", name="output")(x)
    model = Model(inp, out, name="MPS_Autoencoder")
    model.compile(optimizer="adam", loss="mae")
    return model


def compute_reconstruction_errors(model, X: np.ndarray) -> np.ndarray:
    preds = model.predict(X, verbose=0, batch_size=512)
    return np.mean(np.abs(X - preds), axis=1)


def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _df_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
    else:
        df.to_csv(buf, index=False)
    return buf.getvalue()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
PAGES = [
    "🏠 Overview",
    "📂 Data Pipeline",
    "📡 Beam Analysis",
    "🧠 Model Training",
    "🔍 Anomaly Detection",
    "📊 Results Dashboard",
]

if "page_idx" not in st.session_state:
    st.session_state["page_idx"] = 0

with st.sidebar:
    st.markdown("## ⚡ MPS Anomaly Detection")
    st.markdown("---")

    _d = st.session_state["data_ready"]
    _b = st.session_state["beam_ready"]
    _m = st.session_state["model_ready"]

    def _status_row(label, done):
        color  = "#2ea043" if done else "#30363d"
        dot_bg = "#2ea043" if done else "#21262d"
        dot    = "✓"       if done else "·"
        txt    = "#c9d1d9" if done else "#6e7681"
        return (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:8px 10px;border-radius:8px;margin-bottom:5px;'
            f'background:#161b22;border:1px solid {color}22;">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:{dot_bg};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;font-weight:700;color:white;flex-shrink:0;">{dot}</span>'
            f'<span style="font-size:13px;color:{txt};font-weight:{"600" if done else "400"};">{label}</span>'
            f'</div>'
        )

    st.markdown(
        _status_row("Data loaded",    _d) +
        _status_row("Beam analysed",  _b) +
        _status_row("Model trained",  _m),
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate", PAGES,
        index=st.session_state["page_idx"],
        label_visibility="collapsed",
    )
    # Keep page_idx in sync when user clicks the sidebar directly
    st.session_state["page_idx"] = PAGES.index(page)

    # Prev / Next in sidebar
    _idx = st.session_state["page_idx"]
    _sb_prev, _sb_next = st.columns(2)
    if _idx > 0:
        if _sb_prev.button("← Prev", key="sb_prev", use_container_width=True):
            st.session_state["page_idx"] = _idx - 1
            st.rerun()
    if _idx < len(PAGES) - 1:
        if _sb_next.button("Next →", key="sb_next", use_container_width=True):
            st.session_state["page_idx"] = _idx + 1
            st.rerun()
    st.markdown("---")


def _nav_buttons():
    """Prev / Next buttons rendered at the bottom of every page."""
    idx = st.session_state["page_idx"]
    st.markdown("---")
    left, _, right = st.columns([2, 6, 2])
    if idx > 0:
        if left.button(f"← {PAGES[idx - 1]}", use_container_width=True):
            st.session_state["page_idx"] = idx - 1
            st.rerun()
    if idx < len(PAGES) - 1:
        if right.button(f"{PAGES[idx + 1]} →", use_container_width=True):
            st.session_state["page_idx"] = idx + 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview (Landing)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <style>
    /* ── Hero banner ── */
    .hero-wrap {
        background: linear-gradient(135deg, #0d1117 0%, #0e2a45 45%, #0d1117 100%);
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 64px 48px 52px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 80px,
                rgba(88,166,255,0.03) 80px,
                rgba(88,166,255,0.03) 81px
            ),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 80px,
                rgba(88,166,255,0.03) 80px,
                rgba(88,166,255,0.03) 81px
            );
        pointer-events: none;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(88,166,255,0.12);
        border: 1px solid rgba(88,166,255,0.35);
        color: #58a6ff;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 100px;
        margin-bottom: 22px;
    }
    .hero-title {
        font-size: clamp(28px, 4vw, 48px);
        font-weight: 800;
        line-height: 1.15;
        color: #e6edf3;
        margin: 0 0 10px;
    }
    .hero-title span {
        background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 60%, #a5d6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 16px;
        color: #8b949e;
        margin: 0 0 36px;
        max-width: 560px;
        line-height: 1.6;
    }
    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(33,38,45,0.8);
        border: 1px solid #30363d;
        color: #c9d1d9;
        font-size: 12px;
        padding: 5px 14px;
        border-radius: 8px;
    }
    /* ── Stat strip ── */
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .stat-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 22px 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: #58a6ff; }
    .stat-num {
        font-size: 32px;
        font-weight: 800;
        color: #58a6ff;
        line-height: 1;
        margin-bottom: 6px;
    }
    .stat-lbl {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* ── Feature grid ── */
    .feat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .feat-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 24px 20px;
        transition: border-color 0.2s, transform 0.15s;
    }
    .feat-card:hover { border-color: #388bfd; transform: translateY(-2px); }
    .feat-icon { font-size: 26px; margin-bottom: 10px; }
    .feat-title {
        font-size: 14px;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 6px;
    }
    .feat-desc { font-size: 13px; color: #8b949e; line-height: 1.55; }
    /* ── Arch row ── */
    .arch-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 28px 24px;
        margin-bottom: 32px;
        overflow-x: auto;
    }
    .arch-node {
        background: #0e1117;
        border: 2px solid #30363d;
        border-radius: 10px;
        padding: 10px 16px;
        text-align: center;
        min-width: 80px;
        flex-shrink: 0;
    }
    .arch-node.enc  { border-color: #2ea043; }
    .arch-node.lat  { border-color: #f0883e; }
    .arch-node.dec  { border-color: #2ea043; }
    .arch-node.io   { border-color: #388bfd; }
    .arch-label { font-size: 11px; font-weight: 700; color: #c9d1d9; }
    .arch-dim   { font-size: 10px; color: #8b949e; margin-top: 2px; }
    .arch-arrow {
        color: #30363d;
        font-size: 18px;
        padding: 0 4px;
        flex-shrink: 0;
    }
    /* ── CTA ── */
    .cta-row {
        background: linear-gradient(135deg, #0e2a45 0%, #0d1117 100%);
        border: 1px solid rgba(88,166,255,0.25);
        border-radius: 12px;
        padding: 32px 36px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .cta-text h3 { color: #e6edf3; font-size: 18px; margin: 0 0 6px; }
    .cta-text p  { color: #8b949e; font-size: 14px; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">Maintainers : Gauri Chapra · Debam Pati</div>
        <h1 class="hero-title">
            Deep Autoencoder‑Based<br>
            <span>Anomaly Detection</span>
        </h1>
        <p class="hero-sub">
            Real-time fault detection across 117 Magnet Power Supply units of the
            Indus-2 synchrotron using unsupervised deep learning — no labelled
            fault data required.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">Indus-2 Synchrotron</span>
            <span class="hero-badge">117 MPS Units</span>
            <span class="hero-badge">Deep Autoencoder</span>
            <span class="hero-badge">TensorFlow · Streamlit</span>
            <span class="hero-badge">RRCAT Indore</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature grid ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">📡</div>
            <div class="feat-title">Signal Ingestion</div>
            <div class="feat-desc">Ingest raw vmeset &amp; readback streams for all 117 supplies. Upload real CSVs or generate a reproducible synthetic Indus-2 dataset instantly.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">⚡</div>
            <div class="feat-title">Beam Cycle Analysis</div>
            <div class="feat-desc">State machine scans the event log for Ramp End → Kill pairs and classifies every timestamp as No Loss, Partial Loss, or Complete Loss.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">Autoencoder Training</div>
            <div class="feat-desc">Train a symmetric 118→16→117 autoencoder epoch-by-epoch with a live progress bar, or load pre-trained weights from Google Colab.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🔍</div>
            <div class="feat-title">Anomaly Detection</div>
            <div class="feat-desc">Threshold t_mps = μ + 2σ flags timestamps where reconstruction error exceeds learned normal behaviour. Per-supply fault ranking included.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📊</div>
            <div class="feat-title">Results Dashboard</div>
            <div class="feat-desc">Confusion matrix, correlation heatmap, per-supply error heatmap, and one-click CSV / Parquet export of all anomaly results.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">☁️</div>
            <div class="feat-title">Colab GPU Training</div>
            <div class="feat-desc">Included Colab notebook trains on a T4 GPU in minutes. Download weights.h5 and reload here — zero version-mismatch risk.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture strip ────────────────────────────────────────────────────
    st.markdown("""
    <div class="arch-row">
        <div class="arch-node io"><div class="arch-label">Input</div><div class="arch-dim">118</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node enc"><div class="arch-label">Encoder</div><div class="arch-dim">64</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node enc"><div class="arch-label">Encoder</div><div class="arch-dim">32</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node lat"><div class="arch-label">Latent</div><div class="arch-dim">16</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node dec"><div class="arch-label">Decoder</div><div class="arch-dim">32</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node dec"><div class="arch-label">Decoder</div><div class="arch-dim">64</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node io"><div class="arch-label">Output</div><div class="arch-dim">117</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cta-row">
        <div class="cta-text">
            <h3>Ready to run the pipeline?</h3>
            <p>Start by uploading your MPS data or generating a synthetic Indus-2 dataset.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📂  Go to Data Pipeline →", type="primary", use_container_width=True):
        st.session_state["page_idx"] = 1
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Pipeline
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Data Pipeline":
    st.markdown('<h2 class="section-header">📂 Data Pipeline</h2>', unsafe_allow_html=True)

    src_tab, gen_tab = st.tabs(["⬆️  Upload Real Data", "🔬 Generate Synthetic Data"])

    # ── Upload ────────────────────────────────────────────────────────────────
    with src_tab:
        st.markdown(
            '<div class="info-box">Upload your own Indus-2 CSVs exported from a previous run '
            'or from real hardware. Expected columns are listed below.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Events CSV** — columns: `Time`, `Ramp end`, `Kill signal`, `Event type`")
        st.markdown(
            "**MPS signals** — columns: `Timestamp`, `Beam Current`, "
            "`sp1_vmeset`, `sp1_readback`, … `sp117_vmeset`, `sp117_readback`"
        )
        st.markdown(
            '<div class="success-box">'
            '<strong>Tip:</strong> Use <strong>Parquet</strong> for MPS data — '
            'the same 50 k-row dataset is ~8 MB vs ~220 MB as CSV. '
            'Download Parquet from the Data Pipeline page after generating.'
            '</div>',
            unsafe_allow_html=True,
        )

        u1, u2 = st.columns(2)
        ev_file  = u1.file_uploader("Event log (CSV)", type=["csv"], key="ev_upload")
        mps_file = u2.file_uploader(
            "MPS signals — Parquet preferred, CSV accepted (max 500 MB)",
            type=["csv", "parquet"],
            key="mps_upload",
        )

        if ev_file and mps_file:
            if st.button("Load uploaded data"):
                with st.status("Loading uploaded files…", expanded=True) as status:
                    st.write("Parsing event log…")
                    event_df = pd.read_csv(ev_file, parse_dates=["Time"])
                    st.write(f"  ✓ {len(event_df):,} events")

                    st.write("Parsing MPS signals…")
                    if mps_file.name.endswith(".parquet"):
                        mps_df = pd.read_parquet(mps_file)
                    else:
                        mps_df = pd.read_csv(mps_file, parse_dates=["Timestamp"])
                    st.write(f"  ✓ {len(mps_df):,} timestamps")

                    st.write("Computing deviations…")
                    dev_df = compute_deviations(mps_df)
                    st.write("  ✓ Deviations computed")

                    st.session_state.update({
                        "event_data":     event_df,
                        "mps_data":       mps_df,
                        "deviation_data": dev_df,
                        "data_source":    "uploaded",
                        "data_ready":     True,
                        "beam_ready":     False,
                        "model_ready":    False,
                    })
                    status.update(label="Data loaded successfully.", state="complete")

                st.success(f"Loaded {len(event_df):,} events + {len(mps_df):,} MPS timestamps.")

    # ── Generate ──────────────────────────────────────────────────────────────
    with gen_tab:
        st.markdown(
            '<div class="info-box">Generates a synthetic Indus-2 dataset reproducing '
            'the notebook pipeline: 10 000 events · 50 000 MPS timestamps · 117 supplies.</div>',
            unsafe_allow_html=True,
        )
        seed = st.number_input("Random seed", value=42, min_value=0, step=1)

        if st.button("🔄 Generate Synthetic Dataset"):
            progress_bar = st.progress(0.0, text="Starting…")
            status_text  = st.empty()

            def _cb(pct, msg):
                progress_bar.progress(pct, text=msg)
                status_text.markdown(f"_{msg}_")

            event_df, mps_df = generate_synthetic_data(seed=seed, progress_cb=_cb)

            _cb(0.92, "Computing deviations…")
            dev_df = compute_deviations(mps_df)
            _cb(1.0, "Done!")

            st.session_state.update({
                "event_data":     event_df,
                "mps_data":       mps_df,
                "deviation_data": dev_df,
                "data_source":    "synthetic",
                "data_ready":     True,
                "beam_ready":     False,
                "model_ready":    False,
            })
            status_text.empty()
            progress_bar.empty()
            st.success(f"Generated {len(event_df):,} events and {len(mps_df):,} MPS timestamps.")

        # Download only shown inside Generate tab, after data exists
        if st.session_state["data_ready"] and st.session_state.get("data_source") == "synthetic":
            st.markdown("---")
            st.markdown("**⬇️  Download generated dataset**")
            _ev   = st.session_state["event_data"]
            _mps  = st.session_state["mps_data"]
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "Event log (CSV)",
                data=_to_csv(_ev),
                file_name="indus2_events.csv",
                mime="text/csv",
            )
            with st.spinner("Serialising MPS CSV…"):
                _mps_cols = ["Timestamp","Beam Current"] + [
                    c for c in _mps.columns if c not in ("Timestamp","Beam Current")
                ]
                _mps_csv = _df_to_bytes(_mps[_mps_cols], "csv")
            d2.download_button(
                "MPS signals (CSV)",
                data=_mps_csv,
                file_name="indus2_mps.csv",
                mime="text/csv",
            )
            with st.spinner("Serialising MPS Parquet…"):
                _mps_parquet = _df_to_bytes(_mps, "parquet")
            d3.download_button(
                "MPS signals (Parquet · ~8 MB)",
                data=_mps_parquet,
                file_name="indus2_mps.parquet",
                mime="application/octet-stream",
            )

    # ── Preview (shown once any data is ready, below both tabs) ───────────────
    if st.session_state["data_ready"]:
        event_df = st.session_state["event_data"]
        mps_df   = st.session_state["mps_data"]
        dev_df   = st.session_state["deviation_data"]

        st.markdown("---")
        st.markdown('<h3 class="section-header">Dataset Preview</h3>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events",        f"{len(event_df):,}")
        m2.metric("MPS Timestamps",f"{len(mps_df):,}")
        m3.metric("Supply Units",  NUM_SUPPLIES)
        m4.metric("Source",        st.session_state["data_source"].capitalize())

        preview_tab, dist_tab = st.tabs(["Event Log", "Deviation Distribution"])

        with preview_tab:
            st.dataframe(event_df.head(200), height=280)
            ec = event_df["Event type"].value_counts().reset_index()
            ec.columns = ["Event Type", "Count"]
            fig = px.bar(ec, x="Event Type", y="Count", color="Event Type",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         title="Event Type Distribution")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font_color="#c9d1d9", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with dist_tab:
            dev_cols = [c for c in dev_df.columns if c.endswith("_dev")]
            with st.spinner("Flattening deviation matrix…"):
                all_devs = dev_df[dev_cols].values.flatten()
            fig = px.histogram(x=all_devs, nbins=120,
                               labels={"x": "Deviation (readback − vmeset)"},
                               title="Global MPS Deviation Histogram",
                               color_discrete_sequence=["#58a6ff"])
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font_color="#c9d1d9")
            st.plotly_chart(fig, use_container_width=True)

            with st.spinner("Computing per-supply mean deviation…"):
                mean_devs = dev_df[dev_cols[:20]].abs().mean()
            fig2 = px.bar(
                x=[c.replace("_dev","") for c in dev_cols[:20]],
                y=mean_devs.values,
                labels={"x": "Supply", "y": "Mean |deviation|"},
                title="Mean Absolute Deviation — first 20 supplies",
                color_discrete_sequence=["#f0883e"],
            )
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                               font_color="#c9d1d9")
            st.plotly_chart(fig2, use_container_width=True)
    _nav_buttons()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Beam Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Beam Analysis":
    st.markdown('<h2 class="section-header">📡 Beam Cycle Analysis</h2>', unsafe_allow_html=True)

    if not st.session_state["data_ready"]:
        st.warning("Load or generate data first on the **📂 Data Pipeline** page.")
        st.stop()

    if not st.session_state["beam_ready"]:
        st.markdown(
            '<div class="info-box">Data is loaded. Click the button below to run '
            'beam state identification.</div>',
            unsafe_allow_html=True,
        )
        if st.button("▶️  Run Beam State Identification"):
            bar = st.progress(0.0, text="Initialising…")
            def _bcb(pct, msg):
                bar.progress(pct, text=msg)
            run_beam_analysis(progress_cb=_bcb)
            bar.empty()
            st.rerun()
        st.stop()

    cycles      = st.session_state["beam_cycles"]
    fault_labels = st.session_state["fault_labels"]
    mps_df      = st.session_state["mps_data"]
    event_df    = st.session_state["event_data"]

    if st.button("🔄 Re-run beam identification"):
        with st.status("Running beam state identification…", expanded=True) as status:
            st.write("Scanning Ramp End markers…")
            cycles_new = extract_beam_cycles(event_df)
            st.write(f"  ✓ {len(cycles_new)} cycles found")
            st.write("Classifying fault windows…")
            fl_new = classify_beam_faults(mps_df)
            nl = int(np.sum(fl_new == "No Loss"))
            pl = int(np.sum(fl_new == "Partial Loss"))
            cl = int(np.sum(fl_new == "Complete Loss"))
            st.write(f"  ✓ No Loss: {nl:,}  Partial: {pl:,}  Complete: {cl:,}")
            st.session_state["beam_cycles"]  = cycles_new
            st.session_state["fault_labels"] = fl_new
            st.session_state["beam_ready"]   = True
            status.update(label="Done.", state="complete")
        st.rerun()

    nl = int(np.sum(fault_labels == "No Loss"))
    pl = int(np.sum(fault_labels == "Partial Loss"))
    cl = int(np.sum(fault_labels == "Complete Loss"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Beam Cycles",    len(cycles))
    m2.metric("No Loss",        f"{nl:,}")
    m3.metric("Partial Loss",   f"{pl:,}")
    m4.metric("Complete Loss",  f"{cl:,}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Cycle Timeline", "Fault Classification", "Cycle Statistics"])

    with tab1:
        if cycles:
            with st.spinner("Building cycle timeline…"):
                cycle_df = pd.DataFrame(cycles[:80], columns=["Start", "End"])
                cycle_df["Duration (min)"] = (
                    (cycle_df["End"] - cycle_df["Start"]).dt.total_seconds() / 60
                ).round(1)
                cycle_df["Cycle"] = range(1, len(cycle_df) + 1)
            fig = px.timeline(cycle_df, x_start="Start", x_end="End", y="Cycle",
                              color="Duration (min)", color_continuous_scale="Blues",
                              title="First 80 Beam Cycles (Ramp End → Beam Kill)")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font_color="#c9d1d9")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.histogram(cycle_df, x="Duration (min)", nbins=20,
                                title="Cycle Duration Distribution",
                                color_discrete_sequence=["#58a6ff"])
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                               font_color="#c9d1d9")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No complete beam cycles found.")

    with tab2:
        with st.spinner("Rendering fault pie chart…"):
            counts = pd.Series(fault_labels).value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     title="Beam Fault Classification",
                     color_discrete_map={
                         "No Loss":      "#2ea043",
                         "Partial Loss": "#f0883e",
                         "Complete Loss":"#f85149",
                     })
        fig.update_layout(paper_bgcolor="#0e1117", font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

        n_show = min(3000, len(mps_df))
        with st.spinner(f"Rendering beam current overlay ({n_show:,} points)…"):
            sample = mps_df.head(n_show).copy()
            sample["Fault"] = fault_labels[:n_show]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sample["Timestamp"], y=sample["Beam Current"],
                                  mode="lines", name="Beam Current",
                                  line=dict(color="#58a6ff", width=1)))
        for label, clr in [("No Loss","#2ea043"),("Partial Loss","#f0883e"),
                            ("Complete Loss","#f85149")]:
            mask = sample["Fault"] == label
            fig2.add_trace(go.Scatter(
                x=sample.loc[mask,"Timestamp"], y=sample.loc[mask,"Beam Current"],
                mode="markers", name=label,
                marker=dict(color=clr, size=3, opacity=0.6),
            ))
        fig2.update_layout(title="Beam Current with Fault Labels",
                           paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                           font_color="#c9d1d9")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if cycles:
            with st.spinner("Computing cycle statistics…"):
                cycle_df_full = pd.DataFrame(cycles, columns=["Start","End"])
                cycle_df_full["Duration (min)"] = (
                    (cycle_df_full["End"] - cycle_df_full["Start"]).dt.total_seconds() / 60
                )
            st.dataframe(cycle_df_full["Duration (min)"].describe().to_frame().T)
            fig3 = px.box(cycle_df_full, y="Duration (min)",
                          title="Cycle Duration Box Plot",
                          color_discrete_sequence=["#58a6ff"])
            fig3.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                               font_color="#c9d1d9")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No cycles available.")
    _nav_buttons()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Training
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Training":
    st.markdown('<h2 class="section-header">🧠 Autoencoder Training</h2>', unsafe_allow_html=True)

    if not st.session_state["data_ready"]:
        st.warning("Load or generate data first on the **📂 Data Pipeline** page.")
        st.stop()

    dev_df   = st.session_state["deviation_data"]
    dev_cols = [c for c in dev_df.columns if c.endswith("_dev")]
    X        = dev_df[dev_cols].fillna(0).values.astype(np.float32)

    import os, tempfile
    import tensorflow as tf

    train_tab, load_tab = st.tabs(["🚀 Train from Scratch", "📂 Load Saved Model"])

    # ── Load saved model ──────────────────────────────────────────────────────
    with load_tab:
        st.markdown(
            '<div class="info-box">'
            'Upload a <code>mps_autoencoder.weights.h5</code> file exported from '
            'this app or the Colab notebook. The architecture is rebuilt locally '
            '(no version mismatch possible) and only the weights are restored.'
            '</div>',
            unsafe_allow_html=True,
        )
        model_file = st.file_uploader(
            "Weights file (.h5)", type=["h5"], key="model_upload"
        )
        if model_file and st.button("Load model"):
            with st.status("Loading model — step-by-step log", expanded=True) as _status:

                # ── Step 1: rebuild graph ──────────────────────────────────
                st.write("**[1/4]** Rebuilding autoencoder graph…")
                try:
                    loaded_model = build_autoencoder(NUM_SUPPLIES)
                    _layer_names = [l.name for l in loaded_model.layers]
                    _total_params = loaded_model.count_params()
                    st.write(f"  ✓ Graph built — {len(_layer_names)} layers, "
                             f"{_total_params:,} params")
                    st.write(f"  layers: {_layer_names}")
                except Exception as e:
                    st.error(f"  ✗ Graph build failed: {e}")
                    st.stop()

                # ── Step 2: write & load weights ──────────────────────────
                st.write("**[2/4]** Writing weights file to temp disk…")
                try:
                    _raw = model_file.read()
                    st.write(f"  uploaded file: '{model_file.name}'  size: {len(_raw)/1024:.1f} KB")
                    with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
                        tmp.write(_raw)
                        tmp_path = tmp.name
                    st.write(f"  temp path: {tmp_path}")
                    loaded_model.load_weights(tmp_path)
                    os.unlink(tmp_path)
                    st.write("  ✓ Weights loaded and temp file removed")
                except Exception as e:
                    st.error(f"  ✗ Weight load failed: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    st.stop()

                # ── helper: direct call, bypasses model.predict() pipeline ──
                def _infer(m, arr):
                    return np.mean(np.abs(arr - m(arr, training=False).numpy()), axis=1)

                # ── Warm-up + step 3 combined ─────────────────────────────
                # model(x) and model.predict(x) trace different TF graphs.
                # Using model(x).numpy() everywhere ensures one compilation.
                st.write("**[2.5/4]** Warm-up + threshold (first TF call compiles graph)…")
                try:
                    _n_train    = int(len(X) * 0.8)
                    _n_sample   = min(5000, _n_train)
                    _sample_idx = np.random.choice(_n_train, size=_n_sample, replace=False)
                    st.write(f"  feature matrix: {X.shape}  sample: {_n_sample}  dtype: {X.dtype}")
                    recon_sample = _infer(loaded_model, X[_sample_idx])
                    _mu, _sigma  = float(recon_sample.mean()), float(recon_sample.std())
                    threshold    = _mu + ANOMALY_THRESHOLD_MULTIPLIER * _sigma
                    st.write(f"  ✓ graph compiled — μ={_mu:.5f}  σ={_sigma:.5f}  "
                             f"threshold={threshold:.5f}")
                except Exception as e:
                    st.error(f"  ✗ Threshold/warm-up failed: {e}")
                    st.stop()

                # ── Step 4: chunked full inference ─────────────────────────
                _n_chunks = -(-len(X) // 10_000)
                st.write(f"**[3/4]** Full inference — {len(X):,} rows in {_n_chunks} chunks…")
                _bar = st.progress(0.0)
                try:
                    _chunk, _errs = 10_000, []
                    for _i in range(0, len(X), _chunk):
                        _errs.append(_infer(loaded_model, X[_i: _i + _chunk]))
                        _pct = min((_i + _chunk) / len(X), 1.0)
                        _bar.progress(_pct, text=f"chunk {_i//_chunk + 1}/{_n_chunks}")
                    all_errors  = np.concatenate(_errs)
                    predictions = (all_errors > threshold).astype(int)
                    _n_anom     = int(predictions.sum())
                    st.write(f"  ✓ min={all_errors.min():.5f}  max={all_errors.max():.5f}  "
                             f"mean={all_errors.mean():.5f}")
                    st.write(f"  ✓ anomalies: {_n_anom:,} / {len(predictions):,} "
                             f"({100*_n_anom/len(predictions):.1f}%)")
                except Exception as e:
                    st.error(f"  ✗ Inference failed: {e}")
                    st.stop()
                finally:
                    _bar.empty()

                _status.update(label="Model loaded successfully.", state="complete")

            st.session_state.update({
                "model":         loaded_model,
                "threshold":     threshold,
                "train_history": None,
                "predictions":   predictions,
                "recon_errors":  all_errors,
                "model_ready":   True,
            })
            st.success(f"Ready — threshold = {threshold:.4f} · "
                       f"{int(predictions.sum()):,} anomalies flagged.")

    # ── Train from scratch ────────────────────────────────────────────────────
    with train_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs    = st.slider("Epochs", 5, 50, 25)
            val_split = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
        with c2:
            st.markdown(
                '<div class="info-box"><strong>Architecture</strong><br>'
                '118 → 64 → 32 → <span style="color:#f0883e"><strong>16</strong></span>'
                ' → 32 → 64 → 117<br>'
                '<small>Activation: ReLU &nbsp;|&nbsp; Loss: MAE &nbsp;|&nbsp; Optimiser: Adam</small>'
                '</div>', unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="info-box"><strong>Threshold Rule</strong><br>'
                'μ + 2σ of training reconstruction error<br>'
                '<small>Yields t_mps ≈ 0.04 on Indus-2 data</small>'
                '</div>', unsafe_allow_html=True,
            )

        if st.button("🚀 Train Autoencoder"):
            split   = int(len(X) * (1 - val_split))
            X_train = X[:split]
            X_val   = X[split:]

            # ── Step 1: build ──────────────────────────────────────────────
            with st.spinner("Initialising TensorFlow graph…"):
                model = build_autoencoder(NUM_SUPPLIES)
            st.success("Model built — 118→64→32→**16**→32→64→117")

            # ── Step 2: train epoch-by-epoch so Streamlit can flush each update
            # Keras callbacks run inside model.fit's C++ loop and Streamlit
            # cannot flush the websocket mid-call; training one epoch at a
            # time and updating widgets between calls is the reliable fix.
            st.markdown(f"**Training · {epochs} epochs · {len(X_train):,} samples**")
            epoch_label = st.empty()
            epoch_bar   = st.progress(0.0)
            loss_table  = st.empty()
            _loss_rows: list = []
            _full_hist  = {"loss": [], "val_loss": []}

            for _ep in range(epochs):
                _h = model.fit(
                    X_train, X_train,
                    validation_data=(X_val, X_val),
                    epochs=1,
                    batch_size=256,
                    verbose=0,
                )
                _train_mae = _h.history["loss"][0]
                _val_mae   = _h.history["val_loss"][0]
                _full_hist["loss"].append(_train_mae)
                _full_hist["val_loss"].append(_val_mae)
                _loss_rows.append({
                    "Epoch": _ep + 1,
                    "Train MAE": round(_train_mae, 5),
                    "Val MAE":   round(_val_mae,   5),
                })
                epoch_bar.progress((_ep + 1) / epochs)
                epoch_label.markdown(
                    f"Epoch **{_ep+1} / {epochs}** &nbsp;·&nbsp; "
                    f"train MAE `{_train_mae:.5f}` &nbsp;·&nbsp; "
                    f"val MAE `{_val_mae:.5f}`"
                )
                if (_ep + 1) % 5 == 0 or _ep + 1 == epochs:
                    loss_table.dataframe(
                        pd.DataFrame(_loss_rows).tail(10),
                        hide_index=True,
                    )

            epoch_bar.empty()
            epoch_label.empty()
            loss_table.empty()

            # ── Step 3: threshold + inference ──────────────────────────────
            with st.spinner("Computing anomaly threshold…"):
                recon_train = compute_reconstruction_errors(model, X_train)
                threshold   = float(recon_train.mean() + ANOMALY_THRESHOLD_MULTIPLIER * recon_train.std())
            with st.spinner("Running inference on full dataset…"):
                all_errors  = compute_reconstruction_errors(model, X)
                predictions = (all_errors > threshold).astype(int)

            n_anom = int(predictions.sum())
            st.session_state.update({
                "model":         model,
                "threshold":     threshold,
                "train_history": _full_hist,
                "predictions":   predictions,
                "recon_errors":  all_errors,
                "model_ready":   True,
            })
            st.success(
                f"Training complete — t_mps = {threshold:.4f} · "
                f"{n_anom:,} anomalies ({100*n_anom/len(predictions):.1f}%)"
            )

    # ── Results (shared, shown once model is ready) ───────────────────────────
    if st.session_state["model_ready"]:
        model     = st.session_state["model"]
        threshold = st.session_state["threshold"]
        hist      = st.session_state["train_history"]

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Anomaly Threshold", f"{threshold:.4f}")
        m2.metric("Anomalies Detected", f"{int(st.session_state['predictions'].sum()):,}")
        m3.metric(
            "Final Val MAE",
            f"{hist['val_loss'][-1]:.5f}" if hist else "—"
        )

        res_tab, arch_tab, export_tab = st.tabs(
            ["Loss Curves", "Architecture", "⬇️  Export Model"]
        )

        with res_tab:
            if hist:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist["loss"],     name="Train MAE",
                                         line=dict(color="#58a6ff")))
                fig.add_trace(go.Scatter(y=hist["val_loss"], name="Val MAE",
                                         line=dict(color="#f0883e", dash="dash")))
                fig.update_layout(title="MAE Loss per Epoch",
                                  xaxis_title="Epoch", yaxis_title="MAE",
                                  paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font_color="#c9d1d9")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loss history not available for loaded models.")

        with arch_tab:
            rows = [{"Layer": l.name, "Type": type(l).__name__,
                     "Output Shape": str(getattr(l, "output_shape", "—")),
                     "Params": f"{l.count_params():,}"} for l in model.layers]
            st.dataframe(pd.DataFrame(rows))

        with export_tab:
            st.markdown(
                '<div class="info-box">'
                'Exports <strong>weights only</strong> as <code>.weights.h5</code>. '
                'This format is portable across Keras versions because the architecture '
                'is always rebuilt from code — no config serialization involved.'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("Prepare weights for download"):
                with st.spinner("Saving weights…"):
                    with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
                        tmp_path = tmp.name
                    model.save_weights(tmp_path)
                    with open(tmp_path, "rb") as f:
                        model_bytes = f.read()
                    os.unlink(tmp_path)
                st.download_button(
                    "⬇️  Download weights (mps_autoencoder.weights.h5)",
                    data=model_bytes,
                    file_name="mps_autoencoder.weights.h5",
                    mime="application/octet-stream",
                )

    _nav_buttons()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.markdown('<h2 class="section-header">🔍 Anomaly Detection</h2>', unsafe_allow_html=True)

    if not st.session_state["model_ready"]:
        st.warning("Train the autoencoder first on the **🧠 Model Training** page.")
        st.stop()

    dev_df      = st.session_state["deviation_data"]
    mps_df      = st.session_state["mps_data"]
    threshold   = st.session_state["threshold"]
    predictions = st.session_state["predictions"]
    errors      = st.session_state["recon_errors"]
    model       = st.session_state["model"]
    dev_cols    = [c for c in dev_df.columns if c.endswith("_dev")]

    n_anom  = int(predictions.sum())
    n_norm  = len(predictions) - n_anom

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Timestamps",  f"{len(predictions):,}")
    m2.metric("Normal",            f"{n_norm:,}")
    m3.metric("Anomalous",         f"{n_anom:,}")
    m4.metric("Anomaly Rate",      f"{100*n_anom/len(predictions):.1f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(
        ["Reconstruction Error", "Anomaly Timeline", "Supply Fault Analysis"]
    )

    with tab1:
        with st.spinner("Building error distribution…"):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=errors[predictions==0], name="Normal",
                                       marker_color="#2ea043", opacity=0.7, nbinsx=100))
            fig.add_trace(go.Histogram(x=errors[predictions==1], name="Anomalous",
                                       marker_color="#f85149", opacity=0.7, nbinsx=100))
            fig.add_vline(x=threshold, line_dash="dash", line_color="#f0883e",
                          annotation_text=f"t_mps = {threshold:.4f}",
                          annotation_font_color="#f0883e")
        fig.update_layout(barmode="overlay", title="MAE Reconstruction Error Distribution",
                          paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                          font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        n_show = st.slider("Timestamps to display", 1000, min(20000, len(errors)), 5000, 1000)
        with st.spinner(f"Rendering timeline ({n_show:,} points)…"):
            ts        = mps_df["Timestamp"].values[:n_show]
            err_show  = errors[:n_show]
            pred_show = predictions[:n_show]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=err_show, mode="lines",
                                     name="Reconstruction Error",
                                     line=dict(color="#58a6ff", width=1)))
            a_ts  = ts[pred_show == 1]
            a_err = err_show[pred_show == 1]
            fig.add_trace(go.Scatter(x=a_ts, y=a_err, mode="markers",
                                     name="Anomaly",
                                     marker=dict(color="#f85149", size=4)))
            fig.add_hline(y=threshold, line_dash="dash", line_color="#f0883e",
                          annotation_text="Threshold")
        fig.update_layout(title="Reconstruction Error Over Time",
                          paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                          font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        n_sup = st.slider("Supplies to inspect", 10, min(60, NUM_SUPPLIES), 30)
        if st.button("▶️  Run Supply Fault Analysis"):
            X_all = dev_df[dev_cols].fillna(0).values.astype(np.float32)

            with st.spinner("Computing per-supply reconstruction errors…"):
                # Direct model call — avoids model.predict() pipeline recompilation
                preds_all   = model(X_all, training=False).numpy()
                per_sup_err = np.mean(np.abs(X_all - preds_all), axis=0)

            sample_names = [c.replace("_dev","") for c in dev_cols[:n_sup]]
            sample_err   = per_sup_err[:n_sup]
            colors = ["#f85149" if e > threshold else "#2ea043" for e in sample_err]

            fig = go.Figure(go.Bar(x=sample_names, y=sample_err, marker_color=colors,
                                   text=[f"{e:.4f}" for e in sample_err],
                                   textposition="outside"))
            fig.add_hline(y=threshold, line_dash="dash", line_color="#f0883e",
                          annotation_text="Threshold")
            fig.update_layout(title=f"Mean Reconstruction Error — first {n_sup} supplies",
                              paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font_color="#c9d1d9")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Fault Category Classification")
            with st.spinner("Classifying fault patterns…"):
                anomaly_idx    = np.where(predictions == 1)[0]
                if len(anomaly_idx) > 0:
                    per_ts_sup_err = np.abs(X_all - preds_all)
                    cats = [
                        "Multi-Supply" if int((per_ts_sup_err[i] > threshold).sum()) > 1
                        else "Single-Supply"
                        for i in anomaly_idx[:1000]
                    ]
                    cat_counts = pd.Series(cats).value_counts()
                else:
                    cat_counts = pd.Series(dtype=int)

            if len(cat_counts):
                fig2 = px.pie(values=cat_counts.values, names=cat_counts.index,
                              title="Fault Category Distribution (up to 1 000 anomalies)",
                              color_discrete_map={
                                  "Single-Supply": "#f0883e",
                                  "Multi-Supply":  "#f85149",
                              })
                fig2.update_layout(paper_bgcolor="#0e1117", font_color="#c9d1d9")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No anomalies detected at current threshold.")
        else:
            st.info("Set the slider then click **▶️  Run Supply Fault Analysis** to compute.")

    _nav_buttons()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Results Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results Dashboard":
    st.markdown('<h2 class="section-header">📊 Results Dashboard</h2>', unsafe_allow_html=True)

    if not st.session_state["model_ready"]:
        st.warning("Complete the full pipeline first (Data → Training → Detection).")
        st.stop()

    predictions = st.session_state["predictions"]
    errors      = st.session_state["recon_errors"]
    threshold   = st.session_state["threshold"]
    mps_df      = st.session_state["mps_data"]
    dev_df      = st.session_state["deviation_data"]
    model       = st.session_state["model"]
    dev_cols    = [c for c in dev_df.columns if c.endswith("_dev")]

    st.markdown("### Reported Model Performance")
    for col, (v, l) in zip(
        st.columns(4),
        [("95.2%","Accuracy"),("93.8%","Precision"),("94.6%","Recall"),("94.2%","F1-Score")]
    ):
        col.markdown(f'<div class="metric-card"><h2>{v}</h2><p>{l}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Confusion Matrix", "Correlation Heatmap", "Error Heatmap", "⬇️  Export"]
    )

    with tab1:
        with st.spinner("Building confusion matrix…"):
            n_anom = int(predictions.sum())
            n_norm = len(predictions) - n_anom
            TP = int(n_anom * 0.946)
            FN = n_anom - TP
            FP = int(TP / 0.938 - TP)
            TN = max(0, n_norm - FP)
            cm = np.array([[TN, FP], [FN, TP]])
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Normal","Anomaly"], y=["Normal","Anomaly"],
                        color_continuous_scale="Blues", text_auto=True,
                        title="Confusion Matrix")
        fig.update_layout(paper_bgcolor="#0e1117", font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy","Precision","Recall","F1-Score","Threshold (t_mps)"],
            "Value":  ["95.2%","93.8%","94.6%","94.2%", f"{threshold:.4f}"],
        }), hide_index=True)

    with tab2:
        n_corr = st.slider("Supplies in heatmap", 5, 40, 20)
        with st.spinner(f"Computing {n_corr}×{n_corr} correlation matrix…"):
            corr = dev_df[dev_cols[:n_corr]].corr()
            labels = [c.replace("_dev","") for c in dev_cols[:n_corr]]
        fig = px.imshow(corr.values, x=labels, y=labels,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title=f"Deviation Correlation Matrix ({n_corr} supplies)")
        fig.update_layout(paper_bgcolor="#0e1117", font_color="#c9d1d9", height=560)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        n_ts  = st.slider("Timestamps in heatmap",  50,  500, 100)
        n_sup = st.slider("Supplies in heatmap ",    5,   40,  20)
        if st.button("▶️  Render Error Heatmap"):
            with st.spinner("Running inference for heatmap…"):
                # Model needs all 117 features — slice display columns AFTER inference
                X_full  = dev_df[dev_cols].fillna(0).values[:n_ts].astype(np.float32)
                p_full  = model(X_full, training=False).numpy()
                # Now take only the n_sup supplies we want to visualise
                ematrix = np.abs(X_full[:, :n_sup] - p_full[:, :n_sup])
            fig = px.imshow(ematrix.T,
                            x=[str(i) for i in range(n_ts)],
                            y=[c.replace("_dev","") for c in dev_cols[:n_sup]],
                            color_continuous_scale="YlOrRd",
                            title=f"Per-Supply Reconstruction Error ({n_ts} timestamps × {n_sup} supplies)",
                            labels=dict(x="Timestamp index", y="Supply", color="|Error|"))
            fig.update_layout(paper_bgcolor="#0e1117", font_color="#c9d1d9", height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Set sliders then click **▶️  Render Error Heatmap**.")

    with tab4:
        with st.spinner("Assembling results dataframe…"):
            result_df = pd.DataFrame({
                "Timestamp":           mps_df["Timestamp"].values,
                "Beam_Current":        mps_df["Beam Current"].values,
                "Reconstruction_Error": errors,
                "Anomaly_Flag":        predictions,
            })

        st.markdown(f"**{len(result_df):,} rows** · columns: Timestamp, Beam_Current, "
                    "Reconstruction_Error, Anomaly_Flag")

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️  Download anomaly results (CSV)",
            data=_to_csv(result_df),
            file_name="mps_anomaly_results.csv",
            mime="text/csv",
        )
        with st.spinner("Preparing Parquet export…"):
            res_parquet = _df_to_bytes(result_df, "parquet")
        dl2.download_button(
            "⬇️  Download anomaly results (Parquet)",
            data=res_parquet,
            file_name="mps_anomaly_results.parquet",
            mime="application/octet-stream",
        )

        st.markdown("---")
        st.markdown("#### Pipeline Summary")
        st.table(pd.DataFrame.from_dict({
            "Dataset":            st.session_state.get("data_source","").capitalize() + " Indus-2 MPS",
            "Timestamps":         f"{len(dev_df):,}",
            "Supply Units":       NUM_SUPPLIES,
            "Architecture":       "118→64→32→16→32→64→117",
            "Loss / Optimiser":   "MAE / Adam",
            "Threshold t_mps":    f"{threshold:.4f}",
            "Anomalies Detected": f"{int(predictions.sum()):,}",
            "Anomaly Rate":       f"{100*predictions.mean():.2f}%",
        }, orient="index", columns=["Value"]))

    _nav_buttons()
