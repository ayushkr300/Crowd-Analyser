import streamlit as st
import pandas as pd
from datetime import datetime
import os
import tempfile
import cv2
import numpy as np
import time
import threading

from crowd_analyzer import get_analyzer
from stream_monitor import StreamMonitor, AlertConfig

# ── Directory Setup ─────────────────────────────────────────────────────────[...]
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Page config ──────────────────────────────────────────────────────────[...]
st.set_page_config(
    page_title="Crowd Density Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────[...]
st.markdown("""
<style>
/* ---- Tokens ---- */
:root {
    --primary:  #2563EB;
    --success:  #10B981;
    --warning:  #F59E0B;
    --danger:   #EF4444;
    --bg:       #F1F5F9;
    --card:     #FFFFFF;
    --border:   #E2E8F0;
    --text:     #1E293B;
    --muted:    #64748B;
    --radius:   12px;
    --shadow:   0 1px 3px rgba(0,0,0,.07), 0 4px 12px rgba(0,0,0,.05);
}

/* ---- Base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}

/* Fixed Top Padding to stop Title Hiding */
.block-container {
    padding: 3.5rem 2.5rem 2rem 2.5rem !important; 
    max-width: 100% !important;
}

/* ---- Page header ---- */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.app-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -.4px;
    margin: 0;
    line-height: 1;
}
.app-badge {
    font-size: .68rem;
    font-weight: 600;
    color: var(--primary);
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 2px 9px;
    text-transform: uppercase;
    letter-spacing: .5px;
    white-space: nowrap;
}

/* ---- Section label ---- */
.sec-label {
    font-size: .7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
    color: var(--muted);
    margin: 0 0 .6rem 0;
}

/* ---- Image Fixes (Cap preview image height) ---- */
[data-testid="stImage"] img {
    max-height: 350px !important;
    width: 100%;
    object-fit: contain;
}

/* ---- KPI row ---- */
.kpi-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1rem;
}
.kpi-card {
    flex: 1;
    background: var(--card);
    border-radius: var(--radius);
    padding: .9rem 1rem;
    box-shadow: var(--shadow);
    min-width: 0;
    border: none !important;
}

.kpi-meta  { font-size: .68rem; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: .3rem; }
.kpi-value { font-size: 1.55rem; font-weight: 700; line-height: 1; color: var(--text); }
.kpi-sub   { font-size: .75rem; color: var(--muted); margin-top: .25rem; }

/* ---- Alert banner ---- */
.alert-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: .6rem .9rem;
    border-radius: 8px;
    font-size: .83rem;
    font-weight: 500;
    margin-bottom: .9rem;
}
.alert-banner.success { background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; }
.alert-banner.info    { background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E40AF; }
.alert-banner.warning { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
.alert-banner.danger  { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }

/* ---- Density meter ---- */
.meter-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .9rem 1.1rem 1rem;
    box-shadow: var(--shadow);
    margin-bottom: .9rem;
}
.meter-track {
    position: relative;
    height: 22px;
    display: flex;
    border-radius: 11px;
    overflow: visible;
    margin: .5rem 0 1.4rem 0;
}
.meter-seg { height: 100%; }
.meter-seg:first-child { border-radius: 11px 0 0 11px; }
.meter-seg:last-child  { border-radius: 0 11px 11px 0; }
.meter-ptr {
    position: absolute;
    top: -5px;
    width: 3px;
    height: 32px;
    background: var(--text);
    border-radius: 2px;
    transform: translateX(-50%);
    z-index: 5;
}
.meter-ptr::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid var(--text);
}
.meter-labels {
    display: flex;
    font-size: .65rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .3px;
}
.meter-labels span { flex: 1; text-align: center; }

/* ---- Image cards (results) ---- */
.img-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.img-card-head {
    padding: .5rem .9rem;
    border-bottom: 1px solid var(--border);
    background: #FAFAFA;
    font-size: .72rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .5px;
}

/* ---- Summary grid ---- */
.summary-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow);
    margin-bottom: .9rem;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .6rem .9rem;
}
.s-key   { font-size: .67rem; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); }
.s-val   { font-size: .88rem; font-weight: 600; color: var(--text); margin-top: 2px; }

/* ---- Risk pill ---- */
.rpill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .4px;
}
.rpill.LOW      { background: #ECFDF5; color: #065F46; }
.rpill.MODERATE { background: #FFFBEB; color: #92400E; }
.rpill.HIGH     { background: #FFF7ED; color: #9A3412; }
.rpill.CRITICAL { background: #FEF2F2; color: #991B1B; }

/* ---- Status indicator ---- */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 600;
}
.status-badge.live { background: #EF4444; color: white; }
.status-badge.paused { background: #F59E0B; color: white; }
.status-badge.offline { background: #6B7280; color: white; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

/* ---- Success toast ---- */
.toast {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 10px;
    padding: .7rem 1rem;
    font-size: .83rem;
    color: #065F46;
    margin-bottom: .9rem;
}
.toast strong { display: block; font-size: .88rem; margin-bottom: .2rem; }

/* ---- Divider ---- */
.div { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* ---- Download buttons ---- */
[data-testid="stDownloadButton"] button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: background .15s;
}
[data-testid="stDownloadButton"] button:hover {
    background: #F1F5F9 !important;
}

/* ---- Primary button ---- */
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.sb-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1.25rem;
}
.sb-label {
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .6px;
    color: var(--muted);
    margin-bottom: .4rem;
}

/* Custom styling for the mode toggle */
div[data-testid="stRadio"] > div {
    gap: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────[...]
RISK_META = {
    "LOW":      {"cls": "success", "dot": "#10B981", "msg": "Crowd density is within safe limits."},
    "MODERATE": {"cls": "info",    "dot": "#3B82F6", "msg": "Crowd density is MODERATE. Monitor the situation."},
    "HIGH":     {"cls": "warning", "dot": "#F59E0B", "msg": "Crowd density is HIGH. Consider crowd management measures."},
    "CRITICAL": {"cls": "danger",  "dot": "#EF4444", "msg": "CRITICAL: Crowd density is dangerous. Immediate action required."},
}
KPI_RISK_CLASS = {"LOW": "c-success", "MODERATE": "c-warning", "HIGH": "c-warning", "CRITICAL": "c-danger"}

# ── HTML helpers ───────────────────────────────────────────────────────────[...]
def kpi_html(meta_label, value, sub, cls):
    return f"""
    <div class="kpi-card {cls}">
        <div class="kpi-meta">{meta_label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def density_meter_html(density):
    pct = min(density / 8.0 * 100, 100)
    segs = [("25%","#10B981"),("25%","#F59E0B"),("25%","#F97316"),("25%","#EF4444")]
    seg_html = "".join(
        f'<div class="meter-seg" style="width:{w};background:{c}"></div>' for w,c in segs
    )
    return f"""
    <div class="meter-wrap">
        <div class="sec-label">Density Level</div>
        <div class="meter-track">
            {seg_html}
            <div class="meter-ptr" style="left:{pct:.1f}%"></div>
        </div>
        <div class="meter-labels">
            <span>Safe &lt;2</span>
            <span>Moderate 2–4</span>
            <span>High 4–6</span>
            <span>Critical &gt;6</span>
        </div>
    </div>"""

def summary_html(results, fname, area):
    items = [
        ("Image",      fname),
        ("Resolution", f"{results['image_resolution'][0]} x {results['image_resolution'][1]} px"),
        ("Count",      f"{results['count']} people"),
        ("Area",       f"{area:.1f} m\u00b2"),
        ("Density",    f"{results['density']:.2f} persons/m\u00b2"),
        ("Risk",       f'<span class="rpill {results["risk"]}">{results["risk"]}</span>'),
        ("Inference",  f"{results['inference_time']:.2f} s"),
        ("Analysed",   datetime.now().strftime("%d %b %Y, %H:%M")),
    ]
    cells = "".join(
        f'<div><div class="s-key">{k}</div><div class="s-val">{v}</div></div>'
        for k, v in items
    )
    return f'<div class="summary-card"><div class="sec-label" style="margin-bottom:.7rem">Analysis Summary</div><div class="summary-grid">{cells}</div></div>'

# ── Model ──────────────────────────────────────────────────────────────[...]
@st.cache_resource(show_spinner=False)
def load_analyzer():
    return get_analyzer()

# ── Session state ──────────────────────────────────────────────────────────[...]
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "vid_analysis_results" not in st.session_state:
    st.session_state.vid_analysis_results = None
if "stream_monitor" not in st.session_state:
    st.session_state.stream_monitor = None
if "stream_stats_history" not in st.session_state:
    st.session_state.stream_stats_history = []

# ── Sidebar ────────────────────────────────────────────────────────────[...]
with st.sidebar:
    st.markdown('<div class="sb-title">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Visible Area</div>', unsafe_allow_html=True)
    area = st.number_input(
        "Area (m\u00b2)",
        min_value=1.0, max_value=10000.0, value=300.0, step=10.0,
        label_visibility="collapsed",
        help="Approximate real-world area visible in the image/video (square metres).",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Risk Level Guide"):
        st.markdown("""
        **LOW** — < 2 persons/m² — Safe  
        **MODERATE** — 2–4 — Normal  
        **HIGH** — 4–6 — Crowded  
        **CRITICAL** — > 6 — Dangerous
        """)

    with st.expander("How to use"):
        st.markdown("""
        **Image Mode:**
        1. Upload an image
        2. Click **Analyse Image**
        
        **Video Mode:**
        1. Upload video
        2. Set sampling rate
        3. Click **Analyse Video**
        
        **Stream Mode:**
        1. Enter stream URL (RTSP/HTTP)
        2. Configure alerts
        3. Click **Start Monitoring**
        """)

# ── Page header ────────────────────────────────────────────────────────────[...]
st.markdown("""
<div class="app-header">
    <span class="app-title">📊 Crowd Density Monitoring System</span>
</div>
""", unsafe_allow_html=True)

# Load model silently
with st.spinner("Loading model…"):
    analyzer = load_analyzer()

# ── MODE TOGGLE ────────────────────────────────────────────────────────────[...]
app_mode = st.radio(
    "Select Analysis Mode",
    ["🖼️ Image Analysis", "🎥 Video Analysis", "📡 Live Stream Monitor"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("<br>", unsafe_allow_html=True)

# ── IMAGE ANALYSIS ─────────────────────────────────────────────────────────[...]
if app_mode == "🖼️ Image Analysis":
    left_col, right_col = st.columns([3, 7], gap="medium")

    with left_col:
        st.markdown('<div class="sec-label">Upload Image</div>', unsafe_allow_html=True)
        uploaded_img = st.file_uploader(
            "upload_image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_img:
            st.session_state.uploaded_img = uploaded_img

        if st.session_state.uploaded_img:
            st.markdown(
                f"<p style='font-size:.78rem;color:var(--muted);margin:.3rem 0 .6rem'>{st.session_state.uploaded_img.name}</p>",
                unsafe_allow_html=True,
            )
            analyse_img_clicked = st.button("Analyse Image", type="primary", use_container_width=True)
        else:
            st.markdown(
                "<p style='font-size:.78rem;color:var(--muted);margin:.4rem 0 0'>JPG, JPEG or PNG</p>",
                unsafe_allow_html=True,
            )
            analyse_img_clicked = False

    with right_col:
        if st.session_state.uploaded_img:
            st.markdown('<div class="sec-label">Preview</div>', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_img, use_container_width=True)
        else:
            st.markdown("""
            <div style="height:180px;background:var(--card);border:2px dashed var(--border);
                        border-radius:var(--radius);display:flex;align-items:center;
                        justify-content:center;color:var(--muted);font-size:.85rem;">
                Image preview will appear here
            </div>
            """, unsafe_allow_html=True)

    if analyse_img_clicked and st.session_state.uploaded_img:
        with st.spinner("Running crowd detection…"):
            upload_path = os.path.join(UPLOAD_DIR, st.session_state.uploaded_img.name)
            with open(upload_path, "wb") as f:
                f.write(st.session_state.uploaded_img.getvalue())
                
            try:
                results = analyzer.predict(upload_path, area)
                st.session_state.analysis_results = results
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        risk    = results["risk"]
        rm      = RISK_META[risk]
        rc      = KPI_RISK_CLASS[risk]
        fname   = st.session_state.uploaded_img.name if st.session_state.uploaded_img else "Unknown"

        output_img_path = os.path.join(OUTPUT_DIR, f"annotated_{fname}")
        cv2.imwrite(output_img_path, results["annotated_image"])

        csv_df = pd.DataFrame([{
            "Image Name":        fname,
            "Crowd Count":       results["count"],
            "Area (m2)":         area,
            "Density (p/m2)":    results["density"],
            "Risk Level":        risk,
            "Inference Time (s)":results["inference_time"],
            "Timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])
        
        csv_path = os.path.join(OUTPUT_DIR, f"crowd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        csv_df.to_csv(csv_path, index=False)

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="toast">
            <strong>Analysis Complete</strong>
            {results['count']} people detected &nbsp;&middot;&nbsp;
            Density: {results['density']:.2f} persons/m\u00b2 &nbsp;&middot;&nbsp;
            Risk: {risk}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-row">
            {kpi_html("Crowd Count",     results['count'],                   "People detected",  "c-primary")}
            {kpi_html("Density",         f"{results['density']:.2f}",        "persons / m\u00b2", rc)}
            {kpi_html("Risk Level",      risk,                               rm['msg'][:38]+"…" if len(rm['msg'])>38 else rm['msg'], rc)}
            {kpi_html("Inference Time",  f"{results['inference_time']:.2f}s","Model processing", "c-primary")}
        </div>""", unsafe_allow_html=True)

        st.markdown(
            f'<div class="alert-banner {rm["cls"]}"><span style="width:8px;height:8px;border-radius:50%;'
            f'background:{rm["dot"]};display:inline-block;flex-shrink:0"></span>{rm["msg"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(density_meter_html(results["density"]), unsafe_allow_html=True)

        ic1, ic2 = st.columns(2, gap="medium")
        with ic1:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.markdown('<div class="img-card-head">Original Image</div>', unsafe_allow_html=True)
            st.image(results["original_image"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ic2:
            annotated_rgb = cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB)
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="img-card-head">Annotated — {results["count"]} detected</div>',
                unsafe_allow_html=True,
            )
            st.image(annotated_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown(summary_html(results, fname, area), unsafe_allow_html=True)
        st.markdown('<hr class="div">', unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-bottom:.5rem">Downloads</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2, gap="medium")

        with dl1:
            st.download_button(
                label="Download Annotated Image",
                data=cv2.imencode(".jpg", results["annotated_image"])[1].tobytes(),
                file_name=f"annotated_{fname}",
                mime="image/jpeg",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                label="Download CSV Report",
                data=csv_df.to_csv(index=False).encode("utf-8"),
                file_name=f"crowd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ── VIDEO ANALYSIS ─────────────────────────────────────────────────────────[...]
elif app_mode == "🎥 Video Analysis":
    col1, col2 = st.columns([1, 2], gap="medium")
    
    with col1:
        st.markdown('<div class="sec-label">Upload Video</div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("upload_vid", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
        
        st.markdown('<div class="sec-label" style="margin-top:1rem;">Sampling Rate</div>', unsafe_allow_html=True)
        sample_rate_sec = st.number_input(
            "Process 1 frame every X seconds", 
            min_value=0.1, max_value=60.0, value=1.0, step=0.5, 
            help="Lower values track closer but process much slower."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_vid_clicked = st.button("Analyse Video", type="primary", use_container_width=True)
        
    with col2:
        if uploaded_video:
            st.info(f"Video ready: **{uploaded_video.name}**")
        else:
            st.markdown("""
            <div style="height:120px;background:var(--card);border:2px dashed var(--border);
                        border-radius:var(--radius);display:flex;align-items:center;
                        justify-content:center;color:var(--muted);font-size:.85rem;">
                Video upload status will appear here
            </div>
            """, unsafe_allow_html=True)

    if analyse_vid_clicked and uploaded_video:
        with st.spinner("Initialising video processing..."):
            vid_upload_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
            with open(vid_upload_path, "wb") as f:
                f.write(uploaded_video.getvalue())
                
        cap = cv2.VideoCapture(vid_upload_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0: fps = 30
        frame_skip = max(1, int(fps * sample_rate_sec))
        frames_to_process = (total_frames + frame_skip - 1) // frame_skip
        
        results_data = []
        
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Processing Status</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown('<br><div class="sec-label">Live Frame Analysis Preview</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % frame_skip == 0:
                    time_sec = frame_idx / fps
                    current_processed_step = (frame_idx // frame_skip) + 1
                    
                    status_text.markdown(f"**Analyzing frame {current_processed_step} of {frames_to_process}** (Video time: {time_sec:.1f}s)")
                    
                    fd_img, tmp_img_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd_img)
                    
                    cv2.imwrite(tmp_img_path, frame)
                    try:
                        res = analyzer.predict(tmp_img_path, area)
                        results_data.append({
                            "Time (s)": round(time_sec, 2),
                            "Frame": frame_idx,
                            "Count": res["count"],
                            "Density (p/m2)": round(res["density"], 2),
                            "Risk Level": res["risk"]
                        })
                        
                        annotated_image = res["annotated_image"]
                        if hasattr(annotated_image, 'cpu'):
                            annotated_image = annotated_image.cpu().numpy()
                        annotated_image = np.ascontiguousarray(annotated_image)
                        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, channels="RGB", caption=f"Time: {time_sec:.1f}s | People Count: {res['count']}")
                        
                    finally:
                        os.unlink(tmp_img_path)
                
                frame_idx += 1
                if frame_idx % 5 == 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    
        except Exception as e:
            st.error(f"Error during video processing: {e}")
        finally:
            cap.release()
            
        progress_bar.progress(1.0)
        status_text.success("✅ **Analysis Complete!**")
        
        if results_data:
            st.session_state.vid_analysis_results = pd.DataFrame(results_data)
            vid_csv_path = os.path.join(OUTPUT_DIR, f"video_crowd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            st.session_state.vid_analysis_results.to_csv(vid_csv_path, index=False)

    if st.session_state.vid_analysis_results is not None:
        df = st.session_state.vid_analysis_results
        
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        
        max_count = int(df["Count"].max())
        avg_count = int(df["Count"].mean())
        max_density = df["Density (p/m2)"].max()
        
        highest_risk = "LOW"
        if "CRITICAL" in df["Risk Level"].values: highest_risk = "CRITICAL"
        elif "HIGH" in df["Risk Level"].values: highest_risk = "HIGH"
        elif "MODERATE" in df["Risk Level"].values: highest_risk = "MODERATE"
        
        rc = KPI_RISK_CLASS.get(highest_risk, "c-info")
        
        st.markdown(f"""
        <div class="kpi-row">
            {kpi_html("Peak Crowd", max_count, "Highest frame count", "c-primary")}
            {kpi_html("Average Crowd", avg_count, "Mean across video", "c-primary")}
            {kpi_html("Peak Density", f"{max_density:.2f}", "persons / m\u00b2", rc)}
            {kpi_html("Peak Risk", highest_risk, "Highest recorded risk", rc)}
        </div>""", unsafe_allow_html=True)
        
        st.markdown('<div class="sec-label" style="margin-top:1.5rem">Crowd Count Over Time</div>', unsafe_allow_html=True)
        st.line_chart(df.set_index("Time (s)")["Count"], use_container_width=True)
        
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label" style="margin-bottom:.5rem">Downloads</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="Download Frame-by-Frame CSV Report",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"video_crowd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── LIVE STREAM MONITORING ─────────────────────────────────────────────────[...]
elif app_mode == "📡 Live Stream Monitor":
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Stream Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        stream_url = st.text_input(
            "Stream URL",
            placeholder="rtsp://example.com/stream or http://example.com/stream.m3u8",
            help="Enter RTSP CCTV URL or HTTP/HTTPS stream URL"
        )
    
    with col2:
        sample_rate = st.number_input(
            "Sample Rate (sec)",
            min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            help="Process 1 frame every N seconds"
        )
    
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Alert Configuration</div>', unsafe_allow_html=True)
    
    acol1, acol2, acol3 = st.columns(3, gap="medium")
    
    with acol1:
        alert_threshold = st.selectbox(
            "Alert Threshold",
            ["LOW", "MODERATE", "HIGH", "CRITICAL"],
            index=2,
            help="Alert when density reaches this level"
        )
    
    with acol2:
        alert_cooldown = st.number_input(
            "Alert Cooldown (sec)",
            min_value=10, max_value=300, value=60, step=10,
            help="Minimum seconds between consecutive alerts"
        )
    
    with acol3:
        enable_alerts = st.checkbox("Enable Alerts", value=True)
    
    sound_alert = st.checkbox("Sound Alert", value=True, help="Play beep when alert triggered")
    log_alerts = st.checkbox("Log Alerts", value=True, help="Save alerts to file")
    
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    
    col_start, col_stop = st.columns(2, gap="medium")
    
    with col_start:
        start_monitoring = st.button("🔴 Start Monitoring", use_container_width=True, type="primary")
    
    with col_stop:
        stop_monitoring = st.button("⏹️ Stop Monitoring", use_container_width=True)
    
    # ── Handle Stream Control ──────────────────────────────────────────────
    if start_monitoring and stream_url:
        with st.spinner("Connecting to stream..."):
            alert_cfg = AlertConfig(
                enable_alerts=enable_alerts,
                alert_on_risk=alert_threshold,
                alert_cooldown_sec=alert_cooldown,
                sound_alert=sound_alert,
                log_alerts=log_alerts,
            )
            
            monitor = StreamMonitor(
                stream_url=stream_url,
                area=area,
                sample_rate=sample_rate,
                alert_config=alert_cfg,
            )
            
            if monitor.start_monitoring():
                st.session_state.stream_monitor = monitor
                st.success("✅ Connected! Monitoring started.")
            else:
                st.error("❌ Failed to connect to stream. Check URL and try again.")
    
    if stop_monitoring and st.session_state.stream_monitor:
        st.session_state.stream_monitor.stop_monitoring()
        st.session_state.stream_monitor = None
        st.info("Monitoring stopped.")
    
    # ── Display Stream Stats ───────────────────────────────────────────────
    if st.session_state.stream_monitor:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Live Statistics</div>', unsafe_allow_html=True)
        
        # Refresh stats every 2 seconds
        stats_placeholder = st.empty()
        chart_placeholder = st.empty()
        alerts_placeholder = st.empty()
        
        while st.session_state.stream_monitor and st.session_state.stream_monitor.is_running:
            stats = st.session_state.stream_monitor.get_stats()
            
            # Store history
            if len(st.session_state.stream_stats_history) == 0 or \
               (time.time() - st.session_state.stream_stats_history[-1].get('timestamp', 0)) > 1:
                st.session_state.stream_stats_history.append({
                    'timestamp': time.time(),
                    'count': stats.current_crowd_count,
                    'density': stats.current_density,
                    'risk': stats.current_risk,
                })
            
            # Cap history at 100 records
            if len(st.session_state.stream_stats_history) > 100:
                st.session_state.stream_stats_history = st.session_state.stream_stats_history[-100:]
            
            # Update display
            with stats_placeholder.container():
                status_color = "live" if stats.current_risk in ["HIGH", "CRITICAL"] else "paused"
                st.markdown(f"""
                <div class="status-badge {status_color}">
                    <span class="status-dot"></span>
                    {stats.current_risk}
                </div>
                """, unsafe_allow_html=True)
                
                kpi_cols = st.columns(4)
                with kpi_cols[0]:
                    st.metric("Current Count", int(stats.current_crowd_count), "people")
                with kpi_cols[1]:
                    st.metric("Density", f"{stats.current_density:.2f}", "p/m²")
                with kpi_cols[2]:
                    st.metric("Peak Count", int(stats.peak_count), "people")
                with kpi_cols[3]:
                    st.metric("Frames", int(stats.frames_processed), "processed")
            
            # Chart
            if st.session_state.stream_stats_history:
                with chart_placeholder.container():
                    chart_df = pd.DataFrame(st.session_state.stream_stats_history)
                    st.line_chart(chart_df.set_index('timestamp')[['count', 'density']])
            
            # Alerts log
            with alerts_placeholder.container():
                if Path(monitor.alert_log_path).exists():
                    with open(monitor.alert_log_path, 'r') as f:
                        alerts_content = f.read()
                    st.text_area("Alerts Log", alerts_content, height=150, disabled=True)
            
            time.sleep(2)
            st.rerun()
