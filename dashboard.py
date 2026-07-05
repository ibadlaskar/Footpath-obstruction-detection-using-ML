import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
import tempfile
import time

st.set_page_config(
    page_title="Footpath Obstruction Detection System",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --primary:       #4f46e5;
    --primary-light: #818cf8;
    --accent:        #0891b2;
    --danger:        #dc2626;
    --warning:       #d97706;
    --success:       #059669;
    --bg:            #f1f5f9;
    --bg-card:       #ffffff;
    --bg-subtle:     #f8fafc;
    --border:        #e2e8f0;
    --border-strong: #cbd5e1;
    --text:          #0f172a;
    --text-muted:    #475569;
    --text-light:    #94a3b8;
    --shadow:        0 1px 3px rgba(15,23,42,0.08),0 1px 2px rgba(15,23,42,0.06);
    --shadow-md:     0 4px 12px rgba(15,23,42,0.10);
}

* { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg) !important; }
.stApp p, .stApp label { color: var(--text); }

[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

.sys-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #334155;
    box-shadow: var(--shadow-md);
}
.sys-title     { font-size:1.2rem; font-weight:700; color:#f8fafc; letter-spacing:-0.02em; }
.sys-subtitle  { font-size:0.7rem; color:#64748b; margin-top:5px; text-transform:uppercase; letter-spacing:0.04em; }
.sys-badge {
    background: rgba(79,70,229,0.2); border:1px solid rgba(79,70,229,0.5); color:#a5b4fc;
    padding:6px 14px; border-radius:6px; font-size:0.68rem; font-weight:700;
    font-family:'DM Mono',monospace; letter-spacing:0.06em; text-transform:uppercase;
}

.metric-card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:10px;
    padding:18px 20px; box-shadow:var(--shadow); border-top:4px solid var(--primary);
}
.metric-card.danger  { border-top-color:var(--danger); }
.metric-card.warning { border-top-color:var(--warning); }
.metric-card.success { border-top-color:var(--success); }
.metric-card.info    { border-top-color:var(--accent); }
.metric-value { font-size:2rem; font-weight:700; color:var(--text); font-family:'DM Mono',monospace; line-height:1; }
.metric-label { font-size:0.68rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:7px; font-weight:600; }
.metric-sub   { font-size:0.7rem; color:var(--text-light); margin-top:3px; }

.section-title {
    font-size:0.7rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;
    letter-spacing:0.12em; margin-bottom:14px; padding-bottom:8px; border-bottom:2px solid var(--border);
}

.img-label {
    font-size:0.66rem; font-weight:700; color:var(--primary); text-transform:uppercase;
    letter-spacing:0.12em; margin-bottom:5px; padding:3px 10px;
    background:rgba(79,70,229,0.07); border-radius:4px;
    border-left:3px solid var(--primary); display:inline-block;
}

.alert-box {
    padding:13px 16px; border-radius:8px; font-size:0.88rem; font-weight:600;
    margin-bottom:12px; display:flex; align-items:flex-start; gap:10px; line-height:1.5;
}
.alert-danger  { background:#fef2f2; border:1px solid #fca5a5; border-left:5px solid #dc2626; color:#7f1d1d; }
.alert-warning { background:#fffbeb; border:1px solid #fcd34d; border-left:5px solid #d97706; color:#78350f; }
.alert-info    { background:#eff6ff; border:1px solid #93c5fd; border-left:5px solid #4f46e5; color:#1e3a8a; }
.alert-success { background:#f0fdf4; border:1px solid #86efac; border-left:5px solid #059669; color:#14532d; font-size:0.9rem; font-weight:700; }

.det-item {
    display:flex; align-items:center; justify-content:space-between;
    padding:11px 13px; border-radius:8px; border:1px solid var(--border);
    margin-bottom:7px; background:var(--bg-card); box-shadow:var(--shadow);
}
.det-class { font-size:0.86rem; font-weight:600; color:var(--text); }
.det-conf  { font-family:'DM Mono',monospace; font-size:0.78rem; font-weight:600; padding:3px 9px; border-radius:4px; }

.safety-panel { background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:14px; margin-top:12px; }
.safety-title { font-size:0.72rem; font-weight:700; color:#92400e; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:9px; }
.safety-item  { font-size:0.82rem; color:#78350f; padding:4px 0; display:flex; align-items:flex-start; gap:7px; line-height:1.45; }

.info-panel  { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:14px; }
.info-row    { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #334155; font-size:0.78rem; }
.info-row:last-child { border-bottom:none; }
.info-key { color:#94a3b8; font-weight:500; }
.info-val { color:#e2e8f0; font-family:'DM Mono',monospace; font-size:0.72rem; }

.severity-high   { background:#fee2e2; color:#991b1b; padding:3px 8px; border-radius:4px; font-size:0.68rem; font-weight:700; border:1px solid #fca5a5; }
.severity-medium { background:#fef9c3; color:#92400e; padding:3px 8px; border-radius:4px; font-size:0.68rem; font-weight:700; border:1px solid #fcd34d; }
.severity-low    { background:#dcfce7; color:#14532d; padding:3px 8px; border-radius:4px; font-size:0.68rem; font-weight:700; border:1px solid #86efac; }

.status-pill { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:6px; font-size:0.8rem; font-weight:600; }
.status-ok    { background:#dcfce7; color:#14532d; border:1px solid #86efac; }
.status-warn  { background:#fef9c3; color:#92400e; border:1px solid #fcd34d; }
.status-error { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }

.result-panel-title {
    font-size:0.7rem; font-weight:700; color:var(--primary); text-transform:uppercase;
    letter-spacing:0.12em; margin-bottom:14px; padding-bottom:10px;
    border-bottom:2px solid rgba(79,70,229,0.2);
}

.stButton > button,
.stButton > button:link,
.stButton > button:visited {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 7px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    transition: background 0.15s, border-color 0.15s !important;
    text-shadow: none !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}
.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    background: #334155 !important;
    color: #f8fafc !important;
    border-color: #475569 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    -webkit-text-fill-color: #f8fafc !important;
    outline: none !important;
}
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}

.stDownloadButton > button,
.stDownloadButton > button:link,
.stDownloadButton > button:visited {
    background: #064e3b !important;
    color: #d1fae5 !important;
    border: 1px solid #065f46 !important;
    border-radius: 7px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    transition: background 0.15s !important;
    text-shadow: none !important;
    -webkit-text-fill-color: #d1fae5 !important;
}
.stDownloadButton > button:hover,
.stDownloadButton > button:focus,
.stDownloadButton > button:active {
    background: #065f46 !important;
    color: #ecfdf5 !important;
    border-color: #047857 !important;
    -webkit-text-fill-color: #ecfdf5 !important;
    outline: none !important;
}
.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div {
    color: #d1fae5 !important;
    -webkit-text-fill-color: #d1fae5 !important;
}

[data-testid="stFileUploader"] {
    background:#ffffff !important;
    border:2px dashed #94a3b8 !important;
    border-radius:10px !important;
}
[data-testid="stFileUploader"]:hover { border-color:#4f46e5 !important; }
[data-testid="stFileUploader"] * { color:#0f172a !important; }
[data-testid="stFileUploaderDropzone"] { background:#f8fafc !important; color:#475569 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background:#1e293b !important; color:#f1f5f9 !important;
    border:1px solid #334155 !important; border-radius:6px !important; font-weight:600 !important;
    -webkit-text-fill-color:#f1f5f9 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background:var(--bg-card); border-radius:8px; padding:4px; gap:4px;
    border:1px solid var(--border); box-shadow:var(--shadow);
}
.stTabs [data-baseweb="tab"] {
    background:transparent; color:var(--text-muted); border-radius:6px;
    font-weight:500; font-size:0.84rem; padding:8px 16px; transition:all 0.15s;
}
.stTabs [aria-selected="true"] { background:#0f172a !important; color:#f8fafc !important; font-weight:600 !important; }

.stSlider > div > div > div { background:var(--primary) !important; }
.stCheckbox > label { font-size:0.84rem; color:#94a3b8 !important; }
[data-testid="stExpander"] { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; }
[data-testid="stProgress"] > div > div { background:var(--primary) !important; }

#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }
.stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['light_pole', 'pothole-ditch', 'roadside_stall', 'vehicle']

CLASS_COLORS = {
    "light_pole":     "#0891b2",
    "pothole-ditch":  "#dc2626",
    "roadside_stall": "#d97706",
    "vehicle":        "#7c3aed",
}
CLASS_ICONS = {
    "light_pole":     "💡",
    "pothole-ditch":  "⚠️",
    "roadside_stall": "🏪",
    "vehicle":        "🚗",
}
CLASS_LABELS = {
    "light_pole":     "Light Pole",
    "pothole-ditch":  "Pothole / Ditch",
    "roadside_stall": "Roadside Stall",
    "vehicle":        "Vehicle on Footpath",
}
CLASS_SEVERITY = {
    "pothole-ditch":  ("HIGH",   "severity-high"),
    "vehicle":        ("HIGH",   "severity-high"),
    "roadside_stall": ("MEDIUM", "severity-medium"),
    "light_pole":     ("LOW",    "severity-low"),
}
SAFETY_RECOMMENDATIONS = {
    "pothole-ditch": [
        "Pedestrians should avoid this section of footpath",
        "Recommend reporting to municipal authority for repair",
        "Risk of injury - especially for elderly and visually impaired",
    ],
    "vehicle": [
        "Vehicle blocking pedestrian right of way",
        "Pedestrians may be forced onto road - high accident risk",
        "Recommend alerting traffic enforcement authority",
    ],
    "roadside_stall": [
        "Footpath partially obstructed - reduced walkable width",
        "May affect accessibility for wheelchair users",
        "Recommend relocating stall to designated vendor zone",
    ],
    "light_pole": [
        "Structural element - verify if it restricts walkable space",
        "Ensure adequate clearance for pedestrian movement",
    ],
}

FOOTPATH_MODEL_PATH = r"runs\segment\model\footpath_segment\weights\best.pt"
HAZARD_MODEL_PATH   = r"runs\detect\weights\best.pt"
HISTORY_PATH        = "history/detection_log.json"

MASK_BINARY_THRESHOLD       = 0.45
FOOTPATH_COVERAGE_THRESHOLD = 0.30
MIN_FOOTPATH_PIXELS         = 2500

CLASS_CONF_FLOORS = {
    "light_pole":     0.35,
    "pothole-ditch":  0.35,
    "roadside_stall": 0.45,
    "vehicle":        0.35,
}

def init_session():
    for k, v in {"detection_log": [], "footpath_model": None,
                 "hazard_model": None, "upload_key": 0}.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(log):
    os.makedirs("history", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(log, f, indent=2)

def get_class_counts(log):
    counts = {cls: 0 for cls in CLASS_NAMES}
    for d in log:
        if d.get("class") in counts:
            counts[d["class"]] += 1
    return counts

def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def sanitize_pdf(text):
    return (str(text)
        .replace('\u2014', '-').replace('\u2013', '-')
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2026', '...').replace('\u00a0', ' ')
    )

@st.cache_resource
def load_footpath_model(path):
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception:
            return None
    return None

@st.cache_resource
def load_hazard_model(path):
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception:
            return None
    return None

if st.session_state.footpath_model is None:
    st.session_state.footpath_model = load_footpath_model(FOOTPATH_MODEL_PATH)
if st.session_state.hazard_model is None:
    st.session_state.hazard_model   = load_hazard_model(HAZARD_MODEL_PATH)
if not st.session_state.detection_log:
    st.session_state.detection_log  = load_history()

footpath_ok = st.session_state.footpath_model is not None
hazard_ok   = st.session_state.hazard_model   is not None

def run_detection(image_array, conf_threshold, check_footpath=True):
    h, w             = image_array.shape[:2]
    resized          = cv2.resize(image_array, (640, 640))
    footpath_overlay = None
    footpath_mask    = None

    if check_footpath and st.session_state.footpath_model is not None:
        seg_results = st.session_state.footpath_model(
            resized, conf=0.40, imgsz=640, iou=0.5, verbose=False
        )
        result = seg_results[0]

        if result.masks is None or len(result.masks) == 0:
            return image_array, [], "no_footpath", 0.0, None

        footpath_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_tensor in result.masks.data:
            mask_np      = mask_tensor.cpu().numpy().astype(np.float32)
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_binary  = (mask_resized > MASK_BINARY_THRESHOLD).astype(np.uint8)
            kernel       = np.ones((11, 11), np.uint8)
            mask_binary  = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
            footpath_mask = np.maximum(footpath_mask, mask_binary)

        coverage = (np.sum(footpath_mask) / (h * w)) * 100.0

        if coverage < FOOTPATH_COVERAGE_THRESHOLD or np.sum(footpath_mask) < MIN_FOOTPATH_PIXELS:
            return image_array, [], "no_footpath", coverage, None

        overlay      = image_array.copy()
        colour_layer = np.zeros_like(image_array)
        colour_layer[footpath_mask == 1] = [79, 70, 229]
        footpath_overlay = cv2.addWeighted(overlay, 0.75, colour_layer, 0.25, 0)
        status = "footpath_found"

    else:
        coverage = 0.0
        status   = "check_disabled"

    if st.session_state.hazard_model is None:
        return image_array, [], "no_model", coverage, footpath_overlay

    det_results = st.session_state.hazard_model(
        resized, conf=conf_threshold, imgsz=640, verbose=False
    )

    detections = []
    scale_x    = w / 640
    scale_y    = h / 640

    for box in det_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(max(0, min(x1 * scale_x, w - 1)))
        y1 = int(max(0, min(y1 * scale_y, h - 1)))
        x2 = int(max(0, min(x2 * scale_x, w - 1)))
        y2 = int(max(0, min(y2 * scale_y, h - 1)))

        cls_id      = int(box.cls[0])
        model_names = getattr(st.session_state.hazard_model, "names", {})
        cls_name    = model_names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
        conf        = float(box.conf[0])

        if conf < CLASS_CONF_FLOORS.get(cls_name, conf_threshold):
            continue

        if footpath_mask is not None:
            overlap = footpath_mask[y1:y2, x1:x2]
            if np.sum(overlap) < 500:
                continue

        sev, _ = CLASS_SEVERITY.get(cls_name, ("LOW", "severity-low"))
        detections.append({
            "class":      cls_name,
            "confidence": round(conf, 3),
            "severity":   sev,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    annotated = image_array.copy()
    for d in detections:
        bgr             = hex_to_bgr(CLASS_COLORS.get(d["class"], "#888888"))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        label           = f"{CLASS_LABELS.get(d['class'], d['class'])} {d['confidence']:.0%}"
        (lw, lh), _     = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 6, y1), bgr, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return annotated, detections, status, coverage, footpath_overlay


def log_detections(detections, source):
    for d in detections:
        d["source"] = source
        st.session_state.detection_log.append(d)
    save_history(st.session_state.detection_log)


def render_detection_results(detections, status, coverage, source_name=""):
    if status == "no_footpath":
        st.markdown(
            f'<div class="alert-box alert-warning">'
            f'⚠️ No footpath detected ({coverage:.1f}% coverage). '
            f'Ensure the scene contains a visible footpath or sidewalk.</div>',
            unsafe_allow_html=True)
        return

    if status == "no_model":
        st.markdown('<div class="alert-box alert-danger">Model not loaded. Check model path.</div>',
                    unsafe_allow_html=True)
        return

    if status == "footpath_found":
        st.markdown(
            f'<div class="alert-box alert-success">'
            f'✅ Footpath identified &nbsp;|&nbsp; <b>{coverage:.1f}% coverage</b> &nbsp;|&nbsp; Spatial filter active</div>',
            unsafe_allow_html=True)

    if status == "check_disabled":
        st.markdown(
            '<div class="alert-box alert-info">ℹ️ Footpath check disabled. Scanning full image.</div>',
            unsafe_allow_html=True)

    if not detections:
        st.markdown(
            '<div class="alert-box alert-success">✅ <b>No obstructions detected</b> on this footpath.</div>',
            unsafe_allow_html=True)
        return

    high_risk = list({d["class"] for d in detections if d["severity"] == "HIGH"})
    if high_risk:
        labels = ", ".join(CLASS_LABELS.get(c, c) for c in high_risk)
        st.markdown(
            f'<div class="alert-box alert-danger">'
            f'🚨 <b>HIGH RISK:</b> {labels}<br>'
            f'<span style="font-weight:500;font-size:0.82rem;">Pedestrian safety compromised.</span></div>',
            unsafe_allow_html=True)

    det_counts = {}
    for d in detections:
        det_counts[d["class"]] = det_counts.get(d["class"], 0) + 1

    st.markdown('<div class="section-title" style="margin-top:14px;">Obstructions Identified</div>',
                unsafe_allow_html=True)

    for cls, cnt in det_counts.items():
        color        = CLASS_COLORS.get(cls, "#64748b")
        icon         = CLASS_ICONS.get(cls, "•")
        label        = CLASS_LABELS.get(cls, cls)
        sev, sev_cls = CLASS_SEVERITY.get(cls, ("LOW", "severity-low"))
        avg_conf     = sum(d["confidence"] for d in detections if d["class"] == cls) / cnt

        st.markdown(
            f'<div class="det-item">'
            f'<div>'
            f'  <div class="det-class">{icon} {label}</div>'
            f'  <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">'
            f'  {cnt} instance{"s" if cnt > 1 else ""}</div>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'  <span class="{sev_cls}">{sev}</span>'
            f'  <span class="det-conf" style="background:{color}18;color:{color};">{avg_conf:.0%}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True)

    shown, recs = set(), []
    for d in detections:
        if d["class"] not in shown:
            recs.extend(SAFETY_RECOMMENDATIONS.get(d["class"], []))
            shown.add(d["class"])

    if recs:
        items_html = "".join(
            f'<div class="safety-item"><span style="color:#d97706;font-weight:700;">›</span>'
            f'<span>{r}</span></div>' for r in recs
        )
        st.markdown(
            f'<div class="safety-panel">'
            f'<div class="safety-title">⚡ Safety Recommendations</div>'
            f'{items_html}</div>',
            unsafe_allow_html=True)


_CL = dict(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    font_family="DM Sans", font_color="#0f172a",
    title_font_size=13, title_font_color="#1e293b",
    margin=dict(l=20, r=20, t=40, b=20),
)

def generate_bar_chart(counts, total):
    if total == 0:
        return None
    df = pd.DataFrame([{"Class": CLASS_LABELS.get(k, k), "Count": v} for k, v in counts.items()])
    fig = px.bar(df, x="Class", y="Count", color="Class",
                 color_discrete_map={CLASS_LABELS.get(k, k): v for k, v in CLASS_COLORS.items()},
                 title="Obstruction Frequency by Class")
    fig.update_layout(**_CL, showlegend=False,
                      xaxis=dict(gridcolor="#f1f5f9", title=""),
                      yaxis=dict(gridcolor="#f1f5f9", title="Count"))
    return fig

def generate_trend_chart(log):
    if not log:
        return None
    df = pd.DataFrame(log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    daily           = df.groupby(["date","class"]).size().reset_index(name="count")
    daily["class"]  = daily["class"].map(lambda x: CLASS_LABELS.get(x, x))
    fig = px.line(daily, x="date", y="count", color="class",
                  title="Detection Trends Over Time",
                  labels={"count":"Detections","date":"Date","class":"Class"},
                  color_discrete_map={CLASS_LABELS.get(k,k):v for k,v in CLASS_COLORS.items()})
    fig.update_layout(**_CL, hovermode="x unified",
                      xaxis=dict(gridcolor="#f1f5f9"),
                      yaxis=dict(gridcolor="#f1f5f9"),
                      legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", borderwidth=1))
    return fig

def generate_severity_pie(log):
    if not log:
        return None
    sev_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for d in log:
        sev_counts[CLASS_SEVERITY.get(d.get("class",""), ("LOW",""))[0]] += 1
    fig = go.Figure(data=[go.Pie(
        labels=list(sev_counts.keys()), values=list(sev_counts.values()),
        hole=0.55, marker_colors=["#dc2626","#d97706","#059669"],
        textfont_family="DM Sans")])
    fig.update_layout(**_CL, showlegend=True,
                      legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", borderwidth=1),
                      annotations=[dict(text="Risk", x=0.5, y=0.5, font_size=13, showarrow=False)])
    return fig


def generate_pdf_report(log):
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "fpdf2 not installed. Run: pip install fpdf2"

    total       = len(log)
    counts      = get_class_counts(log)
    most_common = max(counts, key=counts.get) if log else "N/A"
    avg_conf    = sum(d["confidence"] for d in log) / total if total > 0 else 0
    high_count  = sum(1 for d in log if CLASS_SEVERITY.get(d.get("class",""),("LOW",""))[0]=="HIGH")

    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 45, "F")
    pdf.set_text_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(15, 12)
    pdf.cell(0, 8, sanitize_pdf("SMART VISION SYSTEM"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(15, 22)
    pdf.cell(0, 6, sanitize_pdf("Footpath Obstruction Detection - Pedestrian Safety Report"), ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(15, 32)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, sanitize_pdf(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}"), ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(12)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, sanitize_pdf("EXECUTIVE SUMMARY"), ln=True)
    pdf.set_draw_color(79, 70, 229)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    rows = [
        ("Total Obstructions Detected", str(total)),
        ("High-Risk Detections",        str(high_count)),
        ("Most Frequent Obstruction",   CLASS_LABELS.get(most_common, most_common)),
        ("Average Detection Confidence",f"{avg_conf:.1%}"),
        ("Report Date",                 datetime.now().strftime("%d %B %Y")),
        ("Detection System",            "YOLOv11 Two-Stage Pipeline"),
    ]
    for i, (lbl, val) in enumerate(rows):
        pdf.set_fill_color(248,250,252) if i%2==0 else pdf.set_fill_color(255,255,255)
        pdf.set_font("Helvetica","",9); pdf.set_text_color(80,80,80)
        pdf.cell(90, 8, sanitize_pdf(lbl), fill=True)
        pdf.set_font("Helvetica","B",9); pdf.set_text_color(15,23,42)
        pdf.cell(90, 8, sanitize_pdf(val), fill=True, ln=True)
        pdf.ln(0.5)

    pdf.set_text_color(30,30,30); pdf.ln(8)

    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,sanitize_pdf("OBSTRUCTION CLASS ANALYSIS"),ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(4)
    pdf.set_fill_color(15,23,42); pdf.set_text_color(248,250,252); pdf.set_font("Helvetica","B",9)
    for hdr, wc in [("Class",65),("Detections",35),("Percentage",35),("Risk Level",40)]:
        pdf.cell(wc, 8, sanitize_pdf(hdr), fill=True)
    pdf.ln()
    for i, cls in enumerate(CLASS_NAMES):
        cnt = counts[cls]; pct = (cnt/total*100) if total>0 else 0
        sev = CLASS_SEVERITY.get(cls,("LOW",""))[0]
        pdf.set_fill_color(248,250,252) if i%2==0 else pdf.set_fill_color(255,255,255)
        pdf.set_text_color(30,30,30); pdf.set_font("Helvetica","",9)
        pdf.cell(65,8,sanitize_pdf(CLASS_LABELS.get(cls,cls)),fill=True)
        pdf.cell(35,8,sanitize_pdf(str(cnt)),fill=True)
        pdf.cell(35,8,sanitize_pdf(f"{pct:.1f}%"),fill=True)
        pdf.cell(40,8,sanitize_pdf(sev),fill=True,ln=True)

    pdf.ln(8)

    pdf.set_font("Helvetica","B",12); pdf.set_text_color(30,30,30)
    pdf.cell(0,8,sanitize_pdf("PEDESTRIAN SAFETY RECOMMENDATIONS"),ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(4)
    for cls in [c for c,n in counts.items() if n>0]:
        recs = SAFETY_RECOMMENDATIONS.get(cls,[])
        if recs:
            pdf.set_font("Helvetica","B",9); pdf.set_text_color(30,30,30)
            pdf.cell(0,7,sanitize_pdf(f"{CLASS_LABELS.get(cls,cls).upper()}:"),ln=True)
            pdf.set_font("Helvetica","",9); pdf.set_text_color(80,80,80)
            for rec in recs:
                pdf.cell(8,6,""); pdf.cell(0,6,sanitize_pdf(f"- {rec}"),ln=True)
            pdf.ln(2)

    pdf.set_y(-18)
    pdf.set_draw_color(226,232,240); pdf.set_line_width(0.3)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(2)
    pdf.set_font("Helvetica","I",7); pdf.set_text_color(148,163,184)
    pdf.cell(0,5,sanitize_pdf("Smart Vision System - Footpath Obstruction Detection Report"),align="C")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name, None


with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 12px;">
        <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">🚶 Smart Vision System for Footpath Obstruction Detection</div>
        <div style="font-size:0.68rem;color:#475569;margin-top:5px;text-transform:uppercase;letter-spacing:0.1em;">
            Pedestrian Obstruction Monitor</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">System Status</p>', unsafe_allow_html=True)
    fp_col  = "#10b981" if footpath_ok else "#f43f5e"
    haz_col = "#10b981" if hazard_ok   else "#f43f5e"
    st.markdown(
        f'<div class="info-panel">'
        f'<div class="info-row"><span class="info-key">Footpath Model</span>'
        f'<span style="color:{fp_col};font-size:0.74rem;font-weight:700;">{"Active" if footpath_ok else "Not Loaded"}</span></div>'
        f'<div class="info-row"><span class="info-key">Detection Model</span>'
        f'<span style="color:{haz_col};font-size:0.74rem;font-weight:700;">{"Active" if hazard_ok else "Not Loaded"}</span></div>'
        f'<div class="info-row"><span class="info-key">Pipeline</span>'
        f'<span class="info-val">Two-Stage YOLOv11</span></div>'
        f'<div class="info-row"><span class="info-key">Classes</span>'
        f'<span class="info-val">{len(CLASS_NAMES)}</span></div>'
        f'</div>', unsafe_allow_html=True)

    if st.button("⟳ Reload Models"):
        load_footpath_model.clear(); load_hazard_model.clear()
        st.session_state.footpath_model = load_footpath_model(FOOTPATH_MODEL_PATH)
        st.session_state.hazard_model   = load_hazard_model(HAZARD_MODEL_PATH)
        st.rerun()

    st.markdown("---")
    st.markdown('<p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">Detection Settings</p>', unsafe_allow_html=True)

    enable_footpath_check = st.checkbox("Enable Footpath Segmentation", value=True,
                                        disabled=not footpath_ok,
                                        help="Stage 1: Identify footpath before detection")
    show_footpath_overlay = st.checkbox("Show Footpath Overlay", value=True,
                                        disabled=not footpath_ok)
    conf_threshold = st.slider("Detection Confidence", 0.10, 0.90, 0.35, 0.05)

    st.markdown("---")
    st.markdown('<p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">Monitored Classes</p>', unsafe_allow_html=True)
    for cls in CLASS_NAMES:
        color = CLASS_COLORS[cls]
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:6px 10px;border-radius:6px;margin:3px 0;'
            f'background:rgba(255,255,255,0.04);border:1px solid #1e293b;">'
            f'<span style="font-size:0.79rem;color:#cbd5e1;">{CLASS_ICONS[cls]} {CLASS_LABELS[cls]}</span>'
            f'<span style="font-size:0.63rem;font-weight:700;padding:2px 7px;border-radius:3px;'
            f'background:{color}28;color:{color};border:1px solid {color}55;">'
            f'{CLASS_SEVERITY[cls][0]}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    log    = st.session_state.detection_log
    total  = len(log)
    counts = get_class_counts(log)
    high_n = sum(1 for d in log if CLASS_SEVERITY.get(d.get("class",""),("LOW",""))[0]=="HIGH")

    st.markdown('<p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">Session Summary</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-panel">'
        f'<div class="info-row"><span class="info-key">Total Detections</span>'
        f'<span class="info-val">{total}</span></div>'
        f'<div class="info-row"><span class="info-key">High Risk</span>'
        f'<span style="color:#f43f5e;font-family:DM Mono,monospace;font-size:0.74rem;font-weight:700;">{high_n}</span></div>'
        f'<div class="info-row"><span class="info-key">Potholes</span>'
        f'<span class="info-val">{counts["pothole-ditch"]}</span></div>'
        f'<div class="info-row"><span class="info-key">Vehicles</span>'
        f'<span class="info-val">{counts["vehicle"]}</span></div>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear All Data & Reset"):
        st.session_state.detection_log = []
        save_history([])
        st.session_state.upload_key += 1
        st.success("All data cleared.")
        st.rerun()


st.markdown(
    '<div class="sys-header">'
    '<div>'
    '<div class="sys-title">🚶 Smart Vision System for Footpath Obstruction Detection</div>'
    '<div class="sys-subtitle">Pedestrian Safety Monitoring &nbsp;·&nbsp; YOLOv11 Two-Stage Pipeline &nbsp;·&nbsp; Spatial Filtering</div>'
    '</div>'
    '<div class="sys-badge">Pedestrian Safety AI</div>'
    '</div>', unsafe_allow_html=True)

if not hazard_ok:
    st.markdown(
        f'<div class="alert-box alert-danger">⚠️ Detection model not found at '
        f'<code>{HAZARD_MODEL_PATH}</code>. Update HAZARD_MODEL_PATH in dashboard.py.</div>',
        unsafe_allow_html=True)
    st.stop()

log    = st.session_state.detection_log
total  = len(log)
counts = get_class_counts(log)
avg_c  = sum(d["confidence"] for d in log)/total if total > 0 else 0
high_n = sum(1 for d in log if CLASS_SEVERITY.get(d.get("class",""),("LOW",""))[0]=="HIGH")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card info"><div class="metric-value">{total}</div>'
                f'<div class="metric-label">Total Obstructions</div>'
                f'<div class="metric-sub">All sessions</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card danger"><div class="metric-value">{high_n}</div>'
                f'<div class="metric-label">High Risk Events</div>'
                f'<div class="metric-sub">Vehicles + Potholes</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card warning"><div class="metric-value">{counts["pothole-ditch"]}</div>'
                f'<div class="metric-label">Potholes Detected</div>'
                f'<div class="metric-sub">Surface hazards</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card success"><div class="metric-value">'
                f'{f"{avg_c:.0%}" if total>0 else "--"}</div>'
                f'<div class="metric-label">Avg Confidence</div>'
                f'<div class="metric-sub">Detection reliability</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Image Analysis", "🎬 Video Analysis",
    "📹 Live Detection", "📊 Analytics", "📄 Reports",
])


with tab1:
    st.markdown('<div class="section-title">Image-Based Obstruction Analysis</div>',
                unsafe_allow_html=True)

    with st.expander("ℹ️ System Information", expanded=False):
        info_rows = [
            ("Accepted Formats",     "JPG, JPEG, PNG, BMP, WEBP"),
            ("Min Resolution",       "640x640 recommended"),
            ("Pipeline",             "Seg -> Detection -> Spatial filter"),
            ("Multiple Upload",      "Supported - images processed in sequence"),
        ]
        html = '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;">'
        for i, (k, v) in enumerate(info_rows):
            bdr = "border-bottom:1px solid #e2e8f0;" if i < len(info_rows)-1 else ""
            html += (f'<div style="display:flex;justify-content:space-between;padding:5px 0;{bdr}font-size:0.8rem;">'
                     f'<span style="color:#475569;font-weight:500;">{k}</span>'
                     f'<span style="font-family:DM Mono,monospace;color:#0f172a;font-size:0.75rem;">{v}</span></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload footpath images",
        type=["jpg","jpeg","png","bmp","webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}",
        help="Upload street-level images containing footpath areas",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img        = Image.open(uploaded_file).convert("RGB")
            img_array  = np.array(img)
            h_img, w_img = img_array.shape[:2]

            st.markdown(
                f'<div style="margin:12px 0 8px;font-size:0.8rem;color:#475569;'
                f'padding:6px 12px;background:#ffffff;border:1px solid #e2e8f0;'
                f'border-radius:6px;display:inline-block;">'
                f'📄 {uploaded_file.name} &nbsp;&mdash;&nbsp; {w_img}&times;{h_img}px</div>',
                unsafe_allow_html=True)

            with st.spinner("Running two-stage detection..."):
                annotated, detections, status, coverage, fp_overlay = run_detection(
                    img_array, conf_threshold, enable_footpath_check)

            show_overlay = show_footpath_overlay and fp_overlay is not None and enable_footpath_check

            left_col, right_col = st.columns([5, 2], gap="large")

            with left_col:
                if show_overlay:
                    c1, c2, c3 = st.columns(3, gap="small")
                    with c1:
                        st.markdown('<div class="img-label">① Original</div>', unsafe_allow_html=True)
                        st.image(img, use_container_width=True)
                    with c2:
                        st.markdown('<div class="img-label">② Footpath Region</div>', unsafe_allow_html=True)
                        st.image(fp_overlay, use_container_width=True)
                    with c3:
                        st.markdown('<div class="img-label">③ Detection Output</div>', unsafe_allow_html=True)
                        st.image(annotated, use_container_width=True)
                else:
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        st.markdown('<div class="img-label">① Original</div>', unsafe_allow_html=True)
                        st.image(img, use_container_width=True)
                    with c2:
                        st.markdown('<div class="img-label">② Detection Output</div>', unsafe_allow_html=True)
                        st.image(annotated, use_container_width=True)

            with right_col:
                st.markdown('<div class="result-panel-title">🔍 Detection Results</div>',
                            unsafe_allow_html=True)
                render_detection_results(detections, status, coverage, uploaded_file.name)

            if detections:
                log_detections(detections, uploaded_file.name)

            st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:24px 0;'>",
                        unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-title">Video-Based Obstruction Analysis</div>',
                unsafe_allow_html=True)

    with st.expander("ℹ️ Video Input Information", expanded=False):
        info_rows = [
            ("Accepted Formats", "MP4, AVI, MOV, MKV"),
            ("Processing Mode",  "Frame-by-frame with configurable skip rate"),
            ("Best Use",         "CCTV footage, dashcam, street walkthroughs"),
            ("Output",           "Live annotated frames + detection log"),
        ]
        html = '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;">'
        for i, (k, v) in enumerate(info_rows):
            bdr = "border-bottom:1px solid #e2e8f0;" if i < len(info_rows)-1 else ""
            html += (f'<div style="display:flex;justify-content:space-between;padding:5px 0;{bdr}font-size:0.8rem;">'
                     f'<span style="color:#475569;font-weight:500;">{k}</span>'
                     f'<span style="font-family:DM Mono,monospace;color:#0f172a;font-size:0.75rem;">{v}</span></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload video file", type=["mp4","avi","mov","mkv"],
        key=f"video_uploader_{st.session_state.upload_key}")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read()); tfile.flush()

        cap          = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid      = cap.get(cv2.CAP_PROP_FPS)
        duration     = total_frames / fps_vid if fps_vid > 0 else 0
        cap.release()

        st.markdown(
            f'<div class="alert-box alert-info">📹 {uploaded_video.name} &nbsp;|&nbsp; '
            f'{total_frames} frames &nbsp;|&nbsp; {fps_vid:.1f} FPS &nbsp;|&nbsp; {duration:.1f}s</div>',
            unsafe_allow_html=True)

        process_every = st.slider("Process every N frames", 1, 30, 5)

        if st.button("▶ Start Video Analysis"):
            cap = cv2.VideoCapture(tfile.name)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="img-label">Original Frame</div>', unsafe_allow_html=True)
                orig_ph = st.empty()
            with c2:
                st.markdown('<div class="img-label">Obstruction Detection</div>', unsafe_allow_html=True)
                det_ph  = st.empty()
            prog = st.progress(0); status_ph = st.empty()
            frame_idx = 0; vid_dets = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig_ph.image(frame_rgb, use_container_width=True)

                if frame_idx % process_every == 0:
                    annotated, detections, status, coverage, _ = run_detection(
                        frame_rgb, conf_threshold, enable_footpath_check)
                    vid_dets.extend(detections)
                    det_ph.image(annotated, use_container_width=True)
                    if status == "no_footpath":
                        msg = f'<div class="status-pill status-warn">⚠️ No footpath — {coverage:.1f}%</div>'
                    elif detections:
                        msg = f'<div class="status-pill status-error">🚨 {len(detections)} obstruction(s)</div>'
                    else:
                        msg = '<div class="status-pill status-ok">✅ Clear</div>'
                    status_ph.markdown(msg, unsafe_allow_html=True)

                prog.progress(min(frame_idx/max(total_frames,1), 1.0))
                frame_idx += 1

            cap.release()
            if vid_dets:
                log_detections(vid_dets, uploaded_video.name)
            st.markdown(
                f'<div class="alert-box alert-success">✅ Analysis complete — '
                f'{len(vid_dets)} obstructions logged from {frame_idx} frames.</div>',
                unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="section-title">Live Webcam Obstruction Detection</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="alert-box alert-info">ℹ️ Real-time monitoring via connected camera. '
        'Detections are logged automatically. Press Stop to end session.</div>',
        unsafe_allow_html=True)

    c_start, c_stop = st.columns(2)
    start_cam = c_start.button("▶ Start Live Detection")
    stop_cam  = c_stop.button("⬛ Stop")

    if start_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.markdown(
                '<div class="alert-box alert-danger">Cannot access webcam. '
                'Check connection and permissions.</div>', unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="img-label">Live Feed</div>', unsafe_allow_html=True)
                orig_w = st.empty()
            with c2:
                st.markdown('<div class="img-label">Obstruction Detection</div>', unsafe_allow_html=True)
                det_w = st.empty()
            stat_w = st.empty(); count_w = st.empty(); session_dets = []

            while not stop_cam:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig_w.image(frame_rgb, use_container_width=True)
                annotated, detections, status, coverage, _ = run_detection(
                    frame_rgb, conf_threshold, enable_footpath_check)
                det_w.image(annotated, use_container_width=True)

                if status == "no_footpath":
                    stat_w.markdown(f'<div class="status-pill status-warn">⚠️ No footpath — {coverage:.1f}%</div>', unsafe_allow_html=True)
                elif detections:
                    stat_w.markdown(f'<div class="status-pill status-error">🚨 {len(detections)} obstruction(s)</div>', unsafe_allow_html=True)
                else:
                    stat_w.markdown('<div class="status-pill status-ok">✅ Clear</div>', unsafe_allow_html=True)

                if detections:
                    session_dets.extend(detections)
                    log_detections(detections, "webcam")
                    count_w.markdown(f'<div style="font-size:0.78rem;color:#475569;padding:4px 0;">Session detections: {len(session_dets)}</div>', unsafe_allow_html=True)
                time.sleep(0.04)

            cap.release()
            st.markdown(
                f'<div class="alert-box alert-success">✅ Session ended — {len(session_dets)} obstructions recorded.</div>',
                unsafe_allow_html=True)


with tab4:
    st.markdown('<div class="section-title">Pedestrian Safety Analytics</div>', unsafe_allow_html=True)

    log    = st.session_state.detection_log
    total  = len(log)
    counts = get_class_counts(log)

    if not log:
        st.markdown(
            '<div class="alert-box alert-info">ℹ️ No detection data yet. '
            'Run image, video, or live detection first.</div>', unsafe_allow_html=True)
    else:
        high_n = sum(1 for d in log if CLASS_SEVERITY.get(d.get("class",""),("LOW",""))[0]=="HIGH")
        med_n  = sum(1 for d in log if CLASS_SEVERITY.get(d.get("class",""),("LOW",""))[0]=="MEDIUM")
        low_n  = sum(1 for d in log if CLASS_SEVERITY.get(d.get("class",""),("LOW",""))[0]=="LOW")

        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown(f'<div class="metric-card danger"><div class="metric-value">{high_n}</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
        with cb:
            st.markdown(f'<div class="metric-card warning"><div class="metric-value">{med_n}</div><div class="metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
        with cc:
            st.markdown(f'<div class="metric-card success"><div class="metric-value">{low_n}</div><div class="metric-label">Low Risk</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = generate_bar_chart(counts, total)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with ch2:
            fig = generate_severity_pie(log)
            if fig: st.plotly_chart(fig, use_container_width=True)

        fig = generate_trend_chart(log)
        if fig: st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Per-Class Breakdown</div>', unsafe_allow_html=True)
        for cls in CLASS_NAMES:
            cnt  = counts[cls]
            pct  = (cnt/total*100) if total>0 else 0
            color = CLASS_COLORS[cls]
            sev, scl = CLASS_SEVERITY[cls]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid #f1f5f9;">'
                f'<span style="font-size:0.84rem;width:200px;color:#0f172a;font-weight:500;">{CLASS_ICONS[cls]} {CLASS_LABELS[cls]}</span>'
                f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:8px;">'
                f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;box-shadow:0 0 4px {color}66;"></div></div>'
                f'<span style="font-family:DM Mono,monospace;font-size:0.8rem;color:{color};min-width:36px;text-align:right;font-weight:700;">{cnt}</span>'
                f'<span style="color:#94a3b8;font-size:0.76rem;min-width:48px;">{pct:.1f}%</span>'
                f'<span class="{scl}">{sev}</span>'
                f'</div>', unsafe_allow_html=True)


with tab5:
    st.markdown('<div class="section-title">Report Generation</div>', unsafe_allow_html=True)

    log   = st.session_state.detection_log
    total = len(log)

    if not log:
        st.markdown(
            '<div class="alert-box alert-info">ℹ️ No detection data available for export. '
            'Run detection first.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-box alert-success">✅ {total} detection records ready for export.</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="section-title">Detection Log</div>', unsafe_allow_html=True)
        log_df = pd.DataFrame(log)
        display_cols = [c for c in ["timestamp","source","class","confidence","severity"] if c in log_df.columns]
        log_df["class"] = log_df["class"].map(lambda x: CLASS_LABELS.get(x,x))
        st.dataframe(log_df[display_cols].sort_values("timestamp",ascending=False),
                     use_container_width=True, height=280)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Export Options</div>', unsafe_allow_html=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            csv_data = log_df[display_cols].to_csv(index=False)
            st.download_button(
                label="⬇ Download CSV", data=csv_data,
                file_name=f"obstruction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True)

        with ec2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    pdf_path, err = generate_pdf_report(log)
                if err:
                    st.markdown(f'<div class="alert-box alert-danger">{err}</div>', unsafe_allow_html=True)
                else:
                    with open(pdf_path,"rb") as f:
                        pdf_bytes = f.read()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="⬇ Download PDF", data=pdf_bytes,
                        file_name=f"safety_report_{ts}.pdf",
                        mime="application/pdf", use_container_width=True)
                    os.unlink(pdf_path)
                    st.markdown(
                        '<div class="alert-box alert-success">✅ PDF report ready for download.</div>',
                        unsafe_allow_html=True)