import streamlit as st
import pandas as pd
import numpy as np
import math
import json
import streamlit.components.v1 as components

st.set_page_config(
    layout="wide",
    page_title="Cluster NBR Map",
    page_icon="🗺️",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS to minimize Streamlit chrome and unify the look
st.markdown("""
<style>
    /* Remove default Streamlit padding and margins */
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
    .stApp { background: #0f172a !important; }
    
    /* Hide Streamlit menu and footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDecoration { display: none !important; }
    
    /* Slim uploaders */
    .stFileUploader { margin: 0 !important; padding: 0 !important; }
    .stFileUploader > div { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 6px !important; padding: 8px 12px !important; }
    .stFileUploader label { font-size: 11px !important; color: #94a3b8 !important; margin-bottom: 4px !important; }
    .stFileUploader [data-testid="stFileUploaderDropzone"] { min-height: 40px !important; padding: 6px !important; }
    
    /* Compact columns */
    .stColumn { padding: 0 4px !important; }
    
    /* Hide Streamlit title default styling */
    h1 { margin: 0 !important; padding: 0 !important; font-size: 0 !important; }
    .stMarkdown h1 { display: none !important; }
    
    /* Remove extra spacing */
    .element-container { margin-bottom: 0.2rem !important; }
    div[data-testid="stVerticalBlock"] > div { margin-bottom: 0 !important; }
    
    /* Custom header bar */
    .custom-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 8px 20px;
        font-size: 16px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 6px 6px 0 0;
        margin-bottom: 0;
    }
    .custom-header .subtitle { font-size: 11px; opacity: 0.85; font-weight: 400; }
    
    /* Upload bar */
    .upload-bar {
        background: #1e293b;
        border: 1px solid #334155;
        border-top: none;
        padding: 8px 16px;
        display: flex;
        gap: 12px;
        align-items: center;
        border-radius: 0 0 6px 6px;
        margin-bottom: 0;
    }
    .upload-bar .upload-label {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 500;
        margin-right: 4px;
    }
    .upload-bar .upload-item {
        flex: 1;
    }
    .upload-bar .status {
        color: #10b981;
        font-size: 11px;
        font-weight: 500;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 4px;
        white-space: nowrap;
    }
    .upload-bar .status.error {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* Map container */
    .map-container {
        border: 1px solid #334155;
        border-radius: 6px;
        overflow: hidden;
        margin-top: 0;
    }
    
    /* Hide Streamlit uploader default labels */
    .stFileUploader > label { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── CONFIG & HELPERS ─────────────────────────
MAX_DISTANCE_MILES = 25.0
BUFFER_MILES = 1.8
CLOSEST_N = 15
ZERO_HO_IS_MISSING = False
WEDGE_RADIUS_M = 400

USID_COL, SITE_NAME_COL, LAT_COL, LON_COL = "USID", "ENODEB_New", "LATITUDE", "LONGITUDE"
AZIMUTH_COL, CELL_COL = "AZIMUTH", "CELL"
NBR_SRC_COL, NBR_COL, CLUSTER_NAME_COL = "USID", "ClusterUSID", "ClusterName"
HO_COL, PCT_COL = "HO ATT", "% Sharing"

def norm_id(x):
    s = str(x).strip().upper()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s

def num(v):
    try: return float(v) if pd.notna(v) else 0.0
    except: return 0.0

def load_sites(df):
    df = df.dropna(subset=[USID_COL])
    df[USID_COL] = df[USID_COL].map(norm_id)
    for c in (LAT_COL, LON_COL, AZIMUTH_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[LAT_COL, LON_COL])
    if SITE_NAME_COL in df.columns:
        names = df[SITE_NAME_COL].where(df[SITE_NAME_COL].notna(), df[USID_COL]).astype(str).str.strip()
        df["_name"] = names.where(names != "", df[USID_COL])
    else:
        df["_name"] = df[USID_COL]
    sites = df.groupby(USID_COL, sort=False).agg(lat=(LAT_COL, "first"), lon=(LON_COL, "first"), name=("_name", "first"))
    return df, sites

def load_nbr_table(df):
    t = df.dropna(subset=[NBR_SRC_COL, NBR_COL])
    t[NBR_SRC_COL] = t[NBR_SRC_COL].map(norm_id)
    t[NBR_COL] = t[NBR_COL].map(norm_id)
    return t

def select_sites(sites, tbl, sources):
    lat, lon = sites.lat.values, sites.lon.values
    keep = np.zeros(len(sites), dtype=bool)
    dlat = MAX_DISTANCE_MILES / 69.0
    for s in sources:
        if s not in sites.index: continue
        la, lo = sites.at[s, "lat"], sites.at[s, "lon"]
        dlon = dlat / max(math.cos(math.radians(la)), 0.01)
        keep |= (np.abs(lat - la) <= dlat) & (np.abs(lon - lo) <= dlon)
    keep |= sites.index.isin(set(tbl.loc[tbl[NBR_SRC_COL].isin(sources), NBR_COL]))
    return sites[keep]

def build_payload(df, tbl):
    df, sites = load_sites(df)
    tbl = load_nbr_table(tbl)
    sources = list(tbl[NBR_SRC_COL].unique())
    keep = select_sites(sites, tbl, sources)
    
    sub = df[df[USID_COL].isin(keep.index)]
    cell_series = sub[CELL_COL] if CELL_COL in sub.columns else pd.Series([None] * len(sub), index=sub.index)
    cells = {}
    for u, c, az in zip(sub[USID_COL], cell_series, sub[AZIMUTH_COL]):
        if pd.notna(az):
            label = str(c).strip() if pd.notna(c) and str(c).strip() else f"{u}_Sector"
            cells.setdefault(u, []).append([label, float(az)])
            
    site_json = {u: [r.name, round(float(r.lat), 6), round(float(r.lon), 6), cells.get(u, [])]
                 for u, r in zip(keep.index, keep.itertuples())}

    has_ho, has_pct = HO_COL in tbl.columns, PCT_COL in tbl.columns
    nbr_json = {}
    for src, g in tbl.groupby(NBR_SRC_COL, sort=False):
        cname = str(g[CLUSTER_NAME_COL].dropna().iloc[0]).strip() if CLUSTER_NAME_COL in g.columns and g[CLUSTER_NAME_COL].notna().any() else ""
        hos = g[HO_COL].map(num) if has_ho else [None] * len(g)
        pcts = g[PCT_COL].map(num) if has_pct else [None] * len(g)
        nbr_json[src] = [cname, [[n, h, p] for n, h, p in zip(g[NBR_COL], hos, pcts)]]

    return {"cfg": {"buffer": BUFFER_MILES, "radiusM": WEDGE_RADIUS_M, "maxMiles": MAX_DISTANCE_MILES,
                    "closestN": CLOSEST_N, "zeroHo": ZERO_HO_IS_MISSING, "default": None},
            "sites": site_json, "nbr": nbr_json}

# ───────────────────────── STREAMLIT UI ─────────────────────────

# Custom unified header
st.markdown("""
<div class="custom-header">
    <div>🗺️ Cluster Neighbor Visualization Tool</div>
    <div class="subtitle">@Rajesh Dubey | rajesh.dubey@sds.com</div>
</div>
""", unsafe_allow_html=True)

# Slim upload bar
site_file = None
nbr_file = None

with st.container():
    st.markdown('<div class="upload-bar">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 0.3], gap="small")
    
    with col1:
        site_file = st.file_uploader("", type=["csv"], key="site", label_visibility="collapsed")
    
    with col2:
        nbr_file = st.file_uploader("", type=["xlsx", "xls", "csv"], key="nbr", label_visibility="collapsed")
    
    with col3:
        if site_file and nbr_file:
            st.markdown('<div class="status">✓ Files Loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status error">Upload both files</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Map container
if site_file and nbr_file:
    with st.container():
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        
        with st.spinner("Processing data..."):
            try:
                site_df = pd.read_csv(site_file, low_memory=False)
                if nbr_file.name.lower().endswith('.csv'):
                    nbr_df = pd.read_csv(nbr_file, low_memory=False)
                else:
                    nbr_df = pd.read_excel(nbr_file)
                
                payload = build_payload(site_df, nbr_df)
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
                
                # The HTML Template (compact, professional dark theme)
                HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Cluster NBR Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body{margin:0;height:100%;font-family:'Segoe UI',system-ui,sans-serif;font-size:12px;color:#e2e8f0;background:#0f172a}
#app{display:flex;flex-direction:column;height:100vh}
#wrap{display:flex;flex:1;min-height:0;overflow:hidden}
#side{width:340px;min-width:340px;overflow-y:auto;background:#1e293b;border-right:1px solid #334155;padding:12px;box-sizing:border-box}
#mapbox{flex:1;position:relative;background:#020617;min-width:0}
#map{height:100%;width:100%;--fs:13px}
h4{margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #334155;font-size:11px;color:#818cf8;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
.ttl{font-size:15px;font-weight:700;color:#f87171;margin-top:8px}.sub{color:#94a3b8;margin-top:2px;font-size:11px}.hint{font-size:10px;color:#64748b}
label{display:flex;align-items:center;gap:6px;margin:5px 0;cursor:pointer;color:#cbd5e1;font-size:12px}
input[type=range]{width:100%;accent-color:#6366f1;height:4px}
.box{background:#0f172a;border-radius:5px;padding:8px;margin:6px 0;border:1px solid #334155}
.row{display:flex;justify-content:space-between;align-items:center;margin:6px 0;gap:6px}
button{cursor:pointer;border:1px solid #334155;background:#334155;color:#e2e8f0;border-radius:4px;padding:5px 10px;font-size:11px;font-weight:500;transition:all 0.2s}
button:hover{background:#4f46e5;border-color:#4f46e5;color:#fff}
button.on{background:#4f46e5;color:#fff;border-color:#4f46e5}
.srch{position:relative;margin-bottom:12px}
#q{width:100%;box-sizing:border-box;padding:8px 10px;font-size:12px;background:#0f172a;border:1px solid #334155;border-radius:5px;color:#e2e8f0;outline:none}
#q:focus{border-color:#6366f1}
#ac{position:absolute;top:100%;left:0;right:0;background:#1e293b;border:1px solid #334155;border-radius:5px;max-height:240px;overflow-y:auto;z-index:2000;display:none}
.ai{padding:7px 10px;cursor:pointer;display:flex;gap:8px;align-items:center;border-bottom:1px solid #0f172a}
.ai:hover,.ai.sel{background:#334155}.ai b{color:#818cf8}.ai span{flex:1;color:#94a3b8;font-size:11px}
.stat{display:flex;gap:5px;margin-bottom:8px}.stat div{flex:1;background:#0f172a;border:1px solid #334155;border-radius:5px;padding:6px 4px;text-align:center}
.stat b{display:block;font-size:15px;color:#818cf8}
.pills{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px}.pills button{border-radius:11px;padding:3px 9px;font-size:10px}
.nb{border:1px solid #334155;border-radius:5px;padding:7px;margin:5px 0;cursor:pointer;background:#0f172a}
.nb:hover{background:#1e293b;border-color:#6366f1}
.nbt{display:flex;align-items:center;gap:5px}.nbt .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#94a3b8;font-size:11px}
.bd{color:#fff;font-size:10px;padding:2px 7px;border-radius:9px;white-space:nowrap;font-weight:600}.meta{font-size:10px;color:#64748b;margin:3px 0 0 22px}
.lg{display:flex;align-items:center;gap:7px;margin:3px 0;font-size:11px;color:#cbd5e1}.sw{width:13px;height:13px;border-radius:3px;display:inline-block}
.lbl{font-weight:600;font-size:var(--fs);white-space:nowrap;text-shadow:1px 1px 3px #000,-1px -1px 3px #000}
.lbl-source{font-size:calc(var(--fs) + 2px)}.lbl-other{display:none}.zhi .lbl-other{display:block}.nolbl .lbl{display:none!important}
.vh{width:10px;height:10px;background:#fff;border:2px solid #6366f1;border-radius:50%;cursor:move}
#mbtn{position:absolute;top:8px;left:8px;z-index:1000;display:flex;gap:5px}
.mb{background:#1e293b;color:#e2e8f0;box-shadow:0 2px 6px rgba(0,0,0,0.4);border:1px solid #334155;padding:6px 10px;font-size:11px;border-radius:5px}
.mb:hover{background:#334155;color:#fff}.mb.on{background:#4f46e5;color:#fff;border-color:#4f46e5}
.crosshair,.crosshair.leaflet-grab{cursor:crosshair!important}
#selbar{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 12px;z-index:1000;display:none;align-items:center;gap:8px}
#selN{font-weight:bold;color:#f59e0b;min-width:70px;font-size:11px}
#rbox{position:absolute;top:8px;right:8px;background:#1e293b;border:1px solid #334155;border-radius:6px;padding:10px 12px;z-index:1000;display:none;min-width:240px}
.rtip{background:#1e293b;border:1px solid #f87171;color:#f87171;font-weight:bold;font-size:11px;padding:2px 5px;border-radius:3px}
#mdl{position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:5000;display:none;align-items:center;justify-content:center}
.mbox{background:#1e293b;border-radius:8px;width:540px;max-width:95vw;max-height:85vh;display:flex;flex-direction:column;border:1px solid #334155}
.mh,.fm,.mf{padding:12px 16px;display:flex;align-items:center;gap:8px}.mh{justify-content:space-between;border-bottom:1px solid #0f172a;font-size:13px;color:#818cf8;font-weight:600}
.fm button.on{background:#4f46e5;color:#fff;border-color:#4f46e5}.mf{border-top:1px solid #0f172a}.mf span{flex:1;color:#94a3b8;font-size:11px}
#mTa{margin:0 16px;height:240px;font-family:Consolas,monospace;font-size:11px;resize:none;box-sizing:border-box;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:8px}
#toast{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);background:#10b981;color:#fff;padding:8px 16px;border-radius:5px;z-index:1000;display:none;max-width:70%;font-weight:500;font-size:12px}
.warn{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#fca5a5;border-radius:5px;padding:6px 8px;margin-top:6px;font-size:11px}
select{width:100%;font-size:11px;padding:5px 7px;margin:3px 0;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px}
input[type="color"]{width:36px;height:26px;border:1px solid #334155;border-radius:4px;background:#0f172a;cursor:pointer}
#side::-webkit-scrollbar{width:5px}#side::-webkit-scrollbar-track{background:#0f172a}#side::-webkit-scrollbar-thumb{background:#4f46e5;border-radius:3px}
</style></head><body>
<div id="app">
<div id="wrap">
<div id="side">
  <div style="font-weight:600;font-size:12px;margin-bottom:6px;color:#818cf8;text-transform:uppercase;letter-spacing:0.5px">Provide USID</div>
  <div class="srch"><input id="q" placeholder="Enter USID or site name" autocomplete="off"><div id="ac"></div></div>
  <div id="empty" class="hint" style="margin-top:8px">Type a USID above, or click any site on the map.</div>
  <div id="panel" style="display:none">
  <div class="ttl" id="tSrc"></div><div class="sub" id="tSub"></div><div class="hint" id="tInfo"></div>
  <h4>Polygon Controls</h4>
  <label><input type="checkbox" id="chkShow" checked> <b>Show polygon</b></label>
  <div class="box"><label><input type="checkbox" id="chkEdit"> <b style="color:#818cf8">Enable editing</b></label><div class="hint">Drag handles, click line to add, right-click to delete</div></div>
  <label>Buffer: <b id="bufVal"></b> mi</label><input type="range" id="buf" min="0" max="6" step="0.1">
  <label>Line: <b id="wVal">3</b>px</label><input type="range" id="pw" min="1" max="10" step="1" value="3">
  <select id="ps"><option value="" selected>Solid</option><option value="6, 6">Dashed</option><option value="2, 4">Dotted</option></select>
  <div class="row"><span>Colour</span><input type="color" id="pc" value="#818cf8"></div>
  <label>Keep OUTSIDE:</label>
  <select id="excl"><option value="missing" selected>Missing / suggested</option><option value="all">All non-defined</option><option value="none">None</option></select>
  <div class="row"><button id="btnRebuild">Rebuild</button><button id="btnFit">Fit view</button></div>
  <div class="hint" id="inside" style="font-size:10px;margin-top:3px"></div>
  <h4>View Controls</h4>
  <label>Sector: <b id="sVal">1.0</b>x</label><input type="range" id="ss" min="0.5" max="6" step="0.1" value="1">
  <label>Font: <b id="fVal">13</b>px</label><input type="range" id="fs" min="8" max="28" step="1" value="13">
  <label><input type="checkbox" id="vLines" checked> HO lines</label>
  <label><input type="checkbox" id="vMiss" checked> Missing sites</label>
  <label><input type="checkbox" id="vOther" checked> Other sites</label>
  <label><input type="checkbox" id="vLbl" checked> Labels</label>
  <label><input type="checkbox" id="vBlue"> <b>All blue</b></label>
  <h4>Summary</h4><div class="stat" id="stats"></div><div id="warn"></div>
  <h4>Neighbours</h4><div class="pills" id="pills"></div><div id="list"></div>
  <div class="row"><button id="btnCsv">Export missing</button><button id="btnCsvAll">All sources</button></div>
  <h4>Legend</h4>
  <div class="lg"><span class="sw" style="background:#ef4444"></span>Source</div>
  <div class="lg"><span class="sw" style="background:#3b82f6"></span>Defined NBR</div>
  <div class="lg"><span class="sw" style="background:#f59e0b"></span>Missing</div>
  <div class="lg"><span class="sw" style="background:#9ca3af"></span>Other</div>
  <div class="lg"><span style="width:22px;border-top:2px solid #10b981"></span>HO link</div>
  <div class="lg"><span style="width:22px;border-top:2px dashed #f59e0b"></span>Missing link</div>
  </div>
</div>
<div id="mapbox"><div id="map"></div>
  <div id="mbtn"><button class="mb" id="tg">☰</button><button class="mb" id="btnSel">☑ Select</button><button class="mb" id="btnRul">📏 Ruler</button></div>
  <div id="rbox"><b style="color:#818cf8;font-size:11px">Ruler</b> <div id="rTot" style="margin:5px 0 2px;font-size:12px;font-weight:bold;color:#f87171">Total: 0.00 km / 0.00 mi</div><div id="rSeg" class="hint" style="font-size:10px"></div><div class="row" style="margin-top:5px"><button id="rUndo">Undo</button><button id="rClr">Clear</button></div></div>
  <div id="selbar"><span id="selN">0 selected</span><button id="selCopy">Copy</button><button id="selCsv">CSV</button><button id="selVis">Select visible</button><button id="selClr">Clear</button></div>
  <div id="toast"></div></div>
</div>
<div id="mdl"><div class="mbox"><div class="mh"><b>Selected sites</b><button id="mX">✕</button></div><div class="fm"><button data-f="excel" class="on">Excel</button><button data-f="usid">USID</button><button data-f="csv">CSV</button></div><textarea id="mTa" readonly></textarea><div class="mf"><span id="mN"></span><button id="mCopy">Copy</button><button id="mDl">Download</button></div></div></div>
<script>
const D = __DATA__;
const S = D.sites, CFG = D.cfg, ALL = Object.keys(S);
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const COL = {source:'#ef4444', ok:'#3b82f6', missing:'#f59e0b', other:'#9ca3af'};
const TXT = {source:'#f87171', ok:'#60a5fa', missing:'#f59e0b', other:'#9ca3af'};
const STAT = {ok:['Defined','#10b981'], zero_ho:['0% HO','#f59e0b'], no_coords:['Not in file','#ef4444'], not_defined:['Suggested','#8b5cf6']};
const WHY = {no_coords:'Defined in table, not in site file', zero_ho:'Defined but 0 HO', not_defined:'Closest site, not in table'};
const MI = 69.093;
function hav(la1, lo1, la2, lo2) { const r = Math.PI/180, a = Math.sin((la2-la1)*r/2)**2 + Math.cos(la1*r)*Math.cos(la2*r)*Math.sin((lo2-lo1)*r/2)**2; return 2*3958.8*Math.asin(Math.sqrt(a)); }
function toast(msg) { const t = $('toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', 3000); }
function analyse(u) {
  const s = S[u], entry = D.nbr[u], rows = entry ? entry[1] : [];
  const defd = {};
  rows.forEach(r => { const n = r[0]; if (n === u) return; if (!(n in defd) || (r[1] || 0) > (defd[n][0] || 0)) defd[n] = [r[1], r[2]]; });
  const dist = {}; ALL.forEach(k => dist[k] = hav(s[1], s[2], S[k][1], S[k][2]));
  const mk = (n, st, ho, pct) => ({u:n, st, ho, pct, n:S[n] ? S[n][0] : n, d:S[n] ? Math.round(dist[n]*100)/100 : null, lat:S[n] ? S[n][1] : null, lon:S[n] ? S[n][2] : null});
  const nbrs = [];
  Object.keys(defd).forEach(n => { const ho = defd[n][0], pct = defd[n][1]; let st = 'ok'; if (!S[n]) st = 'no_coords'; else if (CFG.zeroHo && ((ho != null && ho === 0) || (pct != null && pct === 0))) st = 'zero_ho'; nbrs.push(mk(n, st, ho, pct)); });
  if (CFG.closestN) ALL.filter(k => k !== u).sort((a, b) => dist[a] - dist[b]).slice(0, CFG.closestN).forEach(k => { if (!(k in defd)) nbrs.push(mk(k, 'not_defined', null, null)); });
  nbrs.sort((a, b) => ((a.st !== 'ok') - (b.st !== 'ok')) || ((b.ho || 0) - (a.ho || 0)));
  return {src:u, cluster:entry ? entry[0] : '', defined:Object.keys(defd).length, nbrs, dist};
}
const map = L.map('map').setView([39.5, -98.35], 4);
const base = {
  'Topo': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {maxZoom:19}),
  'Street': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19}),
  'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:19}),
  'Light': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {maxZoom:19})
};
base.Topo.addTo(map); L.control.layers(base, null, {position:'topright'}).addTo(map);
map.createPane('poly'); map.getPane('poly').style.zIndex = 450;
const mapEl = map.getContainer();
const zoomCls = () => mapEl.classList.toggle('zhi', map.getZoom() >= 12);
map.on('zoomend', zoomCls); zoomCls();
const G = {other:L.layerGroup(), missing:L.layerGroup(), ok:L.layerGroup(), source:L.layerGroup()};
const LG = {ok:L.layerGroup(), missing:L.layerGroup()};
Object.values(G).concat(Object.values(LG)).forEach(g => g.addTo(map));
function recolor() { const blue = $('vBlue').checked; const c = role => blue && role !== 'source' ? COL.ok : COL[role]; wedges.forEach(w => w.p.setStyle({fillColor:c(w.role)})); }
function setLayer(g, on) { on ? map.addLayer(g) : map.removeLayer(g); }
function applyVis() { setLayer(G.other, $('vOther').checked); setLayer(G.missing, $('vMiss').checked); setLayer(LG.ok, $('vLines').checked); setLayer(LG.missing, $('vLines').checked && $('vMiss').checked); mapEl.classList.toggle('nolbl', !$('vLbl').checked); }
['vLines','vMiss','vOther','vLbl'].forEach(id => $(id).addEventListener('change', applyVis));
$('vBlue').addEventListener('change', recolor);
function wedge(lat, lon, az, r) { const la = lat*Math.PI/180, lo = lon*Math.PI/180, d = r/6371000, pts = [[lat, lon]]; for (let i = 0; i <= 13; i++) { const b = (az - 32.5 + i*5)*Math.PI/180; const pl = Math.asin(Math.sin(la)*Math.cos(d) + Math.cos(la)*Math.sin(d)*Math.cos(b)); const pn = lo + Math.atan2(Math.sin(b)*Math.sin(d)*Math.cos(la), Math.cos(d) - Math.sin(la)*Math.sin(pl)); pts.push([pl*180/Math.PI, pn*180/Math.PI]); } return pts; }
let LAT0 = 0, LON0 = 0;
const toXY = (lat, lon) => [(lon - LON0)*Math.cos(LAT0*Math.PI/180)*MI, (lat - LAT0)*MI];
const toLL = (x, y) => [LAT0 + y/MI, LON0 + x/(Math.cos(LAT0*Math.PI/180)*MI)];
function pip(p, poly) { let inside = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1]; if (((yi > p[1]) !== (yj > p[1])) && (p[0] < (xj - xi)*(p[1] - yi)/(yj - yi) + xi)) inside = !inside; } return inside; }
function segInfo(p, a, b) { const dx = b[0]-a[0], dy = b[1]-a[1], L2 = dx*dx + dy*dy; let t = L2 ? ((p[0]-a[0])*dx + (p[1]-a[1])*dy)/L2 : 0; t = Math.max(0, Math.min(1, t)); const q = [a[0] + t*dx, a[1] + t*dy]; return {q, t, d:Math.hypot(p[0]-q[0], p[1]-q[1]), L:Math.sqrt(L2)}; }
function convexHull(points) { const seen = new Set(), pts = points.filter(p => { const k = p.join(','); if (seen.has(k)) return false; seen.add(k); return true; }).sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (pts.length <= 2) return pts; const cross = (o, a, b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]); const half = arr => { const h = []; arr.forEach(p => { while (h.length >= 2 && cross(h[h.length-2], h[h.length-1], p) <= 0) h.pop(); h.push(p); }); h.pop(); return h; }; return half(pts).concat(half(pts.slice().reverse())); }
function offsetMiles(lat, lon, miles, bearingDeg) { const la = lat*Math.PI/180, lo = lon*Math.PI/180, b = bearingDeg*Math.PI/180, d = miles/3958.8; const nl = Math.asin(Math.sin(la)*Math.cos(d) + Math.cos(la)*Math.sin(d)*Math.cos(b)); const nn = lo + Math.atan2(Math.sin(b)*Math.sin(d)*Math.cos(la), Math.cos(d) - Math.sin(la)*Math.sin(nl)); return [nl*180/Math.PI, nn*180/Math.PI]; }
function bufferHull(hull, miles) { const cLat = hull.reduce((a, p) => a + p[0], 0)/hull.length, cLon = hull.reduce((a, p) => a + p[1], 0)/hull.length; return hull.map(p => offsetMiles(p[0], p[1], miles, Math.atan2(p[1]-cLon, p[0]-cLat)*180/Math.PI)); }
function carvePolygon(poly, targets, members, W, M) { let done = 0, skipped = 0; const gone = []; const bdist = (pl, t) => Math.min(...pl.map((a, i) => segInfo(t, a, pl[(i+1) % pl.length]).d)); targets.map(t => ({p:t, d:bdist(poly, t)})).sort((a, b) => a.d - b.d).forEach(o => { const p = o.p; if (!pip(p, poly)) { gone.push(p); return; } const edges = poly.map((a, i) => Object.assign({i, a, b:poly[(i+1) % poly.length]}, segInfo(p, a, poly[(i+1) % poly.length]))).filter(e => e.d > 1e-6 && e.L > 1e-6).sort((x, y) => x.d - y.d).slice(0, 4); for (const e of edges) { const ux = (p[0]-e.q[0])/e.d, uy = (p[1]-e.q[1])/e.d; const tip = [p[0] + ux*M, p[1] + uy*M], lerp = t => [e.a[0] + (e.b[0]-e.a[0])*t, e.a[1] + (e.b[1]-e.a[1])*t]; const ins = []; if (e.t*e.L > W) ins.push(lerp(e.t - W/e.L)); ins.push(tip); if ((1-e.t)*e.L > W) ins.push(lerp(e.t + W/e.L)); const cand = poly.slice(0, e.i + 1).concat(ins, poly.slice(e.i + 1)); if (!pip(p, cand) && members.every(m => pip(m, cand)) && gone.every(g => !pip(g, cand))) { poly = cand; gone.push(p); done++; return; } } skipped++; }); return {poly, done, skipped}; }
let cur = null, wedges = [], dots = [], siteLayers = {}, radius = CFG.radiusM, incl = {}, polyPts = [], dirty = false, handles = [], lastBuf = CFG.buffer, filter = 'all';
const poly = L.polygon([], {pane:'poly', color:'#818cf8', weight:3, fill:false, bubblingMouseEvents:false}).addTo(map);
function tip(u, extra) { const s = S[u], n = cur && cur.NB[u]; let h = '<b>' + esc(s[0]) + '</b><br>USID: ' + esc(u); if (cur && u === cur.src) h += '<br><span style="color:#f87171">SOURCE</span>'; else if (n) { h += '<br>' + STAT[n.st][0]; if (n.pct != null) h += ' · HO ' + n.pct.toFixed(1) + '%'; if (n.d != null) h += ' · ' + n.d + ' mi'; } return h + (extra || ''); }
const popup = (u, extra) => tip(u, extra) + '<br><a href="#" onclick="loadSource(' + String(u).replace(/'/g, '') + ');return false" style="color:#818cf8">Load as source</a>';
function loadSource(u) {
  u = String(u).trim().toUpperCase(); if (!S[u]) { toast('USID not found'); return; }
  const A = analyse(u); A.NB = {}; A.nbrs.forEach(n => A.NB[n.u] = n);
  A.role = {}; A.nbrs.forEach(n => { if (S[n.u]) A.role[n.u] = n.st === 'ok' ? 'ok' : 'missing'; }); A.role[u] = 'source';
  cur = A; LAT0 = S[u][1]; LON0 = S[u][2];
  Object.values(G).concat(Object.values(LG)).forEach(g => g.clearLayers()); wedges = [];
  const src = S[u]; const draw = ALL.filter(k => A.dist[k] <= CFG.maxMiles || A.role[k]); A.draw = draw; dots = []; siteLayers = {};
  ['other', 'missing', 'ok', 'source'].forEach(role => {
    draw.filter(k => (A.role[k] || 'other') === role).forEach(k => {
      const s = S[k], g = G[role], sl = siteLayers[k] = {wedges:[]}, onClick = e => siteClick(k, e);
      s[3].forEach(c => {
        const p = L.polygon(wedge(s[1], s[2], c[1], radius), {color:'#000', weight:1, fillColor:COL[role], fillOpacity:.7});
        p.bindTooltip(tip(k, '<br>Cell: ' + esc(c[0]) + ' (' + c[1] + '°)'), {sticky:true});
        p.bindPopup(popup(k, '<br>Cell: ' + esc(c[0]) + ' (' + c[1] + '°)'));
        p.on('click', onClick); sl.wedges.push(p); g.addLayer(p); wedges.push({p, s, az:c[1], role});
      });
      g.addLayer(L.marker([s[1], s[2]], {interactive:false, icon:L.divIcon({className:'', iconSize:[150, 20], iconAnchor:[-8, 10], html:'<div class="lbl lbl-' + role + '" style="color:' + TXT[role] + '">' + esc(s[0]) + '</div>'})}));
    });
  });
  A.nbrs.forEach(n => { if (!S[n.u]) return; const miss = n.st !== 'ok', s = S[n.u]; L.polyline([[src[1], src[2]], [s[1], s[2]]], miss ? {color:'#f59e0b', weight:1.8, dashArray:'6,5', opacity:.85} : {color:'#10b981', weight:1.8, opacity:.75}).addTo(miss ? LG.missing : LG.ok); });
  applyVis(); recolor(); selected.forEach(styleSel);
  incl = {}; incl[u] = true; A.nbrs.forEach(n => { if (S[n.u]) incl[n.u] = (n.st === 'ok'); });
  $('empty').style.display = 'none'; $('panel').style.display = ''; $('q').value = u;
  $('tSrc').textContent = 'Source: ' + u + ' (' + src[0] + ')'; $('tSub').textContent = A.cluster || ''; $('tInfo').textContent = src[1].toFixed(5) + ', ' + src[2].toFixed(5) + ' | ' + src[3].length + ' cell(s)';
  const c = st => A.nbrs.filter(n => n.st === st).length;
  $('stats').innerHTML = [['Defined', A.defined, '#3b82f6'], ['OK', c('ok'), '#10b981'], ['Issue', c('no_coords') + c('zero_ho'), '#f59e06'], ['Suggested', c('not_defined'), '#8b5cf6']].map(x => '<div><b style="color:' + x[2] + '">' + x[1] + '</b>' + x[0] + '</div>').join('');
  $('warn').innerHTML = ''; filter = 'all'; renderPills(); renderList(); buildPoly(); fit();
}
function buildPoly() { const pts = Object.keys(incl).filter(k => incl[k] && S[k]).map(k => [S[k][1], S[k][2]]); const hull = convexHull(pts); polyPts = hull.length >= 3 ? bufferHull(hull, parseFloat($('buf').value)) : []; poly.setLatLngs(polyPts); applyExclusion(); dirty = false; makeHandles(); updateInside(); }
function tryRebuild() { if (dirty && !confirm('Discard manual edits?')) return false; buildPoly(); return true; }
function updateInside() { if (!cur || !polyPts.length) { $('inside').innerHTML = ''; return; } const P = polyPts.map(p => toXY(p[0], p[1])); const ins = Object.keys(cur.role).filter(k => cur.role[k] === 'missing' && !incl[k] && pip(toXY(S[k][1], S[k][2]), P)); $('inside').innerHTML = ins.length ? '<span style="color:#f59e0b">' + ins.length + ' missing inside</span>' : '<span style="color:#10b981">No missing inside</span>'; }
function clearHandles() { handles.forEach(h => map.removeLayer(h)); handles = []; }
function makeHandles() { clearHandles(); if (!$('chkEdit').checked || !$('chkShow').checked) return; polyPts.forEach((pt, i) => { const h = L.marker(pt, {draggable:true, zIndexOffset:1000, icon:L.divIcon({className:'', html:'<div class="vh"></div>', iconSize:[10, 10], iconAnchor:[5, 5]})}).addTo(map); h.on('drag', e => { const ll = e.target.getLatLng(); polyPts[i] = [ll.lat, ll.lng]; poly.setLatLngs(polyPts); dirty = true; }); h.on('dragend', updateInside); h.on('contextmenu', () => { if (polyPts.length > 3) { polyPts.splice(i, 1); poly.setLatLngs(polyPts); dirty = true; makeHandles(); updateInside(); } }); handles.push(h); }); }
poly.on('click', e => { if (!$('chkEdit').checked || polyPts.length < 3) return; const P = ll => map.latLngToLayerPoint(ll), c = P(e.latlng); let best = 1, min = Infinity; polyPts.forEach((a, i) => { const d = L.LineUtil.pointToSegmentDistance(c, P(a), P(polyPts[(i+1) % polyPts.length])); if (d < min) { min = d; best = i + 1; } }); polyPts.splice(best, 0, [e.latlng.lat, e.latlng.lng]); poly.setLatLngs(polyPts); dirty = true; makeHandles(); updateInside(); });
function applyExclusion() { const kind = $('excl').value; if (kind === 'none' || !cur || polyPts.length < 3) return; const members = Object.keys(incl).filter(k => incl[k] && S[k]).map(k => toXY(S[k][1], S[k][2])); const keys = kind === 'all' ? cur.draw.filter(k => k !== cur.src) : Object.keys(cur.role).filter(k => cur.role[k] === 'missing'); const targets = keys.filter(k => !incl[k]).map(k => toXY(S[k][1], S[k][2])); const r = carvePolygon(polyPts.map(p => toXY(p[0], p[1])), targets, members, 1.0, 0.3); polyPts = r.poly.map(p => toLL(p[0], p[1])); poly.setLatLngs(polyPts); }
$('chkShow').addEventListener('change', e => { setLayer(poly, e.target.checked); makeHandles(); });
$('chkEdit').addEventListener('change', e => { if (e.target.checked && !$('chkShow').checked) { $('chkShow').checked = true; map.addLayer(poly); } makeHandles(); });
$('buf').value = CFG.buffer; $('bufVal').textContent = CFG.buffer;
$('buf').addEventListener('input', e => $('bufVal').textContent = e.target.value);
$('buf').addEventListener('change', e => { if (tryRebuild()) lastBuf = e.target.value; else { e.target.value = lastBuf; $('bufVal').textContent = lastBuf; } });
$('btnRebuild').addEventListener('click', tryRebuild); $('excl').addEventListener('change', tryRebuild);
$('pw').addEventListener('input', e => { $('wVal').textContent = e.target.value; poly.setStyle({weight:+e.target.value}); });
$('ps').addEventListener('change', e => poly.setStyle({dashArray:e.target.value || null}));
$('pc').addEventListener('input', e => poly.setStyle({color:e.target.value}));
$('ss').addEventListener('input', e => { $('sVal').textContent = (+e.target.value).toFixed(1); radius = CFG.radiusM * e.target.value; wedges.forEach(w => w.p.setLatLngs(wedge(w.s[1], w.s[2], w.az, radius))); });
$('fs').addEventListener('input', e => { $('fVal').textContent = e.target.value; mapEl.style.setProperty('--fs', e.target.value + 'px'); });
function fit() { if (!cur) return; const b = L.latLngBounds([]); if (polyPts.length && $('chkShow').checked) polyPts.forEach(p => b.extend(p)); Object.keys(cur.role).forEach(k => b.extend([S[k][1], S[k][2]])); if (b.isValid()) map.fitBounds(b, {padding:[30, 30]}); }
$('btnFit').addEventListener('click', fit); $('tg').addEventListener('click', () => { const s = $('side'); s.style.display = s.style.display === 'none' ? '' : 'none'; setTimeout(() => map.invalidateSize(), 50); });
const PILLS = [['all', 'All'], ['ok', 'Defined OK'], ['miss', 'Missing / suggested']];
function renderPills() { $('pills').innerHTML = PILLS.map(p => '<button data-f="' + p[0] + '" class="' + (filter === p[0] ? 'on' : '') + '">' + p[1] + '</button>').join(''); }
function renderList() { const rows = cur.nbrs.filter(n => filter === 'all' || (filter === 'ok' && n.st === 'ok') || (filter === 'miss' && n.st !== 'ok')); $('list').innerHTML = rows.length ? rows.map(n => { const has = !!S[n.u]; const meta = [n.ho != null ? (n.ho === 0 ? '<span style="color:#f59e0b">HO 0</span>' : 'HO ' + n.ho) : '', n.pct != null ? n.pct.toFixed(1) + '%' : '', n.d != null ? n.d + ' mi' : ''].filter(Boolean).join(' · '); return '<div class="nb" data-u="' + esc(n.u) + '"><div class="nbt">' + (has ? '<input type="checkbox" class="inc" data-u="' + esc(n.u) + '"' + (incl[n.u] ? ' checked' : '') + '>' : '<span style="width:13px"></span>') + '<b>' + esc(n.u) + '</b><span class="nm">' + esc(n.n) + '</span><span class="bd" style="background:' + STAT[n.st][1] + '">' + STAT[n.st][0] + '</span></div>' + (meta ? '<div class="meta">' + meta + '</div>' : '') + '</div>'; }).join('') : '<div class="hint">Nothing to show.</div>'; }
$('pills').addEventListener('click', e => { if (e.target.dataset.f) { filter = e.target.dataset.f; renderPills(); renderList(); } });
$('list').addEventListener('change', e => { if (!e.target.classList.contains('inc')) return; const u = e.target.dataset.u; incl[u] = e.target.checked; if (!tryRebuild()) { e.target.checked = !e.target.checked; incl[u] = e.target.checked; } });
$('list').addEventListener('click', e => { if (e.target.closest('.inc')) return; const row = e.target.closest('.nb'); if (!row) return; const s = S[row.dataset.u]; if (s) map.flyTo([s[1], s[2]], Math.max(map.getZoom(), 13)); });
function csvDownload(rows, name) { const q = v => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; const csv = [['Source_USID','Missing_USID','Site_Name','Reason','HO_ATT','Pct_Sharing','Dist_miles','Latitude','Longitude']].concat(rows).map(r => r.map(q).join(',')).join('\n'); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'})); a.download = name; a.click(); }
const missRows = A => A.nbrs.filter(n => n.st !== 'ok').map(n => [A.src, n.u, n.n, WHY[n.st], n.ho, n.pct, n.d, n.lat, n.lon]);
$('btnCsv').addEventListener('click', () => { const rows = missRows(cur); if (!rows.length) { toast('No missing'); return; } csvDownload(rows, 'missing_' + cur.src + '.csv'); });
$('btnCsvAll').addEventListener('click', () => { toast('Analysing all...'); setTimeout(() => { let rows = []; Object.keys(D.nbr).forEach(u => { if (S[u]) rows = rows.concat(missRows(analyse(u))); }); if (!rows.length) { toast('No missing'); return; } csvDownload(rows, 'missing_all.csv'); toast('Exported ' + rows.length + ' rows'); }, 50); });
let acIdx = -1;
function acHits(v) { v = v.trim().toUpperCase(); if (v.length < 2) return []; const starts = [], inc = []; for (const k of ALL) { const nm = String(S[k][0]).toUpperCase(); if (k === v || k.startsWith(v) || nm.startsWith(v)) starts.push(k); else if (k.includes(v) || nm.includes(v)) inc.push(k); if (starts.length >= 40) break; } return starts.concat(inc).slice(0, 40); }
function renderAc() { const hits = acHits($('q').value), dd = $('ac'); acIdx = -1; if (!hits.length) { dd.style.display = 'none'; return; } dd.innerHTML = hits.map(k => '<div class="ai" data-u="' + esc(k) + '"><b>' + esc(k) + '</b><span>' + esc(S[k][0]) + '</span></div>').join(''); dd.style.display = 'block'; }
function pick(u) { $('ac').style.display = 'none'; loadSource(u); }
$('q').addEventListener('input', renderAc);
$('q').addEventListener('keydown', e => { const items = $('ac').querySelectorAll('.ai'); if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { if (!items.length) return; e.preventDefault(); acIdx = Math.max(0, Math.min(items.length - 1, acIdx + (e.key === 'ArrowDown' ? 1 : -1))); items.forEach((el, i) => el.classList.toggle('sel', i === acIdx)); } else if (e.key === 'Enter') { e.preventDefault(); const v = $('q').value.trim().toUpperCase(); if (acIdx >= 0 && items[acIdx]) pick(items[acIdx].dataset.u); else if (S[v]) pick(v); else if (items.length) pick(items[0].dataset.u); } else if (e.key === 'Escape') $('ac').style.display = 'none'; });
$('ac').addEventListener('mousedown', e => { const it = e.target.closest('.ai'); if (it) { e.preventDefault(); pick(it.dataset.u); } });
document.addEventListener('click', e => { if (!e.target.closest('.srch')) $('ac').style.display = 'none'; });
const selected = new Set(); let selMode = false, rulerOn = false, rPts = [], fmtKind = 'excel';
const rLayer = L.layerGroup().addTo(map); const rLine = L.polyline([], {color:'#f87171', weight:3, dashArray:'8,5', interactive:false});
function setMode(m) { selMode = m === 'sel'; rulerOn = m === 'ruler'; $('btnSel').classList.toggle('on', selMode); $('btnRul').classList.toggle('on', rulerOn); mapEl.classList.toggle('crosshair', selMode || rulerOn); $('rbox').style.display = (rulerOn || rPts.length) ? 'block' : 'none'; updateSelBar(); }
$('btnSel').addEventListener('click', () => setMode(selMode ? null : 'sel')); $('btnRul').addEventListener('click', () => setMode(rulerOn ? null : 'ruler'));
function siteClick(k, e) { if (selMode) { L.DomEvent.stopPropagation(e); map.closePopup(); toggleSel(k); } else if (rulerOn) { L.DomEvent.stopPropagation(e); map.closePopup(); addRulerPoint([S[k][1], S[k][2]]); } }
function styleSel(k) { const sl = siteLayers[k]; if (!sl) return; const on = selected.has(k); sl.wedges.forEach(p => p.setStyle(on ? {color:'#f59e0b', weight:3} : {color:'#000', weight:1})); }
function toggleSel(k) { selected.has(k) ? selected.delete(k) : selected.add(k); styleSel(k); updateSelBar(); }
function updateSelBar() { $('selN').textContent = selected.size + ' selected'; $('selbar').style.display = (selMode || selected.size) ? 'flex' : 'none'; }
function selectVisible() { if (!cur) return; const b = map.getBounds(); let n = 0; cur.draw.forEach(k => { const r = cur.role[k] || 'other'; if ((r === 'other' && !$('vOther').checked) || (r === 'missing' && !$('vMiss').checked)) return; if (b.contains([S[k][1], S[k][2]])) { selected.add(k); styleSel(k); n++; } }); updateSelBar(); toast(n + ' visible selected'); }
function clearSel() { const old = Array.from(selected); selected.clear(); old.forEach(styleSel); updateSelBar(); }
$('selVis').addEventListener('click', selectVisible); $('selClr').addEventListener('click', clearSel);
const SEL_HDR = ['USID','Site_Name','Latitude','Longitude','Role','HO_ATT','Pct_Sharing','Dist_mi','Dist_km'];
function selRows() { return Array.from(selected).sort().map(k => { const s = S[k], n = cur && cur.NB[k]; const role = cur && k === cur.src ? 'Source' : n ? STAT[n.st][0] : 'Other'; const d = cur && cur.dist[k] != null ? cur.dist[k] : null; return [k, s[0], s[1], s[2], role, n ? n.ho : '', n ? n.pct : '', d != null ? d.toFixed(2) : '', d != null ? (d*1.609344).toFixed(2) : '']; }); }
function selText(kind) { const rows = selRows(), q = v => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; if (kind === 'usid') return rows.map(r => r[0]).join('\n'); if (kind === 'csv') return [SEL_HDR].concat(rows).map(r => r.map(q).join(',')).join('\n'); return [SEL_HDR].concat(rows).map(r => r.map(v => v == null ? '' : v).join('\t')).join('\n'); }
function downloadText(text, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type:'text/csv'})); a.download = name; a.click(); }
function openModal() { if (!selected.size) { toast('No sites selected'); return; } $('mdl').style.display = 'flex'; showFmt(fmtKind); }
function showFmt(kind) { fmtKind = kind; $('mTa').value = selText(kind); $('mN').textContent = selected.size + ' site(s)'; document.querySelectorAll('.fm button').forEach(b => b.classList.toggle('on', b.dataset.f === kind)); }
function copyText(text) { const done = () => toast('Copied to clipboard'); const fallback = () => { const ta = $('mTa'); ta.value = text; ta.select(); document.execCommand('copy'); done(); }; if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback(); }
$('selCopy').addEventListener('click', openModal); $('selCsv').addEventListener('click', () => { if (!selected.size) { toast('No sites selected'); return; } downloadText(selText('csv'), 'selected_sites.csv'); });
$('mX').addEventListener('click', () => $('mdl').style.display = 'none'); $('mdl').addEventListener('click', e => { if (e.target === $('mdl')) $('mdl').style.display = 'none'; });
document.querySelector('.fm').addEventListener('click', e => { if (e.target.dataset.f) showFmt(e.target.dataset.f); });
$('mCopy').addEventListener('click', () => copyText(selText(fmtKind))); $('mDl').addEventListener('click', () => downloadText(selText('csv'), 'selected_sites.csv'));
const dtxt = mi => (mi*1.609344).toFixed(2) + ' km / ' + mi.toFixed(2) + ' mi';
function drawRuler() { rLayer.clearLayers(); rLine.setLatLngs(rPts); rLayer.addLayer(rLine); let cum = 0, last = 0; rPts.forEach((p, i) => { if (i > 0) { last = hav(rPts[i-1][0], rPts[i-1][1], p[0], p[1]); cum += last; } const m = L.circleMarker(p, {radius:4, color:'#f87171', fillColor:'#fff', fillOpacity:1, weight:2, interactive:false}); if (i > 0) m.bindTooltip(dtxt(cum), {permanent:true, direction:'top', offset:[0, -6], className:'rtip'}); rLayer.addLayer(m); }); $('rTot').textContent = 'Total: ' + dtxt(cum); $('rSeg').textContent = rPts.length > 1 ? 'Last: ' + dtxt(last) + ' | ' + rPts.length + ' pts' : (rPts.length ? '1 point' : ''); }
function addRulerPoint(ll) { rPts.push(ll); drawRuler(); }
map.on('click', e => { if (rulerOn) addRulerPoint([e.latlng.lat, e.latlng.lng]); });
$('rUndo').addEventListener('click', () => { rPts.pop(); drawRuler(); }); $('rClr').addEventListener('click', () => { rPts = []; drawRuler(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !(e.target.tagName === 'INPUT')) { setMode(null); $('mdl').style.display = 'none'; } });
(function init() { if (ALL.length) { const b = L.latLngBounds(ALL.map(k => [S[k][1], S[k][2]])); map.fitBounds(b); } })();
</script></body></html>"""
                
                final_html = HTML.replace("__DATA__", data)
                components.html(final_html, height=850, scrolling=False)
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with st.container():
        st.markdown('<div class="map-container" style="padding:40px;text-align:center;color:#64748b">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload both files to generate the interactive map")
        st.markdown("1. **Site Data** (CSV) - Contains USID, coordinates, azimuth, cell info<br>2. **SQL Table** (Excel/CSV) - Contains neighbor relationships and HO data")
        st.markdown('</div>', unsafe_allow_html=True)
