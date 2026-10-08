"""
Cluster Neighbor Visualization Tool  -  Streamlit app

Layout (top -> bottom):  header  |  upload bar (Site CSV, SQL table, Name, User ID, [Go])  |  sidebar + map  |  footer
Flow: upload both files, enter Name + User ID, press  Go  -> loading message -> map.

Usage logging (who searched which USID, when): set LOG_WEBHOOK_URL (+ LOG_TOKEN) in .streamlit/secrets.toml
 (see usage_log_apps_script.gs for a free Google-Sheet receiver). Leave empty to disable.
"""
import io
import json
import math
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ───────────────────────── CONFIG ─────────────────────────
APP_TITLE = "Cluster Neighbor Visualization Tool"
SUPPORT_NAME = "@Rajesh Dubey"
SUPPORT_EMAIL = "rajesh.dubey@sds.com"

MAX_DISTANCE_MILES, BUFFER_MILES, CLOSEST_N, ZERO_HO_IS_MISSING, WEDGE_RADIUS_M = 25.0, 1.8, 15, False, 400
USID_COL, SITE_NAME_COL, LAT_COL, LON_COL, AZIMUTH_COL, CELL_COL = "USID", "ENODEB_New", "LATITUDE", "LONGITUDE", "AZIMUTH", "CELL"
NBR_SRC_COL, NBR_COL, CLUSTER_NAME_COL, HO_COL, PCT_COL = "USID", "ClusterUSID", "ClusterName", "HO ATT", "% Sharing"
# ──────────────────────────────────────────────────────────


def secret(name):
    """Read from .streamlit/secrets.toml (Streamlit Cloud 'Secrets') or an environment variable."""
    try:
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name, "")


def auth_email():
    """Viewer e-mail when the app is private / login-protected on Streamlit Cloud (empty for public apps)."""
    for attr in ("user", "experimental_user"):
        try:
            u = getattr(st, attr, None)
            e = getattr(u, "email", None) or (u.get("email") if hasattr(u, "get") else None)
            if e:
                return str(e)
        except Exception:
            pass
    return ""


SIGNUM_RE = re.compile(r"[A-Za-z0-9_.-]{3,20}")


def clean_signum(s):
    s = (s or "").strip().lower()
    return s if SIGNUM_RE.fullmatch(s) else ""


def norm_id(x):
    s = str(x).strip().upper()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def num(v):
    try:
        return float(v) if pd.notna(v) else 0.0
    except Exception:
        return 0.0


def need_cols(df, cols, label):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{label}: column(s) {miss} not found. Columns present: {list(df.columns)[:15]}")


def read_upload(f):
    raw = io.BytesIO(f.getvalue())
    df = pd.read_csv(raw, low_memory=False) if f.name.lower().endswith(".csv") else pd.read_excel(raw)
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_sites(df):
    need_cols(df, [USID_COL, LAT_COL, LON_COL, AZIMUTH_COL], "Site file")
    df = df.dropna(subset=[USID_COL]).copy()
    df[USID_COL] = df[USID_COL].map(norm_id)
    for c in (LAT_COL, LON_COL, AZIMUTH_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[LAT_COL, LON_COL])
    if SITE_NAME_COL in df.columns:
        names = df[SITE_NAME_COL].where(df[SITE_NAME_COL].notna(), df[USID_COL]).astype(str).str.strip()
        df["_name"] = names.where(names != "", df[USID_COL])
    else:
        df["_name"] = df[USID_COL]
    return df, df.groupby(USID_COL, sort=False).agg(lat=(LAT_COL, "first"), lon=(LON_COL, "first"), name=("_name", "first"))


def load_nbr_table(df):
    need_cols(df, [NBR_SRC_COL, NBR_COL], "SQL table")
    t = df.dropna(subset=[NBR_SRC_COL, NBR_COL]).copy()
    t[NBR_SRC_COL] = t[NBR_SRC_COL].map(norm_id)
    t[NBR_COL] = t[NBR_COL].map(norm_id)
    return t


def select_sites(sites, tbl, sources):
    lat, lon = sites.lat.values, sites.lon.values
    keep = np.zeros(len(sites), dtype=bool)
    dlat = MAX_DISTANCE_MILES / 69.0
    for s in sources:
        if s not in sites.index:
            continue
        la, lo = sites.at[s, "lat"], sites.at[s, "lon"]
        dlon = dlat / max(math.cos(math.radians(la)), 0.01)
        keep |= (np.abs(lat - la) <= dlat) & (np.abs(lon - lo) <= dlon)
    keep |= sites.index.isin(set(tbl.loc[tbl[NBR_SRC_COL].isin(sources), NBR_COL]))
    return sites[keep]


def build_payload(df, tbl, user=None, log_url="", log_token=""):
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
    site_json = {u: [r.name, round(float(r.lat), 6), round(float(r.lon), 6), cells.get(u, [])] for u, r in zip(keep.index, keep.itertuples())}
    has_ho, has_pct = HO_COL in tbl.columns, PCT_COL in tbl.columns
    nbr_json = {}
    for src, g in tbl.groupby(NBR_SRC_COL, sort=False):
        cname = str(g[CLUSTER_NAME_COL].dropna().iloc[0]).strip() if CLUSTER_NAME_COL in g.columns and g[CLUSTER_NAME_COL].notna().any() else ""
        hos = g[HO_COL].map(num) if has_ho else [None] * len(g)
        pcts = g[PCT_COL].map(num) if has_pct else [None] * len(g)
        nbr_json[src] = [cname, [[n, h, p] for n, h, p in zip(g[NBR_COL], hos, pcts)]]
    cfg = {"buffer": BUFFER_MILES, "radiusM": WEDGE_RADIUS_M, "maxMiles": MAX_DISTANCE_MILES, "closestN": CLOSEST_N,
           "zeroHo": ZERO_HO_IS_MISSING, "default": None, "user": user or {}, "logUrl": log_url, "logToken": log_token}
    return {"cfg": cfg, "sites": site_json, "nbr": nbr_json}


def render_html(payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML.replace("__DATA__", data)


def post_log(url, token, event, user, extra=""):
    """Server-side log line (used for the 'go' event). Never breaks the app."""
    if not url:
        return
    try:
        import requests
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        body = {"token": token, "event": event, "ts_utc": now.isoformat(), "ts_local": now.astimezone().strftime("%d/%m/%Y, %H:%M:%S"),
                "signum": user.get("id", ""), "auth_email": user.get("email", ""), "extra": extra}
        requests.post(url, data=json.dumps(body), headers={"Content-Type": "text/plain;charset=utf-8"}, timeout=6)
    except Exception:
        pass


# ───────────────────────── HTML / JS (map + side panel) ─────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Cluster NBR Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;font-family:'Segoe UI',system-ui,sans-serif;font-size:12px;color:#e2e8f0;background:#0a0e1a;overflow:hidden}
#app{display:flex;flex-direction:column;height:100%}
#hdr{background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);color:#fff;font-size:20px;font-weight:600;padding:10px 20px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between}
#hdr .sub{font-size:11px;opacity:0.85;font-weight:400}
#wrap{display:flex;flex:1;min-height:0;overflow:hidden}
#side{width:clamp(220px,14vw,290px);min-width:220px;overflow-y:auto;background:#0d1220;border-right:1px solid #1e293b;padding:10px;flex-shrink:0}
#side::-webkit-scrollbar{width:5px}#side::-webkit-scrollbar-track{background:#0a0e1a}#side::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
#mapbox{flex:1;position:relative;background:#020617;min-width:0}
#map{height:100%;width:100%;--fs:13px}

.section-title{font-size:10px;font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px;padding-bottom:4px;border-bottom:1px solid #1e293b}
.section-title:first-child{margin-top:0}
.search-label{font-size:10px;font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
#q{width:100%;box-sizing:border-box;padding:7px 10px;font-size:12px;background:#1e293b;border:1px solid #334155;border-radius:4px;color:#e2e8f0;outline:none;margin-bottom:8px}
#q:focus{border-color:#60a5fa}
#ac{position:absolute;top:100%;left:0;right:0;background:#1e293b;border:1px solid #334155;border-radius:4px;max-height:220px;overflow-y:auto;z-index:2000;display:none}
.ai{padding:6px 10px;cursor:pointer;display:flex;gap:6px;align-items:center;border-bottom:1px solid #0f172a;font-size:11px}
.ai:hover,.ai.sel{background:#334155}.ai b{color:#60a5fa}.ai span{flex:1;color:#94a3b8;font-size:10px}

.src-title{font-size:14px;font-weight:700;color:#f97316;margin-bottom:2px}
.src-sub{color:#94a3b8;font-size:11px;margin-bottom:2px}
.src-info{color:#64748b;font-size:10px;margin-bottom:10px}

.ctrl-row{display:flex;align-items:center;gap:6px;margin:6px 0;font-size:11px;color:#cbd5e1}
.ctrl-row label{cursor:pointer;display:flex;align-items:center;gap:5px;flex:1}
.ctrl-row input[type=checkbox]{accent-color:#60a5fa;margin:0}
.ctrl-row b{color:#e2e8f0}
.hint{font-size:9px;color:#64748b;margin:2px 0 6px 0}
input[type=range]{width:100%;accent-color:#60a5fa;height:3px;margin:4px 0}
select{width:100%;font-size:11px;padding:4px 6px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:3px;outline:none;margin:4px 0}
select:focus{border-color:#60a5fa}
input[type="color"]{width:32px;height:24px;border:1px solid #334155;border-radius:3px;background:#1e293b;cursor:pointer;padding:1px}
.ctrl-label{font-size:11px;color:#94a3b8;margin:6px 0 3px}

.btn-row{display:flex;gap:6px;margin:8px 0}
.btn{flex:1;cursor:pointer;border:1px solid #334155;background:#1e293b;color:#e2e8f0;border-radius:4px;padding:5px 8px;font-size:11px;font-weight:500;text-align:center;transition:all 0.15s}
.btn:hover{background:#334155;border-color:#60a5fa}
.btn-primary{background:#3b82f6;border-color:#3b82f6;color:#fff}

.stats{display:flex;gap:4px;margin:8px 0}
.stat-box{flex:1;background:#1e293b;border:1px solid #334155;border-radius:4px;padding:6px 4px;text-align:center}
.stat-box b{display:block;font-size:16px;font-weight:700}
.stat-box span{font-size:9px;color:#94a3b8;text-transform:uppercase}

#mbtn{position:absolute;top:10px;left:58px;z-index:1000;display:flex;gap:4px}
.mb{background:#0d1220;color:#e2e8f0;box-shadow:0 2px 6px rgba(0,0,0,0.5);border:1px solid #334155;padding:5px 10px;font-size:11px;border-radius:4px;cursor:pointer}
.mb:hover{background:#1e293b;border-color:#60a5fa}.mb.on{background:#3b82f6;color:#fff;border-color:#3b82f6}

#rbox{position:absolute;top:56px;right:10px;background:#0d1220;border:1px solid #334155;border-radius:6px;padding:10px 12px;z-index:1000;display:none;min-width:220px;box-shadow:0 4px 12px rgba(0,0,0,0.5)}
.rtip{background:#0d1220;border:1px solid #f87171;color:#f87171;font-weight:bold;font-size:10px;padding:2px 5px;border-radius:3px}

#selbar{position:absolute;bottom:56px;left:50%;transform:translateX(-50%);background:#0d1220;border:1px solid #334155;border-radius:6px;padding:8px 12px;z-index:1000;display:none;align-items:center;gap:8px;box-shadow:0 4px 12px rgba(0,0,0,0.5)}
#selN{font-weight:bold;color:#f59e0b;min-width:70px;font-size:11px}#selbar .btn{flex:none}#drawCtl{display:none;gap:6px;align-items:center}

#toast{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);background:#10b981;color:#fff;padding:8px 16px;border-radius:5px;z-index:1000;display:none;font-weight:500;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.4)}

.lbl{font-weight:600;font-size:var(--fs);white-space:nowrap;text-shadow:1px 1px 3px #000,-1px -1px 3px #000,1px -1px 3px #000,-1px 1px 3px #000}
.lbl-source{font-size:calc(var(--fs) + 2px)}.lbl-other{display:none}.zhi .lbl-other{display:block}.nolbl .lbl{display:none!important}.allblue .lbl-ok,.allblue .lbl-missing,.allblue .lbl-other{color:#3b82f6!important}

.vh{width:10px;height:10px;background:#fff;border:2px solid #60a5fa;border-radius:50%;cursor:move}
.crosshair,.crosshair.leaflet-grab{cursor:crosshair!important}

#mdl{position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:5000;display:none;align-items:center;justify-content:center}
.mbox{background:#1e293b;border-radius:8px;width:520px;max-width:95vw;max-height:85vh;display:flex;flex-direction:column;border:1px solid #334155}
.mh,.fm,.mf{padding:12px 16px;display:flex;align-items:center;gap:8px}
.mh{justify-content:space-between;border-bottom:1px solid #0f172a;font-size:13px;color:#60a5fa;font-weight:600}
.fm button.on{background:#3b82f6;color:#fff;border-color:#3b82f6}
.mf{border-top:1px solid #0f172a}.mf span{flex:1;color:#94a3b8;font-size:11px}
#mTa{margin:0 16px;height:220px;font-family:Consolas,monospace;font-size:11px;resize:none;box-sizing:border-box;background:#0a0e1a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:8px}

#ftr{background:#0d1220;color:#94a3b8;font-size:11px;padding:8px 20px;border-top:1px solid #1e293b;flex-shrink:0;display:flex;justify-content:flex-end;align-items:center}
#ftr a{color:#60a5fa;text-decoration:none}#ftr a:hover{text-decoration:underline}

.warn{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#fca5a5;border-radius:4px;padding:5px 8px;margin-top:6px;font-size:10px}
.inside-ok{color:#10b981;font-size:10px;margin-top:4px}
.inside-warn{color:#f59e0b;font-size:10px;margin-top:4px}

.nb{border:1px solid #334155;border-radius:4px;padding:6px;margin:4px 0;cursor:pointer;background:#1e293b;font-size:11px}
.nb:hover{border-color:#60a5fa}
.nbt{display:flex;align-items:center;gap:5px}
.nbt .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#94a3b8;font-size:10px}
.bd{color:#fff;font-size:9px;padding:1px 6px;border-radius:8px;white-space:nowrap;font-weight:600}
.meta{font-size:9px;color:#64748b;margin:2px 0 0 20px}
.pills{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px}
.pills button{border-radius:10px;padding:2px 8px;font-size:9px;border:1px solid #334155;background:#1e293b;color:#cbd5e1;cursor:pointer}
.pills button.on{background:#3b82f6;color:#fff;border-color:#3b82f6}

#loading{position:fixed;inset:0;background:rgba(10,14,26,.88);z-index:9000;display:flex;align-items:center;justify-content:center}
.ov-box{text-align:center;color:#e2e8f0}.ov-t{font-size:16px;font-weight:600;margin-top:14px}.ov-s{font-size:11px;color:#94a3b8;margin-top:4px}
.spin{width:46px;height:46px;border:5px solid #1e293b;border-top-color:#3b82f6;border-radius:50%;margin:0 auto;animation:sp 0.9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
/* Config panel */
.cfg-panel{background:#1e293b;border:1px solid #334155;border-radius:4px;margin-bottom:12px;overflow:hidden}
.cfg-header{background:#334155;padding:8px 10px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;font-size:11px;font-weight:600;color:#60a5fa}
.cfg-content{padding:10px;display:none}
.cfg-content.active{display:block}
.cfg-content input[type=text]{width:100%;padding:5px 7px;background:#0a0e1a;border:1px solid #334155;border-radius:3px;color:#e2e8f0;font-size:11px;box-sizing:border-box;margin:4px 0}
</style></head><body>
<div id="app">
<div id="wrap">
<div id="side">
  <div class="search-label">PROVIDE USID</div>
  <div style="position:relative" class="srch">
    <input id="q" title="Provide USID" placeholder="Enter USID / site name" autocomplete="off">
    <div id="ac"></div>
  </div>
  <div id="empty" style="color:#64748b;font-size:10px;margin-top:4px">Type a USID above, or click any site on the map.</div>

  <div id="panel" style="display:none">
    <div class="src-title" id="tSrc"></div>
    <div class="src-sub" id="tSub"></div>
    <div class="src-info" id="tInfo"></div>

    <div class="section-title">POLYGON CONTROLS</div>
    <div class="ctrl-row"><label><input type="checkbox" id="chkShow" checked> <b>Show polygon</b></label></div>
    <div class="ctrl-row"><label><input type="checkbox" id="chkEdit"> <b style="color:#60a5fa">Enable editing</b></label></div>
    <div class="hint">Drag handles, click line to add, right-click to delete</div>
    <div class="ctrl-label">Buffer: <b id="bufVal"></b> mi</div>
    <input type="range" id="buf" min="0" max="6" step="0.1">
    <div class="ctrl-label">Line: <b id="wVal">3</b> px</div>
    <input type="range" id="pw" min="1" max="10" step="1" value="3">
    <select id="ps"><option value="" selected>Solid</option><option value="6, 6">Dashed</option><option value="2, 4">Dotted</option></select>
    <div class="ctrl-row"><span style="color:#94a3b8">Colour</span><input type="color" id="pc" value="#3b82f6"></div>
    <div class="ctrl-label">Keep OUTSIDE:</div>
    <select id="excl"><option value="missing" selected>Missing / suggested</option><option value="all">All non-defined sites</option><option value="none">None (convex hull)</option></select>
    <div class="btn-row"><button class="btn" id="btnRebuild">Rebuild</button><button class="btn" id="btnFit">Fit view</button></div>
    <div id="inside"></div>

    <div class="section-title">VIEW CONTROLS</div>
    <div class="ctrl-label">Sector: <b id="sVal">1.0</b> x</div>
    <input type="range" id="ss" min="0.5" max="6" step="0.1" value="1">
    <div class="ctrl-label">Font: <b id="fVal">13</b> px</div>
    <input type="range" id="fs" min="8" max="28" step="1" value="13">
    <div class="ctrl-row"><label><input type="checkbox" id="vLines" checked> HO lines</label></div>
    <div class="ctrl-row"><label><input type="checkbox" id="vMiss" checked> Missing sites</label></div>
    <div class="ctrl-row"><label><input type="checkbox" id="vOther" checked> Other sites</label></div>
    <div class="ctrl-row"><label><input type="checkbox" id="vLbl" checked> Labels</label></div>
    <div class="ctrl-row"><label><input type="checkbox" id="vBlue"> <b>All blue</b> <span style="color:#64748b">(except source)</span></label></div>

    <div class="section-title">SUMMARY</div>
    <div class="stats" id="stats"></div>
    <div id="warn"></div>

    <div class="section-title">NEIGHBOURS <span style="font-weight:normal;color:#64748b">(tick = in polygon)</span></div>
    <div class="pills" id="pills"></div>
    <div id="list"></div>
    <div class="btn-row"><button class="btn" id="btnCsv">Export missing</button><button class="btn" id="btnCsvAll">Export all</button></div>

    <div class="section-title">LEGEND</div>
    <div style="font-size:11px;color:#cbd5e1">
      <div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:12px;height:12px;background:#f97316;border-radius:2px"></span>Source site</div>
      <div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:12px;height:12px;background:#3b82f6;border-radius:2px"></span>Defined neighbour</div>
      <div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:12px;height:12px;background:#f59e0b;border-radius:2px"></span>Missing / suggested</div>
      <div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:12px;height:12px;background:#6b7280;border-radius:2px"></span>Other site</div>
      <div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:20px;border-top:2px solid #10b981"></span>HO link</div>
      <div style="display:flex;align-items:center;gap:6px;margin:3px 0"><span style="width:20px;border-top:2px dashed #f59e0b"></span>Missing link</div>
    </div>
  </div>
</div>

<div id="mapbox">
  <div id="map"></div>
  <div id="mbtn">
    <button class="mb" id="tg" title="Show / hide panel">☰</button>
    <button class="mb" id="btnSel" title="Select sites: click a site, or draw a polygon around many sites">⬚ Select</button>
    <button class="mb" id="btnRul" title="Measure distance (km / mi)">📏 Ruler</button>
  </div>
  <div id="rbox">
    <b style="color:#60a5fa;font-size:11px">Distance ruler</b>
    <div class="hint">Click points on the map (click a site to snap to it)</div>
    <div id="rTot" style="margin:6px 0 2px;font-size:12px;font-weight:bold;color:#f87171">Total: 0.00 km / 0.00 mi</div>
    <div id="rSeg" class="hint"></div>
    <div class="btn-row" style="margin-top:6px"><button class="btn" id="rUndo">Undo last point</button><button class="btn" id="rClr">Clear</button></div>
  </div>
  <div id="selbar">
    <span id="selN">0 selected</span>
    <span id="drawCtl"><button class="btn btn-primary" id="selFin">Close polygon</button><button class="btn" id="selUndo">Undo point</button><button class="btn" id="selCancel">Cancel</button></span>
    <button class="btn" id="selCopy">Copy</button>
    <button class="btn" id="selCsv">Export CSV</button>
    <button class="btn" id="selVis">Select all visible</button>
    <button class="btn" id="selClr">Clear</button>
  </div>
  <div id="toast"></div>
</div>
</div>
</div>
<div id="loading"><div class="ov-box"><div class="spin"></div><div class="ov-t">Rendering map ...</div><div class="ov-s">Please wait</div></div></div>

<div id="mdl"><div class="mbox">
  <div class="mh"><b>Selected sites</b><button id="mX" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:16px">✕</button></div>
  <div class="fm"><button data-f="excel" class="btn on">Excel (tab)</button><button data-f="usid" class="btn">USID only</button><button data-f="csv" class="btn">CSV</button></div>
  <textarea id="mTa" readonly></textarea>
  <div class="mf"><span id="mN"></span><button class="btn" id="mCopy">Copy to clipboard</button><button class="btn" id="mDl">Download CSV</button></div>
</div></div>

<script>
const D = __DATA__;
const S = D.sites, CFG = D.cfg, ALL = Object.keys(S);
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const COL = {source:'#f97316', ok:'#3b82f6', missing:'#f59e0b', other:'#6b7280'};
const TXT = {source:'#f97316', ok:'#3b82f6', missing:'#f59e0b', other:'#9ca3af'};
const STAT = {ok:['Defined','#10b981'], zero_ho:['0% HO','#f59e0b'], no_coords:['Not in file','#ef4444'], not_defined:['Suggested','#8b5cf6']};
const WHY = {no_coords:'Defined in table, not in site file', zero_ho:'Defined but 0 HO', not_defined:'Closest site, not in table'};
const MI = 69.093;

function hav(la1, lo1, la2, lo2) { const r = Math.PI/180, a = Math.sin((la2-la1)*r/2)**2 + Math.cos(la1*r)*Math.cos(la2*r)*Math.sin((lo2-lo1)*r/2)**2; return 2*3958.8*Math.asin(Math.sqrt(a)); }
function toast(msg) { const t = $('toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', 3000); }

// ── usage logging (Google Sheet web-app / any webhook; no-op when CFG.logUrl is empty) ──
const USER = CFG.user || {};
let _lastLog = {k:'', t:0};
function logEvent(event, usid, via, extra) {
  if (!CFG.logUrl) return;
  const k = event + '|' + usid, now = Date.now();
  if (_lastLog.k === k && now - _lastLog.t < 2000) return;        // ignore double fire
  _lastLog = {k, t:now};
  const d = new Date();
  const body = {token:CFG.logToken || '', event, usid:usid || '', site_name:S[usid] ? S[usid][0] : '', via:via || '',
    cluster:(D.nbr[usid] && D.nbr[usid][0]) || '', signum:USER.id || '', auth_email:USER.email || '',
    ts_utc:d.toISOString(), ts_local:d.toLocaleString('en-IN', {timeZone:'Asia/Kolkata', hour12:false}), extra:extra || ''};
  try { fetch(CFG.logUrl, {method:'POST', mode:'no-cors', headers:{'Content-Type':'text/plain;charset=utf-8'}, body:JSON.stringify(body), keepalive:true}).catch(() => {}); } catch (e) {}
}

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
  'Topo': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {maxZoom:19, attribution:'Esri, USGS, NOAA, OpenStreetMap contributors'}),
  'Street': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'}),
  'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:19, attribution:'Esri'}),
  'Light': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {maxZoom:19, attribution:'&copy; OpenStreetMap &copy; CARTO'})
};
base.Topo.addTo(map); L.control.layers(base, null, {position:'topright'}).addTo(map);
map.createPane('poly'); map.getPane('poly').style.zIndex = 450;
const mapEl = map.getContainer();
const zoomCls = () => mapEl.classList.toggle('zhi', map.getZoom() >= 12);
map.on('zoomend', zoomCls); zoomCls();

const G = {other:L.layerGroup(), missing:L.layerGroup(), ok:L.layerGroup(), source:L.layerGroup()};
const LG = {ok:L.layerGroup(), missing:L.layerGroup()};
Object.values(G).concat(Object.values(LG)).forEach(g => g.addTo(map));

function recolor() {
  const blue = $('vBlue').checked, c = role => blue && role !== 'source' ? COL.ok : COL[role];
  wedges.forEach(w => w.p.setStyle({fillColor:c(w.role)}));
  dots.forEach(d => d.m.setStyle({fillColor:c(d.role)}));
  mapEl.classList.toggle('allblue', blue);          // labels follow the same colour rule
}
function setLayer(g, on) { on ? map.addLayer(g) : map.removeLayer(g); }
function applyVis() { setLayer(G.other, $('vOther').checked); setLayer(G.missing, $('vMiss').checked); setLayer(LG.ok, $('vLines').checked); setLayer(LG.missing, $('vLines').checked && $('vMiss').checked); mapEl.classList.toggle('nolbl', !$('vLbl').checked); }
['vLines','vMiss','vOther','vLbl'].forEach(id => $(id).addEventListener('change', applyVis));
$('vBlue').addEventListener('change', recolor);

function wedge(lat, lon, az, r) {
  const la = lat*Math.PI/180, lo = lon*Math.PI/180, d = r/6371000, pts = [[lat, lon]];
  for (let i = 0; i <= 13; i++) {
    const b = (az - 32.5 + i*5)*Math.PI/180;
    const pl = Math.asin(Math.sin(la)*Math.cos(d) + Math.cos(la)*Math.sin(d)*Math.cos(b));
    const pn = lo + Math.atan2(Math.sin(b)*Math.sin(d)*Math.cos(la), Math.cos(d) - Math.sin(la)*Math.sin(pl));
    pts.push([pl*180/Math.PI, pn*180/Math.PI]);
  }
  return pts;
}

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
const poly = L.polygon([], {pane:'poly', color:'#3b82f6', weight:3, fill:false, bubblingMouseEvents:false}).addTo(map);

function tip(u, extra) { const s = S[u], n = cur && cur.NB[u]; let h = '<b>' + esc(s[0]) + '</b><br>USID: ' + esc(u); if (cur && u === cur.src) h += '<br><span style="color:#f97316">SOURCE</span>'; else if (n) { h += '<br>' + STAT[n.st][0]; if (n.pct != null) h += ' · HO ' + n.pct.toFixed(1) + '%'; if (n.d != null) h += ' · ' + n.d + ' mi'; } return h + (extra || ''); }
const popup = (u, extra) => tip(u, extra) + '<br><a href="#" onclick="loadSource(\'' + String(u).replace(/'/g, '') + '\',\'map\');return false" style="color:#60a5fa">Load as source</a>';

function loadSource(u, via) {
  u = String(u).trim().toUpperCase(); if (!S[u]) { toast('USID not found'); return; }
  if (via) logEvent('search', u, via);
  const A = analyse(u); A.NB = {}; A.nbrs.forEach(n => A.NB[n.u] = n);
  A.role = {}; A.nbrs.forEach(n => { if (S[n.u]) A.role[n.u] = n.st === 'ok' ? 'ok' : 'missing'; }); A.role[u] = 'source';
  cur = A; LAT0 = S[u][1]; LON0 = S[u][2];
  Object.values(G).concat(Object.values(LG)).forEach(g => g.clearLayers()); wedges = [];
  const src = S[u]; const draw = ALL.filter(k => A.dist[k] <= CFG.maxMiles || A.role[k]); A.draw = draw; dots = []; siteLayers = {};
  ['other', 'missing', 'ok', 'source'].forEach(role => {
    draw.filter(k => (A.role[k] || 'other') === role).forEach(k => {
      const s = S[k], g = G[role], sl = siteLayers[k] = {wedges:[], dot:null}, onClick = e => siteClick(k, e);
      s[3].forEach(c => {
        const p = L.polygon(wedge(s[1], s[2], c[1], radius), {color:'#000', weight:1, fillColor:COL[role], fillOpacity:.7});
        p.bindTooltip(tip(k, '<br>Cell: ' + esc(c[0]) + ' (' + c[1] + '°)'), {sticky:true});
        p.bindPopup(popup(k, '<br>Cell: ' + esc(c[0]) + ' (' + c[1] + '°)'));
        p.on('click', onClick); sl.wedges.push(p); g.addLayer(p); wedges.push({p, s, az:c[1], role});
      });
      if (!s[3].length) { const dot = L.circleMarker([s[1], s[2]], {radius:5, color:'#000', weight:1, fillColor:COL[role], fillOpacity:.9}).bindTooltip(tip(k)).bindPopup(popup(k)); dot.on('click', onClick); sl.dot = dot; g.addLayer(dot); dots.push({m:dot, role}); }
      g.addLayer(L.marker([s[1], s[2]], {interactive:false, icon:L.divIcon({className:'', iconSize:[150, 20], iconAnchor:[-8, 10], html:'<div class="lbl lbl-' + role + '" style="color:' + TXT[role] + '">' + esc(s[0]) + '</div>'})}));
    });
  });
  A.nbrs.forEach(n => { if (!S[n.u]) return; const miss = n.st !== 'ok', s = S[n.u]; L.polyline([[src[1], src[2]], [s[1], s[2]]], miss ? {color:'#f59e0b', weight:1.8, dashArray:'6,5', opacity:.85, interactive:false} : {color:'#10b981', weight:1.8, opacity:.75, interactive:false}).addTo(miss ? LG.missing : LG.ok); });
  applyVis(); recolor(); selected.forEach(styleSel);
  incl = {}; incl[u] = true; A.nbrs.forEach(n => { if (S[n.u]) incl[n.u] = (n.st === 'ok'); });
  $('empty').style.display = 'none'; $('panel').style.display = ''; $('q').value = u;
  $('tSrc').textContent = 'Source: ' + u + ' (' + src[0] + ')';
  $('tSub').textContent = A.cluster || (D.nbr[u] ? '' : 'No neighbour definition in SQL table for this USID');
  $('tInfo').textContent = src[1].toFixed(5) + ', ' + src[2].toFixed(5) + ' | ' + src[3].length + ' cell(s)';
  const c = st => A.nbrs.filter(n => n.st === st).length;
  $('stats').innerHTML = [
    ['Defined (table)', A.defined, '#3b82f6'],
    ['Plotted OK', c('ok'), '#10b981'],
    ['Defined, issue', c('no_coords') + c('zero_ho'), '#f59e0b'],
    ['Suggested', c('not_defined'), '#8b5cf6']
  ].map(x => '<div class="stat-box"><b style="color:' + x[2] + '">' + x[1] + '</b><span>' + x[0] + '</span></div>').join('');
  const nc = A.nbrs.filter(n => n.st === 'no_coords').map(n => n.u);
  $('warn').innerHTML = nc.length ? '<div class="warn">Defined in SQL table but NOT found in the site file (cannot be plotted): <b>' + nc.map(esc).join(', ') + '</b></div>' : '';
  filter = 'all'; renderPills(); renderList();
  if (!D.nbr[u]) toast('No neighbour definition for ' + u + ' - showing closest-site suggestions only');
  buildPoly(); fit();
}

function buildPoly() { const pts = Object.keys(incl).filter(k => incl[k] && S[k]).map(k => [S[k][1], S[k][2]]); const hull = convexHull(pts); polyPts = hull.length >= 3 ? bufferHull(hull, parseFloat($('buf').value)) : []; poly.setLatLngs(polyPts); applyExclusion(); dirty = false; makeHandles(); updateInside(); if (!polyPts.length) toast('Need at least 3 ticked sites to form a polygon'); }
function tryRebuild() { if (dirty && !confirm('Rebuilding will discard your manual polygon edits. Continue?')) return false; buildPoly(); return true; }
function updateInside() { if (!cur || !polyPts.length) { $('inside').innerHTML = ''; return; } const P = polyPts.map(p => toXY(p[0], p[1])); const ins = Object.keys(cur.role).filter(k => cur.role[k] === 'missing' && !incl[k] && pip(toXY(S[k][1], S[k][2]), P)); $('inside').innerHTML = ins.length ? '<div class="inside-warn">' + ins.length + ' missing site(s) inside polygon: ' + ins.map(k => esc(S[k][0])).join(', ') + '</div>' : '<div class="inside-ok">No missing sites inside the polygon</div>'; }
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
function renderList() { const rows = cur.nbrs.filter(n => filter === 'all' || (filter === 'ok' && n.st === 'ok') || (filter === 'miss' && n.st !== 'ok')); $('list').innerHTML = rows.length ? rows.map(n => { const has = !!S[n.u]; const meta = [n.ho != null ? (n.ho === 0 ? '<span style="color:#f59e0b">HO 0</span>' : 'HO ' + n.ho) : '', n.pct != null ? n.pct.toFixed(1) + '%' : '', n.d != null ? n.d + ' mi' : '', n.st === 'not_defined' ? 'not in SQL table' : ''].filter(Boolean).join(' · '); return '<div class="nb" data-u="' + esc(n.u) + '"><div class="nbt">' + (has ? '<input type="checkbox" class="inc" data-u="' + esc(n.u) + '"' + (incl[n.u] ? ' checked' : '') + '>' : '<span style="width:13px"></span>') + '<b>' + esc(n.u) + '</b><span class="nm">' + esc(n.n) + '</span><span class="bd" style="background:' + STAT[n.st][1] + '">' + STAT[n.st][0] + '</span></div>' + (meta ? '<div class="meta">' + meta + '</div>' : '') + '</div>'; }).join('') : '<div class="hint">Nothing to show.</div>'; }
$('pills').addEventListener('click', e => { if (e.target.dataset.f) { filter = e.target.dataset.f; renderPills(); renderList(); } });
$('list').addEventListener('change', e => { if (!e.target.classList.contains('inc')) return; const u = e.target.dataset.u; incl[u] = e.target.checked; if (!tryRebuild()) { e.target.checked = !e.target.checked; incl[u] = e.target.checked; } });
$('list').addEventListener('click', e => { if (e.target.closest('.inc')) return; const row = e.target.closest('.nb'); if (!row) return; const s = S[row.dataset.u]; if (s) map.flyTo([s[1], s[2]], Math.max(map.getZoom(), 13)); else toast(row.dataset.u + ' has no coordinates in the site file'); });

function csvDownload(rows, name) { const q = v => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; const csv = [['Source_USID','Missing_USID','Site_Name','Reason','HO_ATT','Pct_Sharing','Dist_miles','Latitude','Longitude']].concat(rows).map(r => r.map(q).join(',')).join('\n'); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'})); a.download = name; a.click(); }
const missRows = A => A.nbrs.filter(n => n.st !== 'ok').map(n => [A.src, n.u, n.n, WHY[n.st], n.ho, n.pct, n.d, n.lat, n.lon]);
$('btnCsv').addEventListener('click', () => { const rows = missRows(cur); if (!rows.length) { toast('No missing neighbours'); return; } csvDownload(rows, 'missing_neighbors_' + cur.src + '.csv'); });
$('btnCsvAll').addEventListener('click', () => { toast('Analysing all sources ...'); setTimeout(() => { let rows = []; Object.keys(D.nbr).forEach(u => { if (S[u]) rows = rows.concat(missRows(analyse(u))); }); if (!rows.length) { toast('No missing neighbours'); return; } csvDownload(rows, 'missing_neighbors_all_sources.csv'); toast('Exported ' + rows.length + ' rows'); }, 50); });

let acIdx = -1;
function acHits(v) { v = v.trim().toUpperCase(); if (v.length < 2) return []; const starts = [], inc = []; for (const k of ALL) { const nm = String(S[k][0]).toUpperCase(); if (k === v || k.startsWith(v) || nm.startsWith(v)) starts.push(k); else if (k.includes(v) || nm.includes(v)) inc.push(k); if (starts.length >= 40) break; } return starts.concat(inc).slice(0, 40); }
function renderAc() { const hits = acHits($('q').value), dd = $('ac'); acIdx = -1; if (!hits.length) { dd.style.display = 'none'; dd.innerHTML = ''; return; } dd.innerHTML = hits.map(k => '<div class="ai" data-u="' + esc(k) + '"><b>' + esc(k) + '</b><span>' + esc(S[k][0]) + '</span>' + (D.nbr[k] ? '<i style="color:#10b981;font-size:10px">✓ cluster</i>' : '') + '</div>').join(''); dd.style.display = 'block'; }
function pick(u) { $('ac').style.display = 'none'; loadSource(u, 'search'); }
$('q').addEventListener('input', renderAc);
$('q').addEventListener('keydown', e => { const items = $('ac').querySelectorAll('.ai'); if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { if (!items.length) return; e.preventDefault(); acIdx = Math.max(0, Math.min(items.length - 1, acIdx + (e.key === 'ArrowDown' ? 1 : -1))); items.forEach((el, i) => el.classList.toggle('sel', i === acIdx)); } else if (e.key === 'Enter') { e.preventDefault(); const v = $('q').value.trim().toUpperCase(); if (acIdx >= 0 && items[acIdx]) pick(items[acIdx].dataset.u); else if (S[v]) pick(v); else if (items.length) pick(items[0].dataset.u); } else if (e.key === 'Escape') $('ac').style.display = 'none'; });
$('ac').addEventListener('mousedown', e => { const it = e.target.closest('.ai'); if (it) { e.preventDefault(); pick(it.dataset.u); } });
document.addEventListener('click', e => { if (!e.target.closest('.srch') && !e.target.closest('#q')) $('ac').style.display = 'none'; });

const selected = new Set(); let selMode = false, rulerOn = false, rPts = [], fmtKind = 'excel';
const rLayer = L.layerGroup().addTo(map); const rLine = L.polyline([], {color:'#f87171', weight:3, dashArray:'8,5', interactive:false});
function setMode(m) {
  selMode = m === 'sel'; rulerOn = m === 'ruler';
  $('btnSel').classList.toggle('on', selMode); $('btnRul').classList.toggle('on', rulerOn);
  mapEl.classList.toggle('crosshair', selMode || rulerOn);
  selMode ? map.doubleClickZoom.disable() : map.doubleClickZoom.enable();
  if (!selMode) { sPts = []; drawSel(); }
  $('rbox').style.display = (rulerOn || rPts.length) ? 'block' : 'none'; updateSelBar();
  if (selMode) toast('Select: click empty map to draw a polygon (double-click / Enter to close) - or click one site to toggle it');
  if (rulerOn) toast('Ruler: click points on the map');
}
$('btnSel').addEventListener('click', () => setMode(selMode ? null : 'sel')); $('btnRul').addEventListener('click', () => setMode(rulerOn ? null : 'ruler'));

// polygon (lasso) selection
let sPts = [];
const sLayer = L.layerGroup().addTo(map);
const sShape = L.polygon([], {color:'#f59e0b', weight:2, dashArray:'6,4', fillColor:'#f59e0b', fillOpacity:.12, interactive:false});
function drawSel() {
  sLayer.clearLayers();
  if (sPts.length) {
    sShape.setLatLngs(sPts); sLayer.addLayer(sShape);
    sPts.forEach((p, i) => sLayer.addLayer(L.circleMarker(p, {radius:i === 0 ? 6 : 4, color:'#f59e0b', fillColor:'#fff', fillOpacity:1, weight:2, interactive:false})));
  }
  $('drawCtl').style.display = sPts.length ? 'inline-flex' : 'none'; updateSelBar();
}
function addSelPt(ll) {
  const p = Array.isArray(ll) ? ll : [ll.lat, ll.lng], cp = x => map.latLngToContainerPoint(x);
  if (sPts.length >= 3 && cp(p).distanceTo(cp(sPts[0])) < 9) { finishSel(); return; }             // click first point = close
  if (sPts.length && cp(p).distanceTo(cp(sPts[sPts.length - 1])) < 5) return;                      // ignore the 2nd click of a double-click
  sPts.push(p); drawSel();
}
function finishSel() {
  if (sPts.length < 3) { toast('Need at least 3 points for a polygon'); return; }
  if (!cur) { toast('Load a source first'); return; }
  const P = sPts.map(p => toXY(p[0], p[1])); let n = 0;
  cur.draw.forEach(k => {
    const r = cur.role[k] || 'other';
    if ((r === 'other' && !$('vOther').checked) || (r === 'missing' && !$('vMiss').checked)) return;
    if (pip(toXY(S[k][1], S[k][2]), P) && !selected.has(k)) { selected.add(k); styleSel(k); n++; }
  });
  sPts = []; drawSel(); toast(n + ' site(s) added - ' + selected.size + ' selected in total');
}
$('selFin').addEventListener('click', finishSel);
$('selUndo').addEventListener('click', () => { sPts.pop(); drawSel(); });
$('selCancel').addEventListener('click', () => { sPts = []; drawSel(); });

function siteClick(k, e) {
  if (selMode) { L.DomEvent.stopPropagation(e); map.closePopup(); if (sPts.length) addSelPt([S[k][1], S[k][2]]); else toggleSel(k); }
  else if (rulerOn) { L.DomEvent.stopPropagation(e); map.closePopup(); addRulerPoint([S[k][1], S[k][2]]); }
}
function styleSel(k) {
  const sl = siteLayers[k]; if (!sl) return; const on = selected.has(k);
  sl.wedges.forEach(p => p.setStyle(on ? {color:'#f59e0b', weight:3} : {color:'#000', weight:1}));
  if (sl.dot) { sl.dot.setStyle(on ? {color:'#f59e0b', weight:3} : {color:'#000', weight:1}); sl.dot.setRadius(on ? 8 : 5); }
}
function toggleSel(k) { selected.has(k) ? selected.delete(k) : selected.add(k); styleSel(k); updateSelBar(); }
function updateSelBar() { $('selN').textContent = selected.size + ' selected' + (sPts.length ? ' | drawing ' + sPts.length + ' pts' : ''); $('selbar').style.display = (selMode || selected.size) ? 'flex' : 'none'; }
function selectVisible() { if (!cur) return; const b = map.getBounds(); let n = 0; cur.draw.forEach(k => { const r = cur.role[k] || 'other'; if ((r === 'other' && !$('vOther').checked) || (r === 'missing' && !$('vMiss').checked)) return; if (b.contains([S[k][1], S[k][2]])) { selected.add(k); styleSel(k); n++; } }); updateSelBar(); toast(n + ' visible site(s) selected'); }
function clearSel() { const old = Array.from(selected); selected.clear(); old.forEach(styleSel); updateSelBar(); }
$('selVis').addEventListener('click', selectVisible); $('selClr').addEventListener('click', clearSel);

const SEL_HDR = ['USID','Site_Name','Latitude','Longitude','Role','HO_ATT','Pct_Sharing','Dist_from_source_mi','Dist_from_source_km'];
function selRows() { return Array.from(selected).sort().map(k => { const s = S[k], n = cur && cur.NB[k]; const role = cur && k === cur.src ? 'Source' : n ? STAT[n.st][0] : 'Other'; const d = cur && cur.dist[k] != null ? cur.dist[k] : null; return [k, s[0], s[1], s[2], role, n ? n.ho : '', n ? n.pct : '', d != null ? d.toFixed(2) : '', d != null ? (d*1.609344).toFixed(2) : '']; }); }
function selText(kind) { const rows = selRows(), q = v => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; if (kind === 'usid') return rows.map(r => r[0]).join('\n'); if (kind === 'csv') return [SEL_HDR].concat(rows).map(r => r.map(q).join(',')).join('\n'); return [SEL_HDR].concat(rows).map(r => r.map(v => v == null ? '' : v).join('\t')).join('\n'); }
function downloadText(text, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type:'text/csv'})); a.download = name; a.click(); }
function openModal() { if (!selected.size) { toast('No sites selected'); return; } $('mdl').style.display = 'flex'; showFmt(fmtKind); }
function showFmt(kind) { fmtKind = kind; $('mTa').value = selText(kind); $('mN').textContent = selected.size + ' site(s)'; document.querySelectorAll('.fm button').forEach(b => b.classList.toggle('on', b.dataset.f === kind)); }
function copyText(text) { const done = () => toast('Copied ' + selected.size + ' site(s) to clipboard'); const fallback = () => { const ta = $('mTa'); ta.value = text; ta.select(); document.execCommand('copy'); done(); }; if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback(); }
$('selCopy').addEventListener('click', openModal); $('selCsv').addEventListener('click', () => { if (!selected.size) { toast('No sites selected'); return; } downloadText(selText('csv'), 'selected_sites.csv'); });
$('mX').addEventListener('click', () => $('mdl').style.display = 'none'); $('mdl').addEventListener('click', e => { if (e.target === $('mdl')) $('mdl').style.display = 'none'; });
document.querySelector('.fm').addEventListener('click', e => { if (e.target.dataset.f) showFmt(e.target.dataset.f); });
$('mCopy').addEventListener('click', () => copyText(selText(fmtKind))); $('mDl').addEventListener('click', () => downloadText(selText('csv'), 'selected_sites.csv'));

const dtxt = mi => (mi*1.609344).toFixed(2) + ' km / ' + mi.toFixed(2) + ' mi';
function drawRuler() { rLayer.clearLayers(); rLine.setLatLngs(rPts); rLayer.addLayer(rLine); let cum = 0, last = 0; rPts.forEach((p, i) => { if (i > 0) { last = hav(rPts[i-1][0], rPts[i-1][1], p[0], p[1]); cum += last; } const m = L.circleMarker(p, {radius:4, color:'#f87171', fillColor:'#fff', fillOpacity:1, weight:2, interactive:false}); if (i > 0) m.bindTooltip(dtxt(cum), {permanent:true, direction:'top', offset:[0, -6], className:'rtip'}); rLayer.addLayer(m); }); $('rTot').textContent = 'Total: ' + dtxt(cum); $('rSeg').textContent = rPts.length > 1 ? 'Last segment: ' + dtxt(last) + ' | ' + rPts.length + ' points' : (rPts.length ? '1 point - click next point' : ''); $('rbox').style.display = (rulerOn || rPts.length) ? 'block' : 'none'; }
function addRulerPoint(ll) { rPts.push(ll); drawRuler(); }
map.on('click', e => { if (rulerOn) addRulerPoint([e.latlng.lat, e.latlng.lng]); else if (selMode) addSelPt(e.latlng); });
map.on('dblclick', () => { if (selMode && sPts.length >= 3) finishSel(); });
$('rUndo').addEventListener('click', () => { rPts.pop(); drawRuler(); }); $('rClr').addEventListener('click', () => { rPts = []; drawRuler(); });
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'Enter' && selMode && sPts.length >= 3) finishSel();
  else if (e.key === 'Escape') { if (sPts.length) { sPts = []; drawSel(); } else { setMode(null); $('mdl').style.display = 'none'; } }
});

(function init() {
  const hash = decodeURIComponent(location.hash.slice(1)).toUpperCase();
  const start = S[hash] ? hash : (CFG.default && S[CFG.default] ? CFG.default : null);
  if (start) loadSource(start);
  else if (ALL.length) { const b = L.latLngBounds(ALL.map(k => [S[k][1], S[k][2]])); map.fitBounds(b); }
})();

document.getElementById('loading').style.display = 'none';

// fit this iframe to the free space between the upload bar and the footer
(function fitFrame() {
  const FOOT = 28;
  function fit() {
    try {
      const fe = window.frameElement; if (!fe) return;
      const h = window.parent.innerHeight - fe.getBoundingClientRect().top - FOOT;
      if (h > 300) { fe.style.height = h + 'px'; fe.style.width = '100%'; fe.style.display = 'block'; }
      map.invalidateSize();
    } catch (e) {}
  }
  fit(); setInterval(fit, 700);
  try { window.parent.addEventListener('resize', fit); } catch (e) {}
})();
</script></body></html>"""

# ═══════════════════════════════ UI ═══════════════════════════════
PAGE_CSS = """
<style>
  #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main { background:#0a0e1a !important; overflow:hidden !important; }
  .block-container { padding:0 !important; max-width:100% !important; }
  div[data-testid="stVerticalBlock"] { gap:0 !important; }
  .element-container { margin-bottom:0 !important; }

  .app-hdr { background:linear-gradient(90deg,#0b1d46 0%,#12306b 100%); color:#fff; font-size:18pt; font-weight:600; letter-spacing:.3px;
             padding:8px 22px; height:52px; box-sizing:border-box; display:flex; align-items:center; border-bottom:2px solid #3b82f6; white-space:nowrap; }

  /* ================================================================
     COMPACT TOP BAR TEST CSS
     This intentionally applies to the real Streamlit widgets globally
     so we can verify the selectors before relying on .st-key-upl.
     ================================================================ */
  .st-key-upl { background:#0d1220 !important; border-bottom:1px solid #1e293b !important; padding:2px 8px 3px 8px !important; margin:0 !important; }

  label[data-testid="stWidgetLabel"] { min-height:0 !important; height:auto !important; margin:0 0 1px 0 !important; padding:0 !important; }
  label[data-testid="stWidgetLabel"] p { font-size:8px !important; color:#ffffff !important; margin:0 !important; font-weight:600; text-transform:uppercase; letter-spacing:.3px; line-height:9px !important; padding:0 !important; }

  [data-testid="stFileUploader"], [data-testid="stTextInput"] { margin:0 !important; padding:0 !important; }
  [data-testid="stFileUploader"] { height:24px !important; min-height:24px !important; max-height:24px !important; overflow:hidden !important; }
  [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderFile"] {
      min-height:24px !important; height:24px !important; max-height:24px !important; box-sizing:border-box !important;
      background:#1e293b !important; border:1px solid #334155 !important; border-radius:3px !important;
  }
  [data-testid="stFileUploaderDropzone"] { padding:0 5px !important; margin:0 !important; display:flex !important; align-items:center !important; }
  [data-testid="stFileUploaderDropzone"] section { min-height:22px !important; height:22px !important; padding:0 !important; margin:0 !important; }
  [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; padding:0 !important; margin:0 !important; }
  [data-testid="stFileUploaderDropzone"] button { min-height:20px !important; height:20px !important; max-height:20px !important; padding:0 6px !important; margin:0 !important; font-size:8px !important; line-height:18px !important; }
  [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color:#94a3b8 !important; font-size:8px !important; line-height:20px !important; }
  [data-testid="stFileUploader"]:has([data-testid="stFileUploaderFile"]) [data-testid="stFileUploaderDropzone"] { display:none !important; }
  [data-testid="stFileUploaderFile"] { padding:0 5px !important; margin:0 !important; display:flex !important; align-items:center !important; }
  [data-testid="stFileUploaderFile"] * { color:#e2e8f0 !important; font-size:9px !important; line-height:20px !important; }
  [data-testid="stFileUploaderFile"] small { display:none !important; }

  [data-testid="stTextInput"] { height:24px !important; min-height:24px !important; max-height:24px !important; }
  [data-testid="stTextInput"] > div { height:24px !important; min-height:24px !important; margin:0 !important; padding:0 !important; }
  [data-testid="stTextInput"] div[data-baseweb="input"] {
      min-height:24px !important; height:24px !important; max-height:24px !important; box-sizing:border-box !important;
      background:#1e293b !important; border:1px solid #334155 !important; border-radius:3px !important; margin:0 !important; padding:0 !important;
  }
  [data-testid="stTextInput"] input { color:#e2e8f0 !important; height:22px !important; min-height:22px !important; padding:0 6px !important; margin:0 !important; font-size:9px !important; line-height:20px !important; }
  [data-testid="stTextInput"] input::placeholder { color:#94a3b8 !important; opacity:1 !important; }
  [data-testid="InputInstructions"] { display:none !important; }

  div.stButton > button { height:24px !important; min-height:24px !important; max-height:24px !important; font-weight:600; border-radius:3px !important; padding:0 8px !important; font-size:9px !important; line-height:22px !important; }
  div.stButton > button[kind="primary"] { background:#2563eb; border:1px solid #3b82f6; color:#fff; }
  div.stButton > button:disabled { background:#1e293b !important; color:#64748b !important; border:1px solid #334155 !important; }
  .stat-line { font-size:8px !important; line-height:24px !important; height:24px !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  /* TEST MARKERS - deliberately obvious; remove after diagnosis */
  .st-key-upl { outline:2px solid rgba(255,255,0,.9) !important; }
  [data-testid="stFileUploader"] { outline:2px solid rgba(255,0,0,.55) !important; }
  [data-testid="stTextInput"] { outline:2px solid rgba(0,255,0,.55) !important; }
  div.stButton > button { outline:2px solid rgba(255,165,0,.65) !important; }

  .stApp iframe { height:calc(100vh - 150px); width:100%; border:0; display:block; }
  .landing { height:calc(100vh - 150px); display:flex; align-items:center; justify-content:center; color:#64748b; text-align:center; }
  .app-ftr { position:fixed; left:0; right:0; bottom:0; height:26px; background:#0d1220; border-top:1px solid #1e293b; color:#cbd5e1; font-size:11px;
             display:flex; align-items:center; justify-content:flex-start; padding:0 14px; z-index:99999; }
  .app-ftr a { color:#60a5fa; text-decoration:none; margin-left:4px; }
  .ov { position:fixed; inset:0; background:rgba(10,14,26,.86); z-index:100000; display:flex; align-items:center; justify-content:center; }
  .ov-box { text-align:center; color:#e2e8f0; }
  .ov-t { font-size:18px; font-weight:600; margin-top:16px; } .ov-s { font-size:12px; color:#94a3b8; margin-top:6px; }
  .spin { width:54px; height:54px; border:6px solid #1e293b; border-top-color:#3b82f6; border-radius:50%; margin:0 auto; animation:sp .9s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }

  /* Diagnostic banner */
  .diag-banner { background:#172033; border:1px solid #334155; color:#e2e8f0; padding:4px 8px; font:10px/14px monospace; margin:0; }
  .diag-ok { color:#22c55e; font-weight:700; }
  .diag-bad { color:#ef4444; font-weight:700; }
</style>
"""

def overlay(msg, sub=""):
    return f'<div class="ov"><div class="ov-box"><div class="spin"></div><div class="ov-t">{msg}</div><div class="ov-s">{sub}</div></div></div>'


def cols(spec):
    try:
        return st.columns(spec, gap="small", vertical_alignment="bottom")
    except TypeError:                      # older Streamlit
        return st.columns(spec, gap="small")


def show_map(html):
    """st.iframe (new Streamlit) with a fallback to components.html (older Streamlit)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=700)
    else:
        components.html(html, height=700, scrolling=False)


def sig(f):
    return (f.name, f.size) if f is not None else None


def main():
    st.set_page_config(layout="wide", page_title=APP_TITLE, page_icon="📡", initial_sidebar_state="collapsed")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="app-hdr">{APP_TITLE}</div>', unsafe_allow_html=True)

    # TEMPORARY DIAGNOSTICS - remove after testing.
    st.markdown(
        f'<div class="diag-banner">BUILD TEST: <b>COMPACT-UI-001</b> &nbsp;|&nbsp; Streamlit: <b>{st.__version__}</b></div>',
        unsafe_allow_html=True,
    )

    log_url, log_token = secret("LOG_WEBHOOK_URL"), secret("LOG_TOKEN")
    st.session_state.setdefault("u_id", "")

    upl_key_works = True
    try:
        bar = st.container(key="upl")
    except TypeError:
        upl_key_works = False
        bar = st.container()

    st.markdown(
        '<div class="diag-banner">Container key <b class="'
        + ('diag-ok">WORKING' if upl_key_works else 'diag-bad">NOT WORKING - fallback container used')
        + '</b> &nbsp;|&nbsp; CSS widget selectors are being tested with colored outlines.</div>',
        unsafe_allow_html=True,
    )

    with bar:
        c1, c2, c3, c4, c5 = cols([3.0, 3.0, 2.0, 0.9, 3.0])
        with c1:
            site_file = st.file_uploader("Site data (CSV)", type=["csv"], key="site")
        with c2:
            nbr_file = st.file_uploader("SQL table (XLSX / CSV)", type=["xlsx", "xls", "csv"], key="nbr")
        with c3:
            st.text_input("SIGNUM ID", key="u_id", placeholder="Enter SIGNUM ID", max_chars=20)
        signum = clean_signum(st.session_state.u_id)
        ready = bool(site_file and nbr_file and signum)          # nothing loads unless ALL THREE inputs are valid
        with c4:
            go = st.button("▶ Go", type="primary", disabled=not ready, use_container_width=True, key="go")
        status = c5.empty()

    cur_sig = (sig(site_file), sig(nbr_file))

    if go and not ready:
        st.warning("Please provide the Site file, the SQL table and a valid SIGNUM ID.")
    if go and ready:
        user = {"id": signum, "email": auth_email()}
        ph = st.empty()
        try:
            ph.markdown(overlay("Loading data ...", "Reading site file"), unsafe_allow_html=True)
            site_df = read_upload(site_file)
            ph.markdown(overlay("Loading data ...", "Reading SQL table"), unsafe_allow_html=True)
            nbr_df = read_upload(nbr_file)
            ph.markdown(overlay("Loading data ...", "Building the map - this can take a few seconds"), unsafe_allow_html=True)
            payload = build_payload(site_df, nbr_df, user, log_url, log_token)
            st.session_state["map_html"] = render_html(payload)
            st.session_state["loaded_sig"] = cur_sig
            st.session_state["loaded_info"] = f"{len(payload['sites']):,} sites | {len(payload['nbr']):,} clusters"
            post_log(log_url, log_token, "go", user, f"{site_file.name} | {nbr_file.name}")
        except Exception as e:
            st.session_state.pop("map_html", None)
            ph.empty()
            st.error(f"Could not load the files: {e}")
        ph.empty()

    loaded = st.session_state.get("loaded_sig") is not None and "map_html" in st.session_state
    if loaded and st.session_state["loaded_sig"] == cur_sig:
        msg, color = f"✓ Loaded - {st.session_state['loaded_info']}", "#10b981"
    elif loaded:
        msg, color = "New files selected - click Go to reload", "#f59e0b"
    elif not (site_file and nbr_file):
        msg, color = "Upload both files and enter SIGNUM ID", "#64748b"
    elif not signum:
        msg, color = ("Enter a valid SIGNUM ID" if st.session_state.u_id.strip() else "Enter SIGNUM ID"), "#f59e0b"
    else:
        msg, color = "Ready - click Go", "#60a5fa"
    status.markdown(f'<div class="stat-line" style="color:{color}">{msg}</div>', unsafe_allow_html=True)

    if "map_html" in st.session_state:
        show_map(st.session_state["map_html"])
    else:
        st.markdown(
            '<div class="landing"><div><div style="font-size:46px;margin-bottom:14px">📡</div>'
            '<div style="font-size:18px;color:#94a3b8;margin-bottom:8px">Upload both files, enter your SIGNUM ID, then press <b style="color:#60a5fa">Go</b></div>'
            '<div style="font-size:12px;line-height:1.7">1. <b>Site data</b> (CSV): USID, coordinates, azimuth, cell<br>2. <b>SQL table</b> (Excel / CSV): neighbour relations and HO data</div></div></div>',
            unsafe_allow_html=True)

    st.markdown(f'<div class="app-ftr">{SUPPORT_NAME} for any support, please connect with <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a></div>', unsafe_allow_html=True)


main()
