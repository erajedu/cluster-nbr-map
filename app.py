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
