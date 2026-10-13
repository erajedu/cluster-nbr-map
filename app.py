/**
 * Usage-log receiver for the Cluster Neighbor Visualization Tool.
 *
 * Bound to a Google Sheet (Extensions > Apps Script). Deploy as a Web app:
 *   Execute as: Me      |      Who has access: Anyone
 *
 * Security: put a secret in  Project Settings > Script properties  ->  LOG_TOKEN = <random string>
 * and use the same value as LOG_TOKEN in Streamlit secrets. Requests with a wrong token are ignored.
 */
const SHEET_NAME = 'UsageLog';
const TZ = 'Asia/Kolkata';
const HEADERS = ['Date', 'Time (IST)', 'SIGNUM', 'Event', 'USID', 'Site name', 'Cluster', 'Via', 'Auth e-mail', 'Details', 'UTC timestamp'];

function doPost(e) {
  const lock = LockService.getScriptLock();
  try {
    const d = JSON.parse(e.postData.contents);
    const expected = PropertiesService.getScriptProperties().getProperty('LOG_TOKEN');
    if (expected && d.token !== expected) return out_({ ok: false, error: 'bad token' });

    lock.waitLock(10000);
    const sh = sheet_();
    const now = new Date();
    sh.appendRow([
      Utilities.formatDate(now, TZ, 'dd-MMM-yyyy'),
      Utilities.formatDate(now, TZ, 'HH:mm:ss'),
      String(d.signum || '').toLowerCase(),
      d.event || '',
      d.usid || '',
      d.site_name || '',
      d.cluster || '',
      d.via || '',
      d.auth_email || '',
      d.extra || '',
      d.ts_utc || ''
    ]);
    return out_({ ok: true });
  } catch (err) {
    return out_({ ok: false, error: String(err) });
  } finally {
    try { lock.releaseLock(); } catch (x) {}
  }
}

function doGet() { return out_({ ok: true, service: 'usage-log' }); }   // quick health check in a browser

function sheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.insertSheet(SHEET_NAME);
    sh.appendRow(HEADERS);
    sh.setFrozenRows(1);
    sh.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold').setBackground('#0e2655').setFontColor('#ffffff');
  }
  return sh;
}

function out_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}
