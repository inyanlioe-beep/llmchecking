import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_TEST_PROMPT = os.getenv(
    "LLM_TEST_PROMPT",
    "Reply with exactly: OK",
)


class ProviderError(Exception):
    """Raised when the configured provider cannot be reached or understood."""


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> tuple[dict[str, Any], float]:
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    started = time.perf_counter()
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return json.loads(response_body), elapsed_ms
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(details).get("error", {}).get("message", details)
        except json.JSONDecodeError:
            message = details or error.reason
        raise ProviderError(f"HTTP {error.code}: {message}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProviderError(str(error)) from error


def list_models(base_url: str, api_key: str) -> list[str]:
    response, _ = request_json("GET", f"{base_url}/models", api_key)
    models = response.get("data")
    if not isinstance(models, list):
        raise ProviderError("Respons model tidak memiliki format data yang valid.")
    ids = sorted(
        {
            item["id"]
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    )
    if not ids:
        raise ProviderError("Provider tidak mengembalikan model.")
    return ids


def test_model(
    model: str,
    base_url: str,
    api_key: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        response, latency_ms = request_json(
            "POST",
            f"{base_url}/chat/completions",
            api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("Respons tidak memiliki choices.")
        return {
            "model": model,
            "available": True,
            "latency_ms": latency_ms,
            "response": response,
        }
    except ProviderError as error:
        return {
            "model": model,
            "available": False,
            "latency_ms": None,
            "error": str(error),
        }


def run_check(config: dict[str, Any]) -> dict[str, Any]:
    base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
    api_key = str(config.get("api_key", DEFAULT_API_KEY))
    prompt = str(config.get("prompt", DEFAULT_TEST_PROMPT))
    timeout = max(1.0, min(float(config.get("timeout", 30)), 120))
    concurrency = max(1, min(int(config.get("concurrency", 4)), 10))
    max_latency = float(config.get("max_latency", 0) or 0)

    if not api_key:
        raise ProviderError("API key wajib diisi.")
    if not base_url.startswith(("http://", "https://")):
        raise ProviderError("Base URL harus diawali http:// atau https://.")
    if not prompt.strip():
        raise ProviderError("Prompt pengujian tidak boleh kosong.")

    models = list_models(base_url, api_key)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(test_model, model, base_url, api_key, prompt, timeout): model
            for model in models
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(
        key=lambda item: (
            not item["available"],
            item["latency_ms"] if item["latency_ms"] is not None else float("inf"),
            item["model"],
        )
    )
    available = [item for item in results if item["available"]]
    ready = [
        item
        for item in available
        if max_latency <= 0 or item["latency_ms"] <= max_latency
    ]
    return {
        "models": results,
        "total": len(results),
        "available": len(available),
        "ready": len(ready),
        "max_latency": max_latency,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "min_ms": min((item["latency_ms"] for item in available), default=None),
            "median_ms": (
                round(statistics.median(item["latency_ms"] for item in available), 1)
                if available
                else None
            ),
        },
    }


HTML = r"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Availability Checker</title>
  <style>
    :root { color-scheme: dark; --bg:#0e1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e; --green:#3fb950; --red:#f85149; --blue:#58a6ff; }
    * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text); font:14px system-ui, sans-serif }
    main { max-width:1100px; margin:0 auto; padding:32px 18px 56px } h1 { margin:0 0 7px; font-size:28px } p { color:var(--muted); margin:0 0 24px }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:18px }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px } label { display:grid; gap:6px; color:var(--muted); font-size:12px }
    input, textarea { width:100%; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--text); padding:10px; font:inherit } textarea { min-height:72px; resize:vertical }
    .wide { grid-column:1/-1 } button { background:var(--blue); color:#07111d; border:0; border-radius:6px; padding:10px 16px; font-weight:700; cursor:pointer } button:disabled { opacity:.55; cursor:wait }
    .actions { display:flex; align-items:center; gap:12px; margin-top:16px } #status { color:var(--muted) } .error { color:var(--red)!important }
    .stats { display:flex; flex-wrap:wrap; gap:18px; color:var(--muted) } .stats strong { color:var(--text); font-size:20px; display:block }
    .copybox { display:flex; gap:8px; margin-top:12px } .copybox input { font-family:ui-monospace,monospace } .copybox button { background:var(--line); color:var(--text) }
    table { border-collapse:collapse; table-layout:fixed; width:100%; margin-top:14px } th,td { text-align:left; border-bottom:1px solid var(--line); padding:11px 8px; vertical-align:top; overflow-wrap:anywhere } th { color:var(--muted); font-size:12px } th:nth-child(1),td:nth-child(1) { width:10% } th:nth-child(2),td:nth-child(2) { width:22%; min-width:180px } th:nth-child(3),td:nth-child(3) { width:12% } th:nth-child(5),td:nth-child(5) { width:8% } .row-copy { background:var(--line); color:var(--text); padding:6px 9px; font-size:12px; white-space:nowrap }
    .ok { color:var(--green) } .fail { color:var(--red) } code { font-family:ui-monospace,monospace; word-break:normal; overflow-wrap:anywhere }
    .toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:14px } .toolbar input[type=search] { flex:1 1 220px; padding:8px 10px } .toolbar .inline { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:12px } .toolbar .inline input { width:auto } .toolbar .spacer { flex:1 }
    .ghost { background:var(--line); color:var(--text) } .muted { color:var(--muted) } .small { font-size:12px }
    th.sortable { cursor:pointer; user-select:none } th.sortable:hover, th[aria-sort="ascending"], th[aria-sort="descending"] { color:var(--text) }
    .history-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px } .history-head h2 { margin:0; font-size:15px }
    .history-list { display:grid; gap:10px } .history-item { display:flex; align-items:center; justify-content:space-between; gap:12px; border:1px solid var(--line); border-radius:8px; padding:10px 12px }
    .history-actions { display:flex; gap:8px; flex-shrink:0 }
    @media (max-width:650px) { .grid { grid-template-columns:1fr } .wide { grid-column:auto } .history-item { flex-direction:column; align-items:flex-start } }
  </style>
</head>
<body>
<main>
  <h1>LLM Availability Checker</h1>
  <p>Uji semua model dari satu provider OpenAI-compatible dan urutkan berdasarkan latency terendah.</p>
  <section class="panel">
    <div class="grid">
      <label>Base URL provider<input id="base_url" value="https://api.openai.com/v1" placeholder="https://api.openai.com/v1"></label>
      <label>API key<input id="api_key" type="password" placeholder="sk-..."></label>
      <label class="wide">Prompt pengujian<textarea id="prompt">Reply with exactly: OK</textarea></label>
      <label>Timeout per model (detik)<input id="timeout" type="number" min="1" max="120" value="30"></label>
      <label>Concurrency<input id="concurrency" type="number" min="1" max="10" value="4"></label>
      <label>Latency maksimum (ms, 0 = semua yang tersedia)<input id="max_latency" type="number" min="0" value="0"></label>
    </div>
    <div class="actions"><button id="run" onclick="runCheck()">Cek semua model</button><span id="status"></span></div>
  </section>
  <section id="output" class="panel" hidden>
    <div class="stats">
      <div><strong id="total">-</strong>Total model</div><div><strong id="available">-</strong>Tersedia</div><div><strong id="ready">-</strong>Siap dipakai</div><div><strong id="median">-</strong>Median latency</div>
    </div>
    <div class="copybox"><input id="ready_models" readonly><button onclick="copyModels(event)">Copy model siap</button></div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Cari model...">
      <label class="inline"><input id="chat_only" type="checkbox"> Hanya model chat</label>
      <span id="filter_count" class="muted small"></span>
      <span class="spacer"></span>
      <button class="ghost" id="export_json" onclick="exportJson()">Export JSON</button>
      <button class="ghost" id="export_csv" onclick="exportCsv()">Export CSV</button>
    </div>
    <table><thead><tr><th class="sortable" data-sort="status">Status</th><th class="sortable" data-sort="model">Model</th><th class="sortable" data-sort="latency">Latency</th><th class="sortable" data-sort="note">Keterangan</th><th>Aksi</th></tr></thead><tbody id="rows"></tbody></table>
  </section>
  <section class="panel">
    <div class="history-head"><h2>Riwayat pengecekan</h2><button class="ghost" onclick="clearHistory()">Hapus semua</button></div>
    <div id="history_list" class="history-list"></div>
  </section>
</main>
<script>
const STORAGE_KEY = 'llm_checker_config_v1';
const HISTORY_KEY = 'llm_checker_history_v1';
const HISTORY_LIMIT = 8;
const INPUT_IDS = ['base_url', 'api_key', 'prompt', 'timeout', 'concurrency', 'max_latency'];

function saveConfig() {
  const config = {};
  INPUT_IDS.forEach(id => {
    config[id] = document.getElementById(id).value;
  });
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch (e) {
    // ignore quota or storage errors
  }
}

function loadConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const config = JSON.parse(raw);
    INPUT_IDS.forEach(id => {
      if (config[id] !== undefined && config[id] !== null) {
        document.getElementById(id).value = config[id];
      }
    });
  } catch (e) {
    // ignore parsing errors
  }
}

let currentData = null;
let currentQuery = '';
let chatOnly = false;
let sortKey = 'latency';
let sortDir = 'asc';

const NON_CHAT_PATTERN = /(embed|moderation|whisper|tts|speech|transcri|audio|image|dall-e|realtime|rerank|guard|davinci|babbage|similarity|text-search|code-search|clip|sora)/i;

function isChatModel(model) {
  return !NON_CHAT_PATTERN.test(model);
}

document.addEventListener('DOMContentLoaded', () => {
  loadConfig();
  INPUT_IDS.forEach(id => {
    document.getElementById(id).addEventListener('input', saveConfig);
    document.getElementById(id).addEventListener('change', saveConfig);
  });

  document.getElementById('search').addEventListener('input', (event) => {
    currentQuery = event.target.value;
    renderRows();
  });
  document.getElementById('chat_only').addEventListener('change', (event) => {
    chatOnly = event.target.checked;
    renderRows();
  });
  document.querySelectorAll('th.sortable').forEach(header => {
    header.addEventListener('click', () => {
      const key = header.dataset.sort;
      if (sortKey === key) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = key;
        sortDir = key === 'status' ? 'desc' : 'asc';
      }
      renderRows();
    });
  });
  document.getElementById('history_list').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    if (button.dataset.action === 'open') openHistoryEntry(button.dataset.id);
    if (button.dataset.action === 'delete') deleteHistoryEntry(button.dataset.id);
  });

  renderHistory();
});

function compareModels(a, b) {
  switch (sortKey) {
    case 'status':
      return Number(a.available) - Number(b.available) || compareModelsByLatency(a, b);
    case 'model':
      return a.model.localeCompare(b.model);
    case 'note':
      return noteText(a).localeCompare(noteText(b)) || compareModelsByLatency(a, b);
    default:
      return compareModelsByLatency(a, b);
  }
}

function compareModelsByLatency(a, b) {
  const left = a.latency_ms == null ? Infinity : a.latency_ms;
  const right = b.latency_ms == null ? Infinity : b.latency_ms;
  if (left === right) return a.model.localeCompare(b.model);
  return left - right;
}

function noteText(item) {
  return item.available ? 'Respons diterima' : String(item.error || '');
}

function visibleModels(data) {
  const query = currentQuery.trim().toLowerCase();
  const filtered = data.models.filter(item =>
    (!query || item.model.toLowerCase().includes(query)) &&
    (!chatOnly || isChatModel(item.model))
  );
  const direction = sortDir === 'asc' ? 1 : -1;
  return filtered
    .map((item, index) => ({item, index}))
    .sort((a, b) => (compareModels(a.item, b.item) * direction) || (a.index - b.index))
    .map(entry => entry.item);
}

function showSummary(data) {
  document.getElementById('output').hidden = false;
  document.getElementById('total').textContent = data.total;
  document.getElementById('available').textContent = data.available;
  document.getElementById('ready').textContent = data.ready;
  document.getElementById('median').textContent = data.summary.median_ms == null ? '-' : `${data.summary.median_ms} ms`;
  const ready = data.models.filter(x => x.available && (!data.max_latency || x.latency_ms <= data.max_latency)).map(x => x.model);
  document.getElementById('ready_models').value = ready.join(', ');
}

function renderRows() {
  const rows = document.getElementById('rows');
  if (!currentData) return;
  const items = visibleModels(currentData);
  document.getElementById('filter_count').textContent =
    items.length === currentData.models.length ? `${items.length} model` : `${items.length} dari ${currentData.models.length} model`;
  document.querySelectorAll('th.sortable').forEach(header => {
    const active = header.dataset.sort === sortKey;
    header.setAttribute('aria-sort', active ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none');
    header.textContent = header.textContent.replace(/ [▲▼]$/, '') + (active ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');
  });
  if (!items.length) {
    rows.innerHTML = '<tr><td colspan="5" class="muted">Tidak ada model yang cocok dengan filter.</td></tr>';
    return;
  }
  rows.innerHTML = items.map(x => `<tr><td class="${x.available?'ok':'fail'}">${x.available?'READY':'FAILED'}</td><td class="model-cell"><code>${escapeHtml(x.model)}</code></td><td>${x.latency_ms == null ? '-' : x.latency_ms+' ms'}</td><td>${escapeHtml(noteText(x))}</td><td><button class="row-copy" data-model="${escapeHtml(x.model)}" onclick="copyModel(this)">Copy</button></td></tr>`).join('');
}

function exportJson() {
  if (!currentData) return;
  const models = visibleModels(currentData);
  const payload = {
    ...currentData,
    models,
    exported_total: models.length,
    filtered: models.length !== currentData.models.length,
    exported_at: new Date().toISOString(),
  };
  downloadFile(`llm-check-${fileStamp(currentData)}.json`, JSON.stringify(payload, null, 2), 'application/json');
}

function exportCsv() {
  if (!currentData) return;
  const header = ['status', 'model', 'latency_ms', 'note'];
  const lines = [header.join(',')];
  visibleModels(currentData).forEach(item => {
    lines.push([
      item.available ? 'READY' : 'FAILED',
      item.model,
      item.latency_ms == null ? '' : item.latency_ms,
      noteText(item),
    ].map(csvCell).join(','));
  });
  downloadFile(`llm-check-${fileStamp(currentData)}.csv`, lines.join('\r\n') + '\r\n', 'text/csv');
}

function csvCell(value) {
  const text = String(value == null ? '' : value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function fileStamp(data) {
  return String(data.checked_at || new Date().toISOString()).replace(/[^0-9]/g, '').slice(0, 14);
}

function downloadFile(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], {type: `${type};charset=utf-8`}));
  const link = document.createElement('a');
  link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function saveHistory(entries) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
  } catch (e) {
    // ignore quota errors
  }
}

function saveToHistory(data) {
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    checked_at: data.checked_at,
    base_url: document.getElementById('base_url').value,
    total: data.total,
    available: data.available,
    ready: data.ready,
    max_latency: data.max_latency,
    summary: data.summary,
    models: data.models.map(x => ({
      model: x.model,
      available: x.available,
      latency_ms: x.latency_ms,
      error: x.error ? String(x.error).slice(0, 200) : undefined,
    })),
  };
  const entries = [entry, ...loadHistory()].slice(0, HISTORY_LIMIT);
  saveHistory(entries);
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('history_list');
  const entries = loadHistory();
  if (!entries.length) {
    list.innerHTML = '<p class="muted small">Belum ada riwayat pengecekan.</p>';
    return;
  }
  list.innerHTML = entries.map(entry => `
    <div class="history-item">
      <div>
        <div><strong>${escapeHtml(entry.checked_at || '-')}</strong> <span class="muted small">${escapeHtml(entry.base_url || '-')}</span></div>
        <div class="muted small">${entry.total} model · ${entry.available} tersedia · ${entry.ready} siap dipakai${entry.summary && entry.summary.median_ms != null ? ` · median ${entry.summary.median_ms} ms` : ''}</div>
      </div>
      <div class="history-actions">
        <button class="ghost" data-action="open" data-id="${escapeHtml(entry.id)}">Tampilkan</button>
        <button class="ghost" data-action="delete" data-id="${escapeHtml(entry.id)}">Hapus</button>
      </div>
    </div>`).join('');
}

function openHistoryEntry(id) {
  const entry = loadHistory().find(item => item.id === id);
  if (!entry) return;
  currentData = entry;
  document.getElementById('search').value = '';
  currentQuery = '';
  showSummary(entry);
  renderRows();
  document.getElementById('output').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function deleteHistoryEntry(id) {
  saveHistory(loadHistory().filter(item => item.id !== id));
  renderHistory();
}

function clearHistory() {
  saveHistory([]);
  renderHistory();
}

async function runCheck() {
  const button = document.getElementById('run'), status = document.getElementById('status');
  button.disabled = true; status.className = ''; status.textContent = 'Mengambil model dan melakukan pengujian...';
  document.getElementById('output').hidden = true;
  const payload = Object.fromEntries(['base_url','api_key','prompt','timeout','concurrency','max_latency'].map(id => [id, document.getElementById(id).value]));
  try {
    const response = await fetch('/api/check', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Gagal menjalankan pengecekan.');
    render(data); status.textContent = `Selesai pada ${data.checked_at}`;
    saveToHistory(data);
  } catch (error) { status.className = 'error'; status.textContent = error.message; }
  finally { button.disabled = false; }
}
function render(data) {
  currentData = data;
  showSummary(data);
  renderRows();
}
async function copyText(value) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const helper = document.createElement('textarea');
  helper.value = value; helper.style.position = 'fixed'; helper.style.opacity = '0';
  document.body.appendChild(helper); helper.focus(); helper.select();
  if (!document.execCommand('copy')) throw new Error('Clipboard tidak tersedia.');
  helper.remove();
}
async function copyModel(button) {
  const original = button.textContent;
  try { await copyText(button.dataset.model); button.textContent = 'Copied'; }
  catch (error) { button.textContent = 'Gagal'; }
  setTimeout(() => { button.textContent = original; }, 1200);
}
async function copyModels(event) {
  const button = event.currentTarget;
  try { await copyText(document.getElementById('ready_models').value); button.textContent = 'Copied'; }
  catch (error) { button.textContent = 'Gagal'; }
  setTimeout(() => { button.textContent = 'Copy model siap'; }, 1200);
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/check":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            config = json.loads(self.rfile.read(length))
            result = run_check(config)
            self.send_json(HTTPStatus.OK, result)
        except (ProviderError, ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    print(f"LLM Availability Checker berjalan di http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
