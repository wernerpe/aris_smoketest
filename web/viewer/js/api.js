// Everything that talks to the server.  No DOM, no three.js.

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not json */ }
    throw new Error(`${r.status} ${msg}`);
  }
  return r.json();
}

export const api = {
  config:      ()        => j("/api/config"),
  listJobs:    ()        => j("/api/jobs"),
  getJob:      (id)      => j(`/api/jobs/${id}`),
  createJob:   (params)  => j("/api/jobs", {
                              method: "POST",
                              headers: {"Content-Type": "application/json"},
                              body: JSON.stringify(params)}),
  cancelJob:   (id)      => j(`/api/jobs/${id}/cancel`, {method: "POST"}),
  events:      (id, off) => j(`/api/jobs/${id}/events?offset=${off || 0}`),
  bundle:      (id)      => j(`/api/jobs/${id}/bundle`),
  scene:       (rig, tool) => j(`/api/scene?rig=${rig}&tool=${tool}`),
  // The Drake meshcat view of a programme: one scene, one port, replaced
  // rather than pooled.  See `gui/server.py` MESHCAT_PORT.
  openMeshcat: (body)    => j("/api/meshcat", {
                              method: "POST",
                              headers: {"Content-Type": "application/json"},
                              body: JSON.stringify(body)}),
};

// RUN ON ARM.  Six calls, and not one of them carries a command: the argument
// lists live in `aris_sixarm/gui/operator.py`, which reads `scripts/day1.py`'s
// OPERATOR block, so the browser cannot invent an address or a flag.  `run` is
// the only one that can move an arm and it carries the typed confirmation,
// which the server checks again against its own record of the stack check.
const post = (url, body) => j(url, {
  method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify(body || {})});

export const operator = {
  config:   ()              => j("/api/operator"),
  site:     (patch)         => post("/api/operator/site", patch),
  identify: (body)          => post("/api/operator/identify", body || {}),
  pose:     (body)          => post("/api/operator/pose", body || {}),
  check:    (arm)           => post("/api/operator/check", {arm}),
  copy:     (slot, csv)     => post("/api/operator/copy", {slot, csv}),
  run:      (slot, csv, confirm) =>
                               post("/api/operator/run", {slot, csv, confirm}),
  hold:     (arm)           => post("/api/operator/hold", {arm}),
  kill:     (arm)           => post("/api/operator/kill", {arm}),
  tail:     (arm, action)   => post("/api/operator/tail", {arm, action}),
  tailRead: (arm, offset)   => j(`/api/operator/tail?arm=${arm}`
                                 + `&offset=${offset || 0}`),
};

export async function fetchBin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}

// A BufferRef {offset,length,dtype,shape} over an ArrayBuffer -> a typed array.
// Zero copy: the exporter aligns every block to 8 bytes precisely so this can
// be a view and not a slice.
export function view(buf, ref) {
  if (!ref) return null;
  const n = ref.length / ({f4: 4, f8: 8, i4: 4}[ref.dtype]);
  if (ref.dtype === "f4") return new Float32Array(buf, ref.offset, n);
  if (ref.dtype === "f8") return new Float64Array(buf, ref.offset, n);
  if (ref.dtype === "i4") return new Int32Array(buf, ref.offset, n);
  throw new Error(`unknown dtype ${ref.dtype}`);
}

// --------------------------------------------------------------------------
// The live event stream.
//
// REPLAY THEN TAIL, ALWAYS IN THAT ORDER.  The socket is opened on a byte
// offset of zero, so a browser that connects to a job an hour into its run
// receives the whole history first and then the tail.  That is what makes
// "reload the page" a safe thing to do while the fleet is being planned for,
// and what makes a finished job's viewer identical to a running one's.
export function openStream(jobId, onBatch, onClose) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/jobs/${jobId}/stream`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.kind === "batch") onBatch(m.events);
    else if (m.kind === "closed") onClose && onClose(m.payload);
  };
  ws.onerror = () => {};
  ws.onclose = () => onClose && onClose(null);
  return ws;
}
