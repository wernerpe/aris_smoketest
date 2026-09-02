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
