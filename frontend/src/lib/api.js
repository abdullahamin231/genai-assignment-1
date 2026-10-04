/**
 * Thin client for the FastAPI backend.
 * Every request goes through the same origin (`/api` is proxied by Vite in dev
 * and by nginx inside the container).
 */
const BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch (err) {
    throw new Error(`Cannot reach the backend at ${path}. Is the API container running?`);
  }
  const text = await res.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text.slice(0, 300) };
  }
  if (!res.ok) {
    const message =
      typeof data.detail === "string" ? data.detail : res.status === 422
        ? "The request was rejected - check the uploaded image and the selected options."
        : `Request failed with status ${res.status}`;
    const error = new Error(message);
    error.status = res.status;
    error.payload = data;
    throw error;
  }
  return data;
}

export async function getHealth() {
  return request("/api/health");
}

export async function getMetrics(task) {
  return request(`/api/metrics/${task}`);
}

export async function getSamples() {
  return request("/api/samples");
}

/**
 * Run one of the model workspaces.
 * @param {string} path e.g. "/api/universal-restoration"
 * @param {{file: Blob, fields?: Record<string, unknown>, reference?: Blob}} opts
 */
export async function runTask(path, { file, fields = {}, reference = null }) {
  if (!file) throw new Error("Select or upload an input image first.");
  const fd = new FormData();
  fd.append("file", file, file.name || "input.png");
  if (reference) fd.append("reference", reference, reference.name || "reference.png");
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null || value === "") continue;
    fd.append(key, String(value));
  }
  return request(path, { method: "POST", body: fd });
}

/** Fetch a bundled sample (or any URL) as a File the API accepts. */
export async function urlToFile(url, name) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url}`);
  const blob = await res.blob();
  return new File([blob], name || url.split("/").pop() || "image.png", {
    type: blob.type || "image/png",
  });
}
