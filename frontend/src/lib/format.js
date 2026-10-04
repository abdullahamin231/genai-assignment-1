export const CLASS_NAMES = ["clean", "salt", "blur", "occlusion"];
export const CLASS_LABELS = {
  clean: "Clean",
  salt: "Salt & Pepper",
  blur: "Gaussian Blur",
  occlusion: "Occlusion",
};
export const CLASS_ICONS = {
  clean: "check_circle",
  salt: "grain",
  blur: "blur_on",
  occlusion: "crop_square",
};
export const CLASS_COLORS = {
  clean: "outline",
  salt: "primary-container",
  blur: "secondary",
  occlusion: "tertiary",
};

export const STYLES = [
  { index: 0, name: "Style 1", title: "Style 1: Pencil Sketch", blurb: "Delicate graphite lines and soft hatching" },
  { index: 1, name: "Style 2", title: "Style 2: Charcoal & Ink", blurb: "High contrast strokes and deep shadows" },
  { index: 2, name: "Style 3", title: "Style 3: Anime / Manga", blurb: "Clean graphic contours and bold outlines" },
];

export const WORKSPACES = [
  { key: "task1", to: "/universal", short: "1. Autoencoder", label: "Universal Restoration" },
  { key: "task2", to: "/hard-routing", short: "2. Hard Routing", label: "Hard-Routed Restoration" },
  { key: "task3", to: "/soft-moe", short: "3. Soft MoE", label: "Soft Mixture-of-Experts" },
  { key: "task4", to: "/face-to-sketch", short: "4. Face to Sketch", label: "Face-to-Sketch Generator" },
];

export function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function num(value, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

export function downloadDataUrl(dataUrl, filename) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/** Human-readable description of the corruption the backend applied. */
export function describeCorruption(corruption) {
  if (!corruption) return "No corruption";
  const parts = (corruption.chain || []).map((part) => {
    if (part.type === "salt") return `S&P p=${part.p}`;
    if (part.type === "blur") return `blur k=${part.k} σ=${part.sigma}`;
    if (part.type === "occlusion") return `occlusion ${(part.coverage * 100).toFixed(0)}%`;
    if (part.type === "style") return part.label;
    return "clean";
  });
  return parts.length ? parts.join(" + ") : "No corruption";
}

export function severityFromSlider(value, type) {
  if (type === "salt") return `p = ${(0.02 + value * 0.13).toFixed(3)}`;
  if (type === "blur") {
    const k = [3, 5, 7][Math.round(value * 2)];
    const sigma = (0.5 + value * 2).toFixed(1);
    return `k = ${k}, σ = ${sigma}`;
  }
  if (type === "occlusion") return `${(10 + value * 25).toFixed(0)}% area`;
  return "no degradation";
}

export function fixedSeverity(level, type) {
  if (type === "salt") return ["p = 0.03", "p = 0.08", "p = 0.15"][level];
  if (type === "blur") return ["k=3, σ=0.7", "k=5, σ=1.5", "k=7, σ=2.5"][level];
  if (type === "occlusion") return ["10% / 1 rect", "20% / 2 rects", "35% / 3 rects"][level];
  return "";
}
