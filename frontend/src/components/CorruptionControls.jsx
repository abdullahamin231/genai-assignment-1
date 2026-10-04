import React from "react";
import { fixedSeverity, severityFromSlider } from "../lib/format";
import { Chip, Icon } from "./ui";

const OPTIONS = [
  { id: "clean", label: "Clean Pass", icon: "check_circle", hint: "no degradation" },
  { id: "salt", label: "Salt & Pepper", icon: "grain", hint: "impulse noise" },
  { id: "blur", label: "Gaussian Blur", icon: "blur_on", hint: "lost detail" },
  { id: "occlusion", label: "Occlusion", icon: "crop_square", hint: "masked blocks" },
];

const AXIS = {
  salt: "Noise density (p)",
  blur: "Blur dispersion (σ, k)",
  occlusion: "Mask area ratio",
};

/** Form fields the backend expects for the current control state. */
export function corruptionFields(state) {
  const { types, level, severity } = state;
  const fields = { corruption: types.length ? types.join(",") : "none" };
  if (level === null || level === undefined) fields.severity = severity;
  else fields.level = level;
  return fields;
}

export const CLEAN_STATE = { types: [], level: null, severity: 0.5 };

export default function CorruptionControls({ state, onChange, multi = false }) {
  const { types, level, severity } = state;
  const active = types[0] || "salt";
  const enabled = types.length > 0;
  const has = (id) => types.includes(id);

  const patch = (next) => onChange({ ...state, ...next });

  const toggle = (id) => {
    if (id === "clean") {
      patch({ types: [] });
      return;
    }
    if (multi) {
      patch({ types: has(id) ? types.filter((t) => t !== id) : [...types, id] });
    } else {
      patch({ types: has(id) ? [] : [id] });
    }
  };

  const sliderValue = level === null || level === undefined ? severity : [0, 0.5, 1][level];
  const readout =
    !enabled
      ? "no degradation"
      : level === null || level === undefined
        ? severityFromSlider(severity, active)
        : fixedSeverity(level, active);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="label-caps">
            {multi ? "Corruption type - combine for compound damage" : "Corruption type"}
          </span>
          {multi && types.length > 1 ? <Chip tone="secondary">compound</Chip> : null}
        </div>
        <div className="grid grid-cols-2 gap-2">
          {OPTIONS.map((opt) => {
            const isActive = opt.id === "clean" ? types.length === 0 : has(opt.id);
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => toggle(opt.id)}
                aria-pressed={isActive}
                className={`flex flex-col items-start gap-0.5 rounded-xl border px-3 py-2.5 text-left transition-all ${
                  isActive
                    ? "border-primary-container/70 bg-primary-container/15 text-on-surface shadow-sm"
                    : "border-outline-variant/40 bg-surface-container text-on-surface-variant hover:border-outline hover:bg-surface-container-highest hover:text-on-surface"
                }`}
              >
                <span className="flex w-full items-center gap-2">
                  <Icon name={opt.icon} className={`text-[16px] ${isActive ? "text-primary-container" : ""}`} />
                  <span className="text-xs font-medium">{opt.label}</span>
                  {multi ? (
                    <Icon
                      name={isActive ? "check_box" : "check_box_outline_blank"}
                      className={`ml-auto text-[15px] ${isActive ? "text-primary-container" : "text-outline"}`}
                    />
                  ) : null}
                </span>
                <span className="font-mono text-[9px] uppercase tracking-wide text-outline">{opt.hint}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className={`rounded-xl border border-outline-variant/40 bg-surface-container-lowest/70 p-3.5 transition-opacity ${enabled ? "" : "pointer-events-none opacity-40"}`}>
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="text-on-surface-variant">{AXIS[active] ?? "Severity"}</span>
          <span className="font-mono text-[11px] font-bold text-primary">{readout}</span>
        </div>

        <div className="mt-2.5 grid grid-cols-3 gap-1.5">
          {["Low", "Medium", "High"].map((name, idx) => (
            <button
              key={name}
              type="button"
              onClick={() => patch({ level: idx, severity: [0, 0.5, 1][idx] })}
              className={`rounded border px-2 py-1 font-mono text-[10px] uppercase tracking-wide transition-all ${
                level === idx
                  ? "border-primary-container/60 bg-primary-container/15 text-primary-container"
                  : "border-outline-variant/40 bg-surface-container text-outline hover:text-on-surface-variant"
              }`}
            >
              {name}
            </button>
          ))}
        </div>

        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={sliderValue}
          onChange={(e) => patch({ level: null, severity: Number(e.target.value) })}
          className="mt-3 w-full accent-primary-container"
          aria-label="Corruption severity"
        />
        <div className="mt-1 flex items-center justify-between font-mono text-[10px] text-outline">
          <span>mild</span>
          <span className={level === null || level === undefined ? "text-secondary" : "text-primary-container"}>
            {level === null || level === undefined ? "custom sampling" : `fixed test severity ${["low", "medium", "high"][level]}`}
          </span>
          <span>severe</span>
        </div>
      </div>

      <p className="text-[11px] leading-relaxed text-outline">
        Corruptions are applied <span className="text-on-surface-variant">at runtime</span> with the same definitions used to train
        the models - nothing is pre-baked into the uploaded file.
      </p>
    </div>
  );
}
