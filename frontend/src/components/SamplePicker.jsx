import React, { useEffect, useState } from "react";
import { getSamples } from "../lib/api";
import { Icon } from "./ui";

/**
 * Preset images bundled by ``scripts/prepare_samples.py``.
 * Hidden entirely when the folder is empty so the UI stays honest.
 */
export default function SamplePicker({ onPick, activeName = "" }) {
  const [samples, setSamples] = useState([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    getSamples()
      .then((data) => alive && setSamples(data.samples || []))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, []);

  if (failed || samples.length === 0) return null;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="label-caps">Preset samples</span>
        <span className="font-mono text-[10px] text-outline">Oxford-IIIT Pet</span>
      </div>
      <div className="grid grid-cols-4 gap-2 sm:grid-cols-6">
        {samples.slice(0, 6).map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => onPick(s)}
            title={s.label}
            className={`group flex flex-col items-center gap-1 rounded-lg border p-1 transition-all ${
              activeName === s.label
                ? "border-primary-container bg-surface-container-highest"
                : "border-outline-variant/40 bg-surface-container opacity-75 hover:border-outline hover:opacity-100"
            }`}
          >
            <span className="aspect-square w-full overflow-hidden rounded bg-surface-container-lowest">
              <img src={s.url} alt={s.label} loading="lazy" className="h-full w-full object-cover transition-transform group-hover:scale-105" />
            </span>
            <span className="w-full truncate text-center text-[10px] text-on-surface">{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
