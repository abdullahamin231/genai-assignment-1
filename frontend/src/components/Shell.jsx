import React, { useCallback, useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import { getHealth } from "../lib/api";
import { WORKSPACES } from "../lib/format";
import { Icon } from "./ui";

export function useHealth(pollMs = 15000) {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setHealth(await getHealth());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [load, pollMs]);

  return { health, error, reload: load };
}

function StatusPill({ health, error, onToggle, open }) {
  const ready = health?.models_ready ?? 0;
  const total = health?.models_total ?? 0;
  const online = Boolean(health) && !error;

  return (
    <button
      type="button"
      onClick={onToggle}
      className={`flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[11px] transition-colors ${
        open ? "border-primary-container/50 bg-primary-container/10" : "border-outline-variant/50 bg-surface-container-low hover:border-outline"
      }`}
      title="Model status"
    >
      <span className={`h-2 w-2 rounded-full ${online ? "animate-pulse bg-tertiary" : "bg-error"}`} />
      <span className="hidden text-on-surface-variant sm:inline">API:</span>
      <span className={online ? "text-tertiary" : "text-error"}>{online ? "Ready" : "Offline"}</span>
      <span className="text-outline">|</span>
      <span className={ready === total && total > 0 ? "text-primary-container" : "text-warning"}>
        {ready}/{total} models
      </span>
      <Icon name={open ? "expand_less" : "expand_more"} className="text-[16px] text-outline" />
    </button>
  );
}

function ModelDrawer({ health, error, onClose }) {
  const ref = useRef(null);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    const onClick = (e) => ref.current && !ref.current.contains(e.target) && onClose();
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [onClose]);

  const models = health?.models ?? [];

  return (
    <div
      ref={ref}
      className="absolute right-0 top-12 z-50 w-[min(92vw,26rem)] rounded-xl border border-outline-variant/50 bg-surface-container-low p-4 shadow-panel"
    >
      <div className="flex items-start justify-between gap-3 border-b border-outline-variant/40 pb-3">
        <div>
          <h3 className="font-headline text-sm font-semibold text-on-surface">ONNX model registry</h3>
          <p className="mt-0.5 font-mono text-[10px] uppercase tracking-wide text-outline">
            {health ? `${health.models_ready} of ${health.models_total} loaded` : "connecting"}
          </p>
        </div>
        <button type="button" onClick={onClose} className="text-outline transition-colors hover:text-on-surface">
          <Icon name="close" className="text-[18px]" />
        </button>
      </div>

      {error ? <p className="mt-3 text-xs text-error">{error}</p> : null}

      <ul className="mt-3 flex flex-col gap-1.5">
        {models.length === 0 ? (
          <li className="text-xs text-on-surface-variant">Waiting for the backend...</li>
        ) : (
          models.map((m) => (
            <li
              key={m.file}
              className="flex items-center gap-2.5 rounded border border-outline-variant/30 bg-surface-container/70 px-2.5 py-1.5"
            >
              <Icon
                name={m.present && !m.error ? "check_circle" : m.error ? "error" : "download_for_offline"}
                className={`text-[16px] ${m.error ? "text-error" : m.present ? "text-tertiary" : "text-warning"}`}
              />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-mono text-[11px] text-on-surface">{m.file}</span>
                <span className="block truncate text-[10px] text-outline">{m.title}</span>
              </span>
              <span className={`chip ${m.error ? "bg-error/10 text-error" : m.present ? "bg-tertiary/10 text-tertiary" : "bg-warning/10 text-warning"}`}>
                {m.error ? "error" : m.present ? "ready" : "missing"}
              </span>
            </li>
          ))
        )}
      </ul>

      <div className="mt-3 border-t border-outline-variant/40 pt-3">
        <p className="label-caps">Search paths</p>
        <ul className="mt-1 flex flex-col gap-0.5">
          {(health?.search_paths ?? []).map((p) => (
            <li key={p} className="truncate font-mono text-[10px] text-on-surface-variant">
              {p}
            </li>
          ))}
        </ul>
        <p className="mt-2 text-[10px] leading-relaxed text-outline">
          Missing files? Run <span className="font-mono text-primary-container">python scripts/download_models.py</span> and restart the app.
        </p>
      </div>
    </div>
  );
}

export default function Shell({ children }) {
  const { health, error, reload } = useHealth();
  const [open, setOpen] = useState(false);

  const tabClass = ({ isActive }) =>
    `flex items-center gap-1.5 whitespace-nowrap rounded-lg px-3.5 py-1.5 text-xs font-medium transition-all ${
      isActive
        ? "bg-surface-container-highest text-primary-container shadow-sm"
        : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
    }`;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-outline-variant/40 bg-surface-container-lowest/85 backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 md:px-6">
          <div className="flex min-w-0 items-center gap-6">
            <NavLink to="/universal" className="flex shrink-0 items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-primary-container/30 bg-primary-container/10">
                <Icon name="auto_fix_high" className="text-[20px] text-primary-container" />
              </span>
              <span className="hidden font-headline text-lg font-semibold tracking-tight text-on-surface sm:block">
                GenAI Lab
              </span>
            </NavLink>
            <nav className="hidden items-center gap-1.5 rounded-xl border border-outline-variant/40 bg-surface-container-low p-1 md:flex">
              {WORKSPACES.map((w) => (
                <NavLink key={w.key} to={w.to} className={tabClass}>
                  {w.short}
                </NavLink>
              ))}
            </nav>
          </div>
          <div className="relative flex shrink-0 items-center gap-2">
            <StatusPill
              health={health}
              error={error}
              open={open}
              onToggle={() => setOpen((v) => !v)}
            />
            <button
              type="button"
              onClick={reload}
              className="hidden h-8 w-8 items-center justify-center rounded-full border border-outline-variant/50 bg-surface-container-high text-on-surface-variant transition-colors hover:text-primary-container sm:flex"
              title="Refresh health"
            >
              <Icon name="sync" className="text-[18px]" />
            </button>
            {open ? <ModelDrawer health={health} error={error} onClose={() => setOpen(false)} /> : null}
          </div>
        </div>
        <nav className="flex gap-1.5 overflow-x-auto border-t border-outline-variant/30 px-4 py-2 md:hidden">
          {WORKSPACES.map((w) => (
            <NavLink key={w.key} to={w.to} className={tabClass}>
              {w.short}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-4 py-6 md:px-6 md:py-8">
        {children}
      </main>

      <footer className="border-t border-outline-variant/40 bg-surface-container-lowest">
        <div className="mx-auto flex w-full max-w-7xl flex-col items-center justify-between gap-2 px-4 py-4 text-center sm:flex-row sm:text-left md:px-6">
          <span className="font-mono text-[11px] uppercase tracking-wide text-outline">
            GenAI Lab - Generative Image Restoration Suite
          </span>
          <span className="flex items-center gap-3 font-mono text-[11px] text-outline">
            <span>Oxford-IIIT Pet / FS2K</span>
            <span className="text-outline-variant">|</span>
            <span className="text-tertiary">ONNX Runtime {health?.versions?.onnxruntime ?? ""}</span>
          </span>
        </div>
      </footer>
    </div>
  );
}
