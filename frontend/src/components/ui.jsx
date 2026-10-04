import React, { useRef, useState } from "react";

export function Icon({ name, className = "text-[18px]", filled = false }) {
  return (
    <span className={`material-symbols-outlined leading-none ${filled ? "filled" : ""} ${className}`} aria-hidden="true">
      {name}
    </span>
  );
}

export function Panel({ step, title, subtitle, right, children, className = "", flush = false }) {
  return (
    <section className={`card flex flex-col ${className}`}>
      <header className="flex items-start justify-between gap-3 border-b border-outline-variant/30 px-5 py-4">
        <div className="flex items-start gap-2.5">
          {step ? (
            <span className="mt-0.5 w-5 h-5 shrink-0 rounded-full bg-primary-container/15 text-primary-container font-mono text-[11px] font-bold flex items-center justify-center">
              {step}
            </span>
          ) : null}
          <div>
            <h2 className="font-headline text-[15px] font-semibold leading-5 text-on-surface">{title}</h2>
            {subtitle ? <p className="mt-0.5 text-xs text-on-surface-variant">{subtitle}</p> : null}
          </div>
        </div>
        {right}
      </header>
      <div className={flush ? "flex flex-col" : "flex flex-col gap-4 p-5"}>{children}</div>
    </section>
  );
}

export function Chip({ children, tone = "neutral", className = "" }) {
  const tones = {
    neutral: "bg-surface-container text-on-surface-variant border border-outline-variant/40",
    primary: "bg-primary-container/10 text-primary-container border border-primary-container/25",
    secondary: "bg-secondary/10 text-secondary border border-secondary/25",
    tertiary: "bg-tertiary/10 text-tertiary border border-tertiary/25",
    error: "bg-error/10 text-error border border-error/25",
    warning: "bg-warning/10 text-warning border border-warning/25",
  };
  return <span className={`chip ${tones[tone]} ${className}`}>{children}</span>;
}

export function PrimaryButton({ children, loading = false, disabled = false, onClick, className = "" }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className={`group flex w-full items-center justify-center gap-2 rounded-xl bg-primary-container px-6 py-3.5 font-headline text-sm font-semibold tracking-wide text-on-primary shadow-glow-sm transition-all hover:brightness-110 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none ${className}`}
    >
      {loading ? <Icon name="progress_activity" className="animate-spin text-[20px]" /> : null}
      {children}
    </button>
  );
}

export function GhostButton({ children, onClick, active = false, disabled = false, className = "" }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-center gap-2 rounded-lg border px-3 py-2 font-mono text-[11px] uppercase tracking-wide transition-all disabled:cursor-not-allowed disabled:opacity-50 ${
        active
          ? "border-primary-container/60 bg-primary-container/15 text-primary-container"
          : "border-outline-variant/40 bg-surface-container text-on-surface-variant hover:border-outline hover:text-on-surface"
      } ${className}`}
    >
      {children}
    </button>
  );
}

export function Alert({ kind = "error", children, onDismiss }) {
  const tones = {
    error: "border-error/30 bg-error/10 text-error",
    warning: "border-warning/30 bg-warning/10 text-warning",
    info: "border-outline-variant/50 bg-surface-container text-on-surface-variant",
    success: "border-tertiary/30 bg-tertiary/10 text-tertiary",
  };
  const icons = { error: "error", warning: "warning", info: "info", success: "check_circle" };
  return (
    <div className={`flex items-start gap-2.5 rounded-lg border px-3.5 py-2.5 text-xs ${tones[kind]}`}>
      <Icon name={icons[kind]} className="mt-0.5 shrink-0 text-[16px]" />
      <div className="flex-1 leading-relaxed">{children}</div>
      {onDismiss ? (
        <button type="button" onClick={onDismiss} className="opacity-70 transition-opacity hover:opacity-100">
          <Icon name="close" className="text-[16px]" />
        </button>
      ) : null}
    </div>
  );
}

export function Dropzone({ file, preview, onFile, label = "Upload an image", hint = "PNG / JPG / WebP", className = "" }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const pick = (event) => {
    const chosen = event.target.files?.[0];
    if (chosen) onFile(chosen);
    event.target.value = "";
  };

  const drop = (event) => {
    event.preventDefault();
    setDragging(false);
    const chosen = event.dataTransfer.files?.[0];
    if (chosen) onFile(chosen);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={drop}
      className={`relative flex cursor-pointer items-center gap-3 rounded-xl border border-dashed px-3.5 py-3 text-left transition-colors ${
        dragging ? "border-primary-container bg-primary-container/10" : "border-outline-variant/70 bg-surface-container-lowest/60 hover:border-primary-container/60 hover:bg-surface-container"
      } ${className}`}
    >
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={pick} />
      {preview ? (
        <img src={preview} alt="" className="h-11 w-11 shrink-0 rounded object-cover ring-1 ring-outline-variant/50" />
      ) : (
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded bg-surface-container-high text-primary-container">
          <Icon name="upload_file" />
        </span>
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs font-medium text-on-surface">{file ? file.name : label}</span>
        <span className="mt-0.5 block font-mono text-[10px] uppercase tracking-wide text-outline">
          {file ? "click or drop to replace" : hint}
        </span>
      </span>
      <Icon name="attach_file" className="shrink-0 text-[16px] text-outline" />
    </div>
  );
}

export function ImageView({ src, label, tone = "neutral", badge, sub, overlay, aspect = "aspect-square", emptyText = "Run the model to see the result" }) {
  const toneText = {
    neutral: "text-on-surface",
    error: "text-error",
    primary: "text-primary-container",
    tertiary: "text-tertiary",
    secondary: "text-secondary",
  }[tone];
  const toneDot = {
    neutral: "bg-outline",
    error: "bg-error",
    primary: "bg-primary-container",
    tertiary: "bg-tertiary",
    secondary: "bg-secondary",
  }[tone];

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 px-0.5 text-xs">
        <span className={`flex min-w-0 items-center gap-1.5 font-medium ${toneText}`}>
          <span className={`h-2 w-2 shrink-0 rounded-full ${toneDot}`} />
          <span className="truncate">{label}</span>
        </span>
        {badge ? <span className="shrink-0">{badge}</span> : null}
      </div>
      <div className={`grid-cell ${aspect} ${src ? "" : "placeholder-grid"}`}>
        {src ? (
          <img src={src} alt={label} className="h-full w-full object-contain" />
        ) : (
          <span className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 px-3 text-center text-[11px] text-outline">
            <Icon name="image_search" className="text-2xl opacity-60" />
            {emptyText}
          </span>
        )}
        {overlay}
      </div>
      {sub ? <span className="px-0.5 font-mono text-[10px] uppercase tracking-wide text-outline">{sub}</span> : null}
    </div>
  );
}

export function Bar({ label, value, color = "primary-container", hint, icon, muted = false }) {
  const width = `${Math.max(0, Math.min(1, Number(value) || 0)) * 100}%`;
  const barColor = {
    "primary-container": "bg-primary-container shadow-[0_0_10px_rgba(0,229,255,0.45)]",
    secondary: "bg-secondary",
    tertiary: "bg-tertiary shadow-[0_0_10px_rgba(168,255,210,0.35)]",
    outline: "bg-outline",
    "outline-variant": "bg-outline-variant",
    error: "bg-error",
  }[color];
  const textColor = {
    "primary-container": "text-primary-container",
    secondary: "text-secondary",
    tertiary: "text-tertiary",
    outline: "text-outline",
    "outline-variant": "text-outline",
    error: "text-error",
  }[color];

  return (
    <div className={`flex flex-col gap-1.5 ${muted ? "opacity-70" : ""}`}>
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="flex min-w-0 items-center gap-2 font-medium text-on-surface">
          {icon ? <Icon name={icon} className={`text-[15px] ${textColor}`} /> : null}
          <span className="truncate">{label}</span>
        </span>
        <span className={`font-mono text-[11px] font-bold ${textColor}`}>{hint}</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-surface-container-highest p-px">
        <div className={`h-full rounded-full transition-all duration-500 ${barColor}`} style={{ width }} />
      </div>
    </div>
  );
}

export function MetricStrip({ items }) {
  return (
    <div className="card grid grid-cols-1 divide-y divide-outline-variant/40 md:grid-cols-3 md:divide-x md:divide-y-0">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-4 px-5 py-4">
          <span
            className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border ${
              item.tone === "tertiary"
                ? "border-tertiary/20 bg-tertiary/10 text-tertiary"
                : item.tone === "secondary"
                  ? "border-secondary/20 bg-secondary/10 text-secondary"
                  : item.tone === "error"
                    ? "border-error/20 bg-error/10 text-error"
                    : "border-primary-container/20 bg-primary-container/10 text-primary-container"
            }`}
          >
            <Icon name={item.icon} className="text-[22px]" />
          </span>
          <span className="min-w-0">
            <span className="block text-xs font-medium text-on-surface-variant">{item.label}</span>
            <span className="metric-value block truncate">
              {item.accent ? <span className={item.accent}>{item.value}</span> : item.value}
            </span>
            {item.foot ? <span className="block font-mono text-[10px] uppercase tracking-wide text-outline">{item.foot}</span> : null}
          </span>
        </div>
      ))}
    </div>
  );
}

export function LoadingOverlay({ show, text = "Running inference..." }) {
  if (!show) return null;
  return (
    <span className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 rounded-lg bg-surface-container-lowest/85 backdrop-blur-[2px]">
      <Icon name="progress_activity" className="animate-spin text-3xl text-primary-container" />
      <span className="font-mono text-[11px] uppercase tracking-wide text-on-surface">{text}</span>
    </span>
  );
}

export function KeyValue({ label, value, tone = "" }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded border border-outline-variant/30 bg-surface-container/60 px-2.5 py-1.5">
      <span className="font-mono text-[10px] uppercase tracking-wide text-outline">{label}</span>
      <span className={`truncate font-mono text-[11px] font-semibold ${tone || "text-on-surface"}`}>{value}</span>
    </div>
  );
}
