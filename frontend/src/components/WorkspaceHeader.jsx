import React from "react";
import { Chip, Icon } from "./ui";

export default function WorkspaceHeader({ icon, eyebrow, title, description, model, extra }) {
  return (
    <div className="flex flex-col gap-3 border-b border-outline-variant/40 pb-4 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl">
        <div className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-widest text-primary-container">
          <Icon name={icon} className="text-[14px]" />
          {eyebrow}
        </div>
        <h1 className="mt-1.5 font-headline text-2xl font-semibold tracking-tight text-on-surface md:text-[28px]">
          {title}
        </h1>
        <p className="mt-1.5 text-sm leading-relaxed text-on-surface-variant">{description}</p>
      </div>
      <div className="flex shrink-0 flex-col items-start gap-2 md:items-end">
        <span className="flex items-center gap-2 rounded-lg border border-outline-variant/40 bg-surface-container-low px-3 py-1.5 font-mono text-[11px] text-on-surface-variant">
          <Icon name="memory" className="text-[14px] text-primary-container" />
          <span className="text-primary-container">{model}</span>
          <span className="text-outline-variant">|</span>
          <span>128×128 RGB</span>
        </span>
        {extra}
      </div>
    </div>
  );
}

export function ChipRow({ children }) {
  return <div className="flex flex-wrap items-center gap-2">{children}</div>;
}

export { Chip };
