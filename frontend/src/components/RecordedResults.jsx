import React, { useEffect, useState } from "react";
import { getMetrics } from "../lib/api";
import { num } from "../lib/format";
import { Icon } from "./ui";

function Table({ head, rows }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-outline-variant/40">
            {head.map((h) => (
              <th key={h} className="label-caps whitespace-nowrap px-2 py-2 text-left font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row[0] ?? i} className="border-b border-outline-variant/20 last:border-0">
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={`px-2 py-2 ${j === 0 ? "font-medium text-on-surface" : "font-mono text-on-surface-variant"}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Delta({ value, digits = 1, suffix = " dB" }) {
  const n = Number(value);
  if (!Number.isFinite(n)) return <span className="text-outline">--</span>;
  const good = n > 0;
  return <span className={good ? "text-tertiary" : "text-error"}>{num(n, digits, suffix)}</span>;
}

export default function RecordedResults({ task, className = "" }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let alive = true;
    setStatus("loading");
    getMetrics(task)
      .then((res) => {
        if (!alive) return;
        setData(res.metrics);
        setStatus("ready");
      })
      .catch((err) => alive && setStatus(err.status === 404 ? "missing" : "error"));
    return () => {
      alive = false;
    };
  }, [task]);

  if (status === "loading") {
    return (
      <div className={`card px-5 py-4 ${className}`}>
        <div className="h-4 w-56 animate-pulse rounded bg-surface-container-highest" />
      </div>
    );
  }

  if (status === "missing" || status === "error") {
    return (
      <div className={`card flex items-start gap-3 px-5 py-4 ${className}`}>
        <Icon name="hourglass_top" className="mt-0.5 text-[18px] text-warning" />
        <div>
          <p className="text-xs font-medium text-on-surface">No recorded test results yet</p>
          <p className="mt-0.5 text-[11px] leading-relaxed text-on-surface-variant">
            {task === "task3"
              ? "The soft mixture-of-experts study is still training. Run task3_evaluate.py to publish its metrics."
              : "Evaluate the task (taskN_evaluate.py) to record test metrics for this workspace."}
          </p>
        </div>
      </div>
    );
  }

  let body = null;
  let title = "Recorded test results";
  let source = "";

  if (task === "task1") {
    const rows = Object.entries(data.by_type || {}).map(([name, m]) => [
      name,
      num(m.in_psnr, 2, " dB"),
      num(m.out_psnr, 2, " dB"),
      // clean inputs already sit at the PSNR ceiling, so the delta is meaningless
      <Delta value={m.in_psnr >= 99 ? NaN : m.out_psnr - m.in_psnr} />,
      num(m.out_ssim, 3),
    ]);
    body = <Table head={["Condition", "Input PSNR", "Restored PSNR", "Δ PSNR", "Restored SSIM"]} rows={rows} />;
    source = `epoch ${data.epoch} - overall SSIM ${num(data.overall?.out_ssim, 3)}`;
  } else if (task === "task2") {
    const c = data.classifier || {};
    const rows = Object.entries(data.by_type || {}).map(([name, m]) => [
      name,
      num(m.in_ssim, 3),
      num(m.or_ssim, 3),
      num(m.pr_ssim, 3),
      <Delta value={(m.pr_ssim ?? 0) - (m.in_ssim ?? 0)} digits={3} suffix="" />,
    ]);
    body = (
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {[
            ["Accuracy", num((c.acc ?? 0) * 100, 2, "%")],
            ["Macro F1", num(c.macro_f1, 4)],
            ["Macro Precision", num(c.macro_precision, 4)],
            ["Macro Recall", num(c.macro_recall, 4)],
          ].map(([label, value]) => (
            <div key={label} className="rounded border border-outline-variant/30 bg-surface-container/60 px-2.5 py-2">
              <div className="font-mono text-[10px] uppercase tracking-wide text-outline">{label}</div>
              <div className="font-headline text-base font-bold text-tertiary">{value}</div>
            </div>
          ))}
        </div>
        <Table head={["Condition", "Input SSIM", "Oracle SSIM", "Predicted SSIM", "Δ SSIM"]} rows={rows} />
      </div>
    );
    source = `normalized 4×4 confusion matrix recorded on the official test split`;
  } else if (task === "task3") {
    // task3_evaluate.py is still in progress - render whatever shape it lands on.
    const byType = data.by_type ?? data.by_condition ?? data.conditions ?? null;
    const rows = byType
      ? Object.entries(byType).map(([name, m]) => [
          name,
          num(m.in_psnr ?? m.input_psnr, 2, " dB"),
          num(m.out_psnr ?? m.restored_psnr ?? m.psnr, 2, " dB"),
          <Delta value={(m.in_psnr ?? m.input_psnr) >= 99 ? NaN : (m.out_psnr ?? m.restored_psnr ?? m.psnr) - (m.in_psnr ?? m.input_psnr)} />,
          num(m.out_ssim ?? m.restored_ssim ?? m.ssim, 3),
        ])
      : [];
    const ov = data.overall ?? data;
    body = (
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {[
            ["Input PSNR", num(ov.in_psnr ?? ov.input_psnr, 2, " dB")],
            ["Restored PSNR", num(ov.out_psnr ?? ov.restored_psnr ?? ov.psnr, 2, " dB")],
            ["Restored SSIM", num(ov.out_ssim ?? ov.restored_ssim ?? ov.ssim, 3)],
            ["Gate τ", num(ov.tau ?? data.tau, 2)],
          ].map(([label, value]) => (
            <div key={label} className="rounded border border-outline-variant/30 bg-surface-container/60 px-2.5 py-2">
              <div className="font-mono text-[10px] uppercase tracking-wide text-outline">{label}</div>
              <div className="font-headline text-base font-bold text-primary-container">{value}</div>
            </div>
          ))}
        </div>
        {rows.length ? (
          <Table head={["Condition", "Input PSNR", "Soft-MoE PSNR", "Δ PSNR", "Restored SSIM"]} rows={rows} />
        ) : (
          <p className="text-[11px] leading-relaxed text-on-surface-variant">
            Summary metrics recorded - per-condition breakdown not present in this file.
          </p>
        )}
      </div>
    );
    source = `soft mixture-of-experts - epoch ${data.epoch ?? "--"}`;
  } else if (task === "task4") {
    const rows = ["style_0", "style_1", "style_2"]
      .filter((k) => data[k])
      .map((k, i) => [
        `Style ${i + 1} (${data[k].n} samples)`,
        num(data[k].psnr?.mean, 2, " dB"),
        num(data[k].ssim?.mean, 3),
        num(data[k].l1?.mean, 4),
      ]);
    body = <Table head={["Sketch style", "PSNR", "SSIM", "L1"]} rows={rows} />;
    source = `${data.split} split - n=${data.n}, epoch ${data.epoch}`;
  }

  return (
    <section className={`card px-5 py-4 ${className}`}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon name="query_stats" className="text-[17px] text-primary-container" />
          <h3 className="font-headline text-sm font-semibold text-on-surface">{title}</h3>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-wide text-outline">{source}</span>
      </div>
      {body}
    </section>
  );
}
