import React, { useState } from "react";
import CorruptionControls from "../components/CorruptionControls";
import ImageSource from "../components/ImageSource";
import RecordedResults from "../components/RecordedResults";
import WorkspaceHeader from "../components/WorkspaceHeader";
import { Alert, Bar, Chip, GhostButton, Icon, ImageView, KeyValue, MetricStrip, Panel, PrimaryButton } from "../components/ui";
import useWorkspace from "../hooks/useWorkspace";
import { CLASS_ICONS, CLASS_LABELS, CLASS_NAMES, describeCorruption, downloadDataUrl, num, pct } from "../lib/format";

const BAR_COLOR = {
  clean: "outline",
  salt: "primary-container",
  blur: "secondary",
  occlusion: "tertiary",
};

export default function SoftMoE() {
  const ws = useWorkspace("/api/soft-mixture");
  const [sampleName, setSampleName] = useState("");
  const [tau, setTau] = useState(1.0);
  const { result, loading, error, setError, controls, setControls } = ws;

  const weights = result?.weights ?? [];
  const dominant = result?.dominant;
  const routing = result?.routing ?? {};
  const scored = result?.metrics?.mode === "reference" ? result.metrics.output : result?.metrics?.output;
  const ranked = [...weights].sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0));
  const topTwo = ranked.slice(0, 2).filter((w) => (w.weight ?? 0) > 0.01);

  const run = () => ws.run({ tau });

  const metricItems = [
    {
      icon: "tune",
      label: "Restoration Quality",
      value: `PSNR ${num(scored?.psnr, 2, " dB")}`,
      accent: "text-primary-container",
      foot: result ? `SSIM ${num(scored?.ssim, 3)} - soft blend of 4 branches` : "run the model",
    },
    {
      icon: "hub",
      label: "Dominant Branch",
      value: dominant ? CLASS_LABELS[dominant.name] : "--",
      tone: "tertiary",
      foot: dominant ? `${pct(dominant.weight ?? dominant.prob, 1)} of the routing mass` : "gate weights appear here",
    },
    {
      icon: "speed",
      label: "Processing Time",
      value: num(result?.timing?.inference_ms, 2, " ms"),
      tone: "secondary",
      foot: result ? `total ${num(result.timing.total_ms, 1, " ms")} - single ONNX graph` : "gate + 4 experts",
    },
  ];

  return (
    <>
      <WorkspaceHeader
        icon="filter_alt"
        eyebrow="Task 3 - Jointly Trained Soft Mixture-of-Experts"
        title="Soft Mixture-of-Experts Restoration"
        description="A gating network assigns a continuous weight to the identity branch and the three specialists, so compound or ambiguous degradations are blended instead of hard-routed."
        model="task3_soft_moe.onnx"
        extra={
          <span className="flex flex-wrap items-center gap-2">
            <Chip tone="primary">softmax(G(x̃)/τ)</Chip>
            <Chip tone="secondary">joint fine-tune</Chip>
          </span>
        }
      />

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        {/* ---- input + degradation ---- */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <Panel step={1} title="Input Image" subtitle="Upload or pick a preset sample">
            <ImageSource
              source={ws.source}
              reference={ws.reference}
              sampleName={sampleName}
              onSample={(s) => {
                setSampleName(s.label);
                ws.pickSample(s);
              }}
            />
          </Panel>

          <Panel step={2} title="Apply Corruptions" subtitle="Combine several to stress the router">
            <CorruptionControls state={controls} onChange={setControls} multi />
            <PrimaryButton loading={loading} onClick={run}>
              {loading ? "Blending experts..." : "Run Soft MoE"}
            </PrimaryButton>
            {error ? (
              <Alert kind="error" onDismiss={() => setError(null)}>
                {error}
              </Alert>
            ) : null}
          </Panel>
        </div>

        {/* ---- routing weights ---- */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <Panel
            step={3}
            title="Router Weight Allocation"
            subtitle="w = softmax(G(x̃) / τ)"
            right={result ? <Chip tone="tertiary">Σ = 100%</Chip> : <Chip tone="neutral">idle</Chip>}
          >
            <div className="flex flex-col gap-4">
              {CLASS_NAMES.map((name, i) => (
                <Bar
                  key={name}
                  label={name === "clean" ? "Clean identity branch" : CLASS_LABELS[name]}
                  icon={CLASS_ICONS[name]}
                  value={weights[i]?.weight ?? 0}
                  hint={weights.length ? pct(weights[i].weight, 1) : "0%"}
                  color={BAR_COLOR[name]}
                  muted={Boolean(dominant) && dominant.name !== name && (weights[i]?.weight ?? 0) < 0.1}
                />
              ))}
            </div>

            <div className="border-t border-outline-variant/30 pt-4">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="flex items-center gap-1.5 text-on-surface-variant">
                  <Icon name="thermostat" className="text-[16px] text-primary-container" />
                  Blending softness (temperature)
                </span>
                <span className="font-mono text-[11px] font-bold text-primary-container">τ = {tau.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="2.5"
                step="0.05"
                value={tau}
                onChange={(e) => setTau(Number(e.target.value))}
                className="mt-3 w-full accent-primary-container"
                aria-label="Gating temperature"
              />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-outline">
                <span>0.1 sharp / pick one</span>
                <span>2.5 equal blend</span>
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-outline">
                Applied to the exported gate logits (or fed to the model when it exposes a τ input).
              </p>
            </div>

            <div className="flex flex-col gap-1.5 rounded-lg border border-outline-variant/40 bg-surface-container p-3">
              <KeyValue label="weight tensor" value={routing.weight_output ?? "--"} tone="text-primary" />
              <KeyValue label="image tensor" value={routing.image_output ?? "--"} />
              <KeyValue label="application" value={routing.applied ?? "--"} tone="text-tertiary" />
              <KeyValue label="raw gate" value={(routing.raw ?? []).map((v) => Number(v).toFixed(2)).join(" ") || "--"} />
            </div>
          </Panel>

          <div className="flex items-start gap-2.5 rounded-xl border border-outline-variant/40 bg-surface-container-low p-3.5">
            <Icon name="lightbulb" className="mt-0.5 shrink-0 text-[18px] text-primary-container" />
            <p className="text-[11px] leading-relaxed text-on-surface-variant">
              {result ? (
                <>
                  The gate sent <strong className="text-on-surface">{pct(topTwo[0]?.weight, 0)}</strong> to{" "}
                  <strong className="text-on-surface">{topTwo[0] ? CLASS_LABELS[topTwo[0].name] : "-"}</strong>
                  {topTwo[1] ? (
                    <>
                      {" "}
                      and <strong className="text-on-surface">{pct(topTwo[1].weight, 0)}</strong> to{" "}
                      <strong className="text-on-surface">{CLASS_LABELS[topTwo[1].name]}</strong>
                    </>
                  ) : null}{" "}
                  simultaneously - no single branch had to gamble on the answer.
                </>
              ) : (
                <>
                  Unlike hard routing, which commits to exactly one expert, the soft gate shares the reconstruction across every
                  branch. Run a compound corruption to watch the weights split.
                </>
              )}
            </p>
          </div>
        </div>

        {/* ---- result ---- */}
        <div className="flex flex-col gap-5 lg:col-span-4">
          <Panel
            step={4}
            title="Result Comparison"
            subtitle="Degraded input vs soft-blended reconstruction"
            right={<Chip tone="primary">{result ? describeCorruption(result.corruption) : "awaiting run"}</Chip>}
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ImageView
                src={result?.corrupted}
                label="Before - Degraded"
                tone="error"
                badge={<Chip tone="error">x̃</Chip>}
                emptyText="Run the model to see the input"
              />
              <ImageView
                src={result?.output}
                label="After - Soft MoE"
                tone="primary"
                badge={<Chip tone="primary">{dominant ? `${pct(dominant.weight ?? dominant.prob, 0)} ${CLASS_LABELS[dominant.name]}` : "blended"}</Chip>}
                overlay={
                  loading ? (
                    <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-lg bg-surface-container-lowest/80">
                      <Icon name="progress_activity" className="animate-spin text-3xl text-primary-container" />
                      <span className="font-mono text-[11px] uppercase tracking-wide text-on-surface">gating</span>
                    </span>
                  ) : null
                }
                emptyText="Blended reconstruction appears here"
              />
            </div>

            {weights.length ? (
              <div className="flex h-6 w-full overflow-hidden rounded-full bg-surface-container-highest">
                {CLASS_NAMES.map((name, i) => (
                  <span
                    key={name}
                    title={`${CLASS_LABELS[name]} ${pct(weights[i]?.weight, 1)}`}
                    style={{ width: `${(weights[i]?.weight ?? 0) * 100}%` }}
                    className={`h-full transition-all duration-500 ${
                      name === "clean"
                        ? "bg-outline"
                        : name === "salt"
                          ? "bg-primary-container"
                          : name === "blur"
                            ? "bg-secondary"
                            : "bg-tertiary"
                    }`}
                  />
                ))}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-outline-variant/30 pt-4">
              <span className="font-mono text-[11px] text-on-surface-variant">
                inference <span className="text-primary-container">{num(result?.timing?.inference_ms, 2, " ms")}</span>
                <span className="text-outline"> - single joint graph</span>
              </span>
              <GhostButton
                disabled={!result}
                onClick={() => downloadDataUrl(result.output, "soft_moe_restored.png")}
                className={result ? "border-tertiary/40 text-tertiary" : ""}
              >
                <Icon name="download" className="text-[15px]" />
                Save restored image
              </GhostButton>
            </div>
          </Panel>
        </div>
      </div>

      <MetricStrip items={metricItems} />
      <RecordedResults task="task3" />
    </>
  );
}
