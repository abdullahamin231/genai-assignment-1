import React, { useState } from "react";
import CorruptionControls from "../components/CorruptionControls";
import ImageSource from "../components/ImageSource";
import RecordedResults from "../components/RecordedResults";
import WorkspaceHeader from "../components/WorkspaceHeader";
import { Alert, Bar, Chip, GhostButton, Icon, ImageView, MetricStrip, Panel, PrimaryButton } from "../components/ui";
import useWorkspace from "../hooks/useWorkspace";
import { CLASS_ICONS, CLASS_LABELS, CLASS_NAMES, describeCorruption, downloadDataUrl, num, pct } from "../lib/format";

const ORACLE_OPTIONS = ["clean", "salt", "blur", "occlusion"];

export default function HardRouting() {
  const ws = useWorkspace("/api/hard-routing");
  const [sampleName, setSampleName] = useState("");
  const [mode, setMode] = useState("predicted");
  const [oracle, setOracle] = useState("salt");
  const { result, loading, error, setError, controls, setControls } = ws;

  const probs = result?.probs ?? [];
  const predicted = result?.predicted;
  const selected = result?.selected;
  const activeName = selected?.name;
  const scored = result?.metrics?.mode === "reference" ? result.metrics.output : result?.metrics?.output;

  const run = () => ws.run({ routing_mode: mode, oracle_label: mode === "oracle" ? oracle : "" });

  const metricItems = [
    {
      icon: "tune",
      label: "Restoration Quality",
      value: `PSNR ${num(scored?.psnr, 2, " dB")}`,
      accent: "text-primary-container",
      foot: result ? `${CLASS_LABELS[selected?.name] ?? "-"} - SSIM ${num(scored?.ssim, 3)}` : "run the pipeline",
    },
    {
      icon: "verified",
      label: "Classifier Confidence",
      value: pct(result?.confidence, 1),
      tone: "tertiary",
      foot: predicted ? `argmax p = ${CLASS_LABELS[predicted.name]}` : "stage 1 not run",
    },
    {
      icon: "speed",
      label: "Pipeline Latency",
      value: num(result?.timing?.inference_ms, 2, " ms"),
      tone: "secondary",
      foot: result
        ? `clf ${num(result.timing.classifier_ms, 1)} + expert ${num(result.timing.specialist_ms, 1)} ms`
        : "classifier + specialist",
    },
  ];

  return (
    <>
      <WorkspaceHeader
        icon="alt_route"
        eyebrow="Task 2 - Corruption Classification + Specialist Autoencoders"
        title="Hard-Routed Restoration"
        description="A convolutional classifier predicts one of four conditions and dispatches the image to exactly one specialist autoencoder. Clean inputs take the identity bypass."
        model="task2_classifier.onnx + 3 specialists"
        extra={
          <span className="flex flex-wrap items-center gap-2">
            <Chip tone="secondary">4-way softmax</Chip>
            <Chip tone="tertiary">argmax routing</Chip>
          </span>
        }
      />

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="flex flex-col gap-5 lg:col-span-5">
          <Panel step={1} title="Input Image" subtitle="Upload a (possibly corrupted) image">
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

          <Panel step={2} title="Degradation + Routing" subtitle="Ground truth is only used in oracle mode">
            <CorruptionControls state={controls} onChange={setControls} />

            <div className="border-t border-outline-variant/30 pt-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="label-caps">Routing mode</span>
                <Chip tone={mode === "oracle" ? "warning" : "primary"}>{mode === "oracle" ? "ground truth" : "classifier"}</Chip>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setMode("predicted")}
                  className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                    mode === "predicted"
                      ? "border-primary-container/70 bg-primary-container/15 text-primary-container"
                      : "border-outline-variant/40 bg-surface-container text-on-surface-variant hover:text-on-surface"
                  }`}
                >
                  Predicted routing
                </button>
                <button
                  type="button"
                  onClick={() => setMode("oracle")}
                  className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                    mode === "oracle"
                      ? "border-warning/70 bg-warning/15 text-warning"
                      : "border-outline-variant/40 bg-surface-container text-on-surface-variant hover:text-on-surface"
                  }`}
                >
                  Oracle routing
                </button>
              </div>

              {mode === "oracle" ? (
                <label className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-outline-variant/40 bg-surface-container-lowest/70 px-3 py-2">
                  <span className="text-xs text-on-surface-variant">Known corruption label</span>
                  <select
                    value={oracle}
                    onChange={(e) => setOracle(e.target.value)}
                    className="rounded border border-outline-variant/50 bg-surface-container px-2 py-1 font-mono text-[11px] text-on-surface focus:border-primary-container focus:outline-none"
                  >
                    {ORACLE_OPTIONS.map((o) => (
                      <option key={o} value={o}>
                        {CLASS_LABELS[o]}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </div>
          </Panel>

          <PrimaryButton loading={loading} onClick={run}>
            {loading ? "Classifying + Restoring..." : "Classify & Restore"}
          </PrimaryButton>

          {error ? (
            <Alert kind="error" onDismiss={() => setError(null)}>
              {error}
            </Alert>
          ) : null}
        </div>

        <div className="flex flex-col gap-5 lg:col-span-7">
          <Panel
            step={3}
            title="Stage 1 - Classifier Decision"
            subtitle="C(x̃) = [p_clean, p_salt, p_blur, p_occlusion]"
            right={result ? <Chip tone="tertiary">{pct(result.confidence, 1)} confidence</Chip> : <Chip tone="neutral">idle</Chip>}
          >
            <div className="flex flex-col gap-3 rounded-xl border border-primary-container/40 bg-surface-container-high p-4 shadow-panel sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-primary-container text-on-primary">
                  <Icon name={CLASS_ICONS[predicted?.name] ?? "help"} className="text-[26px]" filled />
                </span>
                <span className="flex flex-col">
                  <span className="flex items-center gap-2 font-headline text-base font-semibold text-on-surface">
                    {predicted ? `${CLASS_LABELS[predicted.name]} predicted` : "Awaiting inference"}
                    {predicted ? <span className="h-2 w-2 animate-ping rounded-full bg-primary-container" /> : null}
                  </span>
                  <span className="mt-0.5 text-xs text-on-surface-variant">
                    {selected
                      ? result.identity_bypass
                        ? "Clean path - identity bypass, no expert invoked"
                        : `Dispatched to ${result.expert}`
                      : "Run the pipeline to classify the input"}
                  </span>
                </span>
              </div>
              <div className="rounded-lg bg-surface-container-lowest/70 px-3 py-2 sm:text-right">
                <span className="label-caps block">Selected expert</span>
                <span className="font-headline text-lg font-bold text-primary-container">
                  {selected ? CLASS_LABELS[selected.name] : "--"}
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-3">
              {CLASS_NAMES.map((name, i) => (
                <Bar
                  key={name}
                  label={CLASS_LABELS[name]}
                  icon={CLASS_ICONS[name]}
                  value={probs[i]?.prob ?? 0}
                  hint={probs.length ? pct(probs[i].prob, 1) : "0%"}
                  color={
                    selected?.name === name
                      ? "primary-container"
                      : predicted?.name === name
                        ? "tertiary"
                        : "outline-variant"
                  }
                  muted={Boolean(selected) && selected.name !== name}
                />
              ))}
            </div>

            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {CLASS_NAMES.map((name, i) => {
                const isActive = activeName === name;
                const isPredicted = predicted?.name === name;
                return (
                  <div
                    key={name}
                    className={`rounded border px-2 py-2 text-center transition-all ${
                      isActive
                        ? "border-primary/40 bg-primary-container/10"
                        : "border-outline-variant/30 bg-surface-container/50 opacity-60"
                    }`}
                  >
                    <span className={`block font-mono text-[10px] uppercase tracking-wide ${isActive ? "text-primary" : "text-outline"}`}>
                      {name === "clean" ? "Clean bypass" : CLASS_LABELS[name]}
                    </span>
                    <span className={`block font-mono text-[11px] font-bold ${isActive ? "text-primary" : "text-outline"}`}>
                      {isActive ? "ACTIVE" : "standby"}
                      {isPredicted && !isActive ? " (argmax)" : ""}
                    </span>
                    <span className="block font-mono text-[10px] text-outline">{pct(probs[i]?.prob, 0)}</span>
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel
            step={4}
            title="Restoration Comparison"
            subtitle={result ? `Routed through ${result.expert ?? "-"}` : "Specialist output appears here"}
            right={<Chip tone="neutral">{result ? result.routing_mode : "predicted"}</Chip>}
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ImageView
                src={result?.corrupted}
                label={result ? `Degraded - ${describeCorruption(result.corruption)}` : "Degraded Input"}
                tone="error"
                badge={<Chip tone="error">x̃</Chip>}
                emptyText="Upload an image and run the pipeline"
              />
              <ImageView
                src={result?.output}
                label={result ? `Restored by ${CLASS_LABELS[selected?.name] ?? "-"}` : "Restored Output"}
                tone="primary"
                badge={<Chip tone="primary">{result?.identity_bypass ? "identity bypass" : "specialist AE"}</Chip>}
                overlay={
                  loading ? (
                    <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-lg bg-surface-container-lowest/80">
                      <Icon name="progress_activity" className="animate-spin text-3xl text-primary-container" />
                      <span className="font-mono text-[11px] uppercase tracking-wide text-on-surface">routing</span>
                    </span>
                  ) : null
                }
                emptyText="Specialist reconstruction appears here"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-outline-variant/30 pt-4">
              <span className="font-mono text-[11px] text-on-surface-variant">
                inference <span className="text-primary-container">{num(result?.timing?.inference_ms, 2, " ms")}</span>
                {result ? (
                  <span className="text-outline">
                    {" "}
                    - classifier {num(result.timing.classifier_ms, 1)} ms / expert {num(result.timing.specialist_ms, 1)} ms
                  </span>
                ) : null}
              </span>
              <GhostButton
                disabled={!result}
                onClick={() => downloadDataUrl(result.output, "hard_routed_restored.png")}
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
      <RecordedResults task="task2" />
    </>
  );
}
