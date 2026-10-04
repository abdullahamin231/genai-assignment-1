import React, { useState } from "react";
import CorruptionControls from "../components/CorruptionControls";
import ImageSource from "../components/ImageSource";
import RecordedResults from "../components/RecordedResults";
import WorkspaceHeader from "../components/WorkspaceHeader";
import { Alert, Chip, GhostButton, Icon, ImageView, MetricStrip, Panel, PrimaryButton } from "../components/ui";
import useWorkspace from "../hooks/useWorkspace";
import { describeCorruption, downloadDataUrl, num } from "../lib/format";

export default function Universal() {
  const ws = useWorkspace("/api/universal-restoration");
  const [sampleName, setSampleName] = useState("");
  const { result, loading, error, setError } = ws;
  const metrics = result?.metrics;
  const scored = metrics?.mode === "reference" ? metrics.output : metrics?.output;
  const isReference = metrics?.mode === "reference";

  const metricItems = [
    {
      icon: "tune",
      label: "Reconstruction Quality",
      value: `PSNR ${num(scored?.psnr, 2, " dB")}`,
      accent: "text-primary-container",
      foot: isReference ? `against clean reference - Δ ${num(metrics.delta_psnr, 2, " dB")}` : "corrupted input vs restored output",
    },
    {
      icon: "high_quality",
      label: "Structural Similarity",
      value: num(scored?.ssim, 4),
      tone: "tertiary",
      foot: isReference ? `Δ SSIM ${num(metrics.delta_ssim, 4)}` : "corrupted input vs restored output",
    },
    {
      icon: "speed",
      label: "Processing Time",
      value: num(result?.timing?.inference_ms, 2, " ms"),
      tone: "secondary",
      foot: result ? `total ${num(result.timing.total_ms, 1, " ms")} - onnxruntime` : "run the model to measure",
    },
  ];

  return (
    <>
      <WorkspaceHeader
        icon="auto_fix_high"
        eyebrow="Task 1 - Universal Multi-Corruption Denoising Autoencoder"
        title="Universal Restoration"
        description="A single convolutional autoencoder restores clean images from salt-and-pepper noise, Gaussian blur or rectangular occlusion - without being told which corruption was applied."
        model="task1_universal_ae.onnx"
        extra={
          <span className="flex flex-wrap items-center gap-2">
            <Chip tone="primary">bottleneck latent</Chip>
            <Chip tone="tertiary">L1 + SSIM loss</Chip>
          </span>
        }
      />

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="flex flex-col gap-5 lg:col-span-5">
          <Panel step={1} title="Input Image" subtitle="Pick a clean sample or upload your own">
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

          <Panel step={2} title="Runtime Corruption" subtitle="Applied server-side before inference">
            <CorruptionControls state={ws.controls} onChange={ws.setControls} />
          </Panel>

          <PrimaryButton loading={loading} onClick={() => ws.run()}>
            {loading ? "Restoring..." : "Run Restoration"}
          </PrimaryButton>

          {error ? (
            <Alert kind="error" onDismiss={() => setError(null)}>
              {error}
            </Alert>
          ) : null}

          {result ? (
            <div className="flex flex-wrap items-center gap-2">
              <Chip tone="neutral">
                <Icon name="psychology" className="text-[13px]" />
                {result.model?.file}
              </Chip>
              <Chip tone="secondary">{describeCorruption(result.corruption)}</Chip>
            </div>
          ) : null}
        </div>

        <div className="flex flex-col gap-5 lg:col-span-7">
          <Panel
            step={3}
            title="Visual Comparison"
            subtitle="Target, degraded input, reconstruction and residual"
            right={
              <span className="hidden items-center gap-2 font-mono text-[10px] uppercase tracking-wide text-outline sm:flex">
                <span className="h-2 w-2 rounded-full bg-error" /> input
                <span className="text-outline-variant">vs</span>
                <span className="h-2 w-2 rounded-full bg-primary-container" /> restored
              </span>
            }
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ImageView
                src={result?.input}
                label="Uploaded Image"
                tone="neutral"
                badge={<Chip tone="neutral">ground source</Chip>}
                emptyText="Run the model to see the input"
              />
              <ImageView
                src={result?.corrupted}
                label="Corrupted Input"
                tone="error"
                badge={<Chip tone="error">{result ? describeCorruption(result.corruption) : "awaiting run"}</Chip>}
                emptyText="The degraded tensor fed to the encoder"
              />
              <ImageView
                src={result?.output}
                label="Restored Output"
                tone="primary"
                badge={<Chip tone="primary">x̂ = D(E(x̃))</Chip>}
                overlay={
                  loading ? (
                    <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-lg bg-surface-container-lowest/80">
                      <Icon name="progress_activity" className="animate-spin text-3xl text-primary-container" />
                      <span className="font-mono text-[11px] uppercase tracking-wide text-on-surface">reconstructing</span>
                    </span>
                  ) : null
                }
                emptyText="Run the model to see the reconstruction"
              />
              <ImageView
                src={result?.error_map}
                label="Absolute Error Map"
                tone="tertiary"
                badge={<Chip tone="tertiary">|Δ| × 4</Chip>}
                sub={result ? (result.error_reference === "reference" ? "vs clean reference" : "vs corrupted input") : ""}
                emptyText="Residual heatmap appears after inference"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-outline-variant/30 pt-4">
              <span className="font-mono text-[11px] text-on-surface-variant">
                inference <span className="text-primary-container">{num(result?.timing?.inference_ms, 2, " ms")}</span>
                {result ? <span className="text-outline"> - {num(result.size?.[0], 0)}×{num(result.size?.[1], 0)} RGB tensor</span> : null}
              </span>
              <GhostButton
                disabled={!result}
                onClick={() => downloadDataUrl(result.output, "universal_restored.png")}
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
      <RecordedResults task="task1" />
    </>
  );
}
