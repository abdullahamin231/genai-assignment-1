import React, { useState } from "react";
import RecordedResults from "../components/RecordedResults";
import WebcamCapture from "../components/WebcamCapture";
import WorkspaceHeader from "../components/WorkspaceHeader";
import { Alert, Chip, Dropzone, GhostButton, Icon, ImageView, MetricStrip, Panel, PrimaryButton } from "../components/ui";
import { useImage } from "../hooks/useImage";
import { STYLES, downloadDataUrl, num } from "../lib/format";
import { runTask } from "../lib/api";

export default function FaceToSketch() {
  const source = useImage();
  const [style, setStyle] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const selected = STYLES[style];

  const run = async () => {
    if (!source.file) {
      setError("Upload a facial photograph or capture one with the webcam first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await runTask("/api/face-to-sketch", { file: source.file, fields: { style } }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const metricItems = [
    {
      icon: "draw",
      label: "Generated Sketch",
      value: selected.name,
      accent: "text-primary-container",
      foot: selected.title.replace("Style 1: ", "").replace("Style 2: ", "").replace("Style 3: ", ""),
    },
    {
      icon: "speed",
      label: "Generation Latency",
      value: num(result?.timing?.inference_ms, 2, " ms"),
      tone: "secondary",
      foot: result ? `total ${num(result.timing.total_ms, 1, " ms")} - generator only` : "discriminator not used at inference",
    },
    {
      icon: "layers",
      label: "Style Conditioning",
      value: `s = ${style}`,
      tone: "tertiary",
      foot: "learned categorical embedding",
    },
  ];

  return (
    <>
      <WorkspaceHeader
        icon="gesture"
        eyebrow="Task 4 - Style-Conditioned Face-to-Sketch cGAN"
        title="Face-to-Sketch Generator"
        description="A U-Net generator conditioned on a learned style embedding turns a facial photograph into one of the three FS2K sketch styles. Only the generator runs at inference."
        model="task4_generator.onnx"
        extra={
          <span className="flex flex-wrap items-center gap-2">
            <Chip tone="primary">U-Net generator</Chip>
            <Chip tone="secondary">PatchGAN trained</Chip>
          </span>
        }
      />

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="flex flex-col gap-5 lg:col-span-5">
          <Panel step={1} title="Input Face" subtitle="Upload a photo or use the webcam">
            <Dropzone file={source.file} preview={source.preview} onFile={source.set} label="Upload a facial photograph" hint="PNG / JPG / WebP" />
            <WebcamCapture onCapture={source.set} disabled={loading} />
            <p className="text-[11px] leading-relaxed text-outline">
              The photograph is resized to 128×128 with the same bicubic resampling used for FS2K training.
            </p>
          </Panel>

          <Panel step={2} title="Choose Style" subtitle="Categorical condition s ∈ {1, 2, 3}">
            <div className="flex flex-col gap-2.5">
              {STYLES.map((s) => {
                const active = s.index === style;
                return (
                  <button
                    key={s.index}
                    type="button"
                    onClick={() => setStyle(s.index)}
                    className={`flex items-center gap-3 rounded-xl border p-3 text-left transition-all ${
                      active
                        ? "border-primary-container bg-surface-container-high shadow-md"
                        : "border-outline-variant/40 bg-surface-container hover:border-outline"
                    }`}
                  >
                    <span
                      className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border font-headline text-lg font-bold ${
                        active ? "border-primary-container/50 bg-primary-container/15 text-primary-container" : "border-outline-variant/40 text-outline"
                      }`}
                    >
                      {s.index + 1}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center justify-between gap-2">
                        <span className="truncate font-headline text-sm font-semibold text-on-surface">{s.title}</span>
                        <Icon
                          name={active ? "check_circle" : "radio_button_unchecked"}
                          className={`text-[18px] ${active ? "text-primary-container" : "text-outline-variant"}`}
                        />
                      </span>
                      <span className="mt-0.5 block truncate text-xs text-on-surface-variant">{s.blurb}</span>
                    </span>
                  </button>
                );
              })}
            </div>

            <PrimaryButton loading={loading} onClick={run}>
              {loading ? "Generating..." : "Generate Sketch"}
            </PrimaryButton>

            {error ? (
              <Alert kind="error" onDismiss={() => setError(null)}>
                {error}
              </Alert>
            ) : null}
          </Panel>
        </div>

        <div className="flex flex-col gap-5 lg:col-span-7">
          <Panel
            step={3}
            title="Generate & Compare"
            subtitle="Original photograph next to the generated sketch"
            right={<Chip tone="primary">{result ? result.style.name : selected.name}</Chip>}
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ImageView
                src={result?.input ?? source.preview}
                label="Original Photograph"
                tone="neutral"
                badge={<Chip tone="neutral">source x</Chip>}
                emptyText="Upload or capture a photograph"
              />
              <ImageView
                src={result?.output}
                label="Generated Sketch"
                tone="tertiary"
                badge={<Chip tone="tertiary">{selected.name}</Chip>}
                sub={result ? `${result.model?.file ?? ""}` : ""}
                overlay={
                  loading ? (
                    <span className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-lg bg-surface-container-lowest/80">
                      <Icon name="progress_activity" className="animate-spin text-3xl text-primary-container" />
                      <span className="font-mono text-[11px] uppercase tracking-wide text-on-surface">drawing</span>
                    </span>
                  ) : null
                }
                emptyText="The generated sketch appears here"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-outline-variant/30 pt-4">
              <span className="font-mono text-[11px] text-on-surface-variant">
                inference <span className="text-primary-container">{num(result?.timing?.inference_ms, 2, " ms")}</span>
                <span className="text-outline"> - y = G(x, s)</span>
              </span>
              <GhostButton
                disabled={!result}
                onClick={() => downloadDataUrl(result.output, `face_to_sketch_${selected.name.toLowerCase().replace(/\s+/g, "_")}.png`)}
                className={result ? "border-tertiary/40 text-tertiary" : ""}
              >
                <Icon name="download" className="text-[15px]" />
                Download sketch PNG
              </GhostButton>
            </div>
          </Panel>

          <div className="flex items-start gap-2.5 rounded-xl border border-outline-variant/40 bg-surface-container-low p-3.5">
            <Icon name="lightbulb" className="mt-0.5 shrink-0 text-[18px] text-secondary" />
            <p className="text-[11px] leading-relaxed text-on-surface-variant">
              The style condition is injected into both the generator and the discriminator as a learned embedding - it is a real
              input to the network, not a label on the interface. Switch styles to see how the same photograph is rendered.
            </p>
          </div>
        </div>
      </div>

      <MetricStrip items={metricItems} />
      <RecordedResults task="task4" />
    </>
  );
}
