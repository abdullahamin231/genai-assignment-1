import React, { useState } from "react";
import SamplePicker from "./SamplePicker";
import { Dropzone, Icon } from "./ui";

/**
 * Step 1 of every restoration workspace: preset sample, upload / drag-and-drop
 * and (optionally) a clean reference image used for PSNR / SSIM reporting.
 */
export default function ImageSource({ source, onSample, showSamples = true, reference = null, sampleName = "", children }) {
  const [showRef, setShowRef] = useState(Boolean(reference?.file));

  return (
    <>
      {showSamples ? <SamplePicker onPick={onSample} activeName={sampleName} /> : null}

      <Dropzone file={source.file} preview={source.preview} onFile={source.set} label="Upload an image" hint="drag & drop or click" />

      {children}

      {reference ? (
        <div className="border-t border-outline-variant/30 pt-3">
          <button
            type="button"
            onClick={() => setShowRef((v) => !v)}
            className="flex w-full items-center justify-between gap-2 text-left"
          >
            <span className="flex items-center gap-1.5 text-xs font-medium text-on-surface-variant">
              <Icon name="target" className="text-[15px] text-outline" />
              Optional clean reference
            </span>
            <Icon name={showRef ? "expand_less" : "expand_more"} className="text-[16px] text-outline" />
          </button>
          <p className="mt-1 text-[11px] leading-relaxed text-outline">
            Add the ground-truth image to report PSNR / SSIM against it instead of against the network input.
          </p>
          {showRef ? (
            <div className="mt-2 flex items-center gap-2">
              <Dropzone
                className="flex-1"
                file={reference.file}
                preview={reference.preview}
                onFile={reference.set}
                label="Reference image"
                hint="clean target"
              />
              {reference.file ? (
                <button
                  type="button"
                  onClick={reference.clear}
                  className="rounded-lg border border-outline-variant/40 p-2 text-on-surface-variant transition-colors hover:border-error/50 hover:text-error"
                  title="Clear reference"
                >
                  <Icon name="close" className="text-[16px]" />
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
