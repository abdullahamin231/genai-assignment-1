import React, { useEffect, useRef, useState } from "react";
import { Icon } from "./ui";

/**
 * Optional webcam capture (Task 4). Falls back to an explanatory message when
 * the browser or origin does not grant camera access.
 */
export default function WebcamCapture({ onCapture, disabled = false }) {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (stream && videoRef.current) videoRef.current.srcObject = stream;
  }, [stream]);

  useEffect(
    () => () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    },
    [stream]
  );

  const start = async () => {
    setError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("unsupported");
      setStream(await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } }));
    } catch {
      setError("Camera unavailable - grant permission or upload a photo instead.");
    }
  };

  const stop = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setStream(null);
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) onCapture(new File([blob], "webcam_capture.png", { type: "image/png" }));
      stop();
    }, "image/png");
  };

  if (!stream) {
    return (
      <div className="flex flex-col gap-1.5">
        <button
          type="button"
          onClick={start}
          disabled={disabled}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-outline-variant/50 bg-surface-container px-3 py-2.5 text-xs font-medium text-on-surface-variant transition-colors hover:border-primary-container/60 hover:text-primary-container disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Icon name="photo_camera" className="text-[16px]" />
          Capture from webcam
        </button>
        {error ? <p className="text-[11px] text-warning">{error}</p> : null}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-primary-container/40 bg-surface-container-lowest p-2">
      <video ref={videoRef} autoPlay playsInline muted className="aspect-[4/3] w-full rounded-lg object-cover" />
      <div className="flex gap-2">
        <button
          type="button"
          onClick={capture}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary-container px-3 py-2 text-xs font-semibold text-on-primary transition hover:brightness-110"
        >
          <Icon name="camera" className="text-[15px]" />
          Use frame
        </button>
        <button
          type="button"
          onClick={stop}
          className="rounded-lg border border-outline-variant/50 px-3 py-2 text-xs text-on-surface-variant transition-colors hover:text-on-surface"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
