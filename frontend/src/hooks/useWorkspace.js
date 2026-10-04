import { useCallback, useState } from "react";
import { runTask } from "../lib/api";
import { CLEAN_STATE, corruptionFields } from "../components/CorruptionControls";
import { useImage } from "./useImage";

/**
 * Shared state machine for the three restoration workspaces:
 * input image, optional reference, corruption controls, result and error.
 */
export default function useWorkspace(path) {
  const source = useImage();
  const reference = useImage();
  const [controls, setControls] = useState(CLEAN_STATE);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRequest, setLastRequest] = useState(null);

  const run = useCallback(
    async (extra = {}) => {
      if (!source.file) {
        setError("Select a preset sample or upload an image first.");
        return null;
      }
      setLoading(true);
      setError(null);
      const fields = { ...corruptionFields(controls), ...extra };
      setLastRequest(fields);
      try {
        const data = await runTask(path, { file: source.file, reference: reference.file, fields });
        setResult(data);
        return data;
      } catch (err) {
        setError(err.message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [path, source.file, reference.file, controls]
  );

  const pickSample = useCallback(
    async (sample) => {
      try {
        await source.setFromUrl(sample.url, `${sample.label}.png`);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    },
    [source]
  );

  return { source, reference, controls, setControls, result, setResult, loading, error, setError, run, pickSample, lastRequest };
}
