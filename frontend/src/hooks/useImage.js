import { useCallback, useRef, useState } from "react";
import { urlToFile } from "../lib/api";

/** Holds an input file plus an object-URL preview. */
export function useImage(initial = null) {
  const [file, setFile] = useState(initial);
  const [preview, setPreview] = useState(initial ? URL.createObjectURL(initial) : "");
  const urlRef = useRef("");

  const set = useCallback(
    (next) => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      setFile(next);
      const url = next ? URL.createObjectURL(next) : "";
      urlRef.current = url;
      setPreview(url);
    },
    []
  );

  const setFromUrl = useCallback(
    async (url, name) => {
      const f = await urlToFile(url, name);
      set(f);
      return f;
    },
    [set]
  );

  const clear = useCallback(() => set(null), [set]);

  return { file, preview, set, setFromUrl, clear };
}
