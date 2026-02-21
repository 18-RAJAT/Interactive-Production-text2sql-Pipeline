"use client";

import { useState, useCallback } from "react";
import { generateSQL } from "@/lib/api";
import type { GenerateSQLResponse, APIError } from "@/types/api";

interface UseGenerateSQLReturn {
  result: GenerateSQLResponse | null;
  loading: boolean;
  error: string | null;
  generate: (question: string, schema: string) => Promise<GenerateSQLResponse | null>;
  reset: () => void;
}

export function useGenerateSQL(): UseGenerateSQLReturn {
  const [result, setResult] = useState<GenerateSQLResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(
    async (question: string, schema: string) => {
      if (!question.trim() || !schema.trim()) {
        setError("Both question and schema are required");
        return null;
      }

      setLoading(true);
      setError(null);
      setResult(null);

      try {
        const data = await generateSQL({ question, schema });
        setResult(data);
        return data;
      } catch (err) {
        const apiErr = err as APIError;
        setError(apiErr.message || "An unexpected error occurred");
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, generate, reset };
}