"use client";

import { useState, useEffect, useCallback } from "react";
import { checkHealth } from "@/lib/api";

export function useBackendStatus(pollIntervalMs = 15000) {
  const [connected, setConnected] = useState<boolean | null>(null);

  const check = useCallback(async () => {
    const ok = await checkHealth();
    setConnected(ok);
  }, []);

  useEffect(() => {
    check();
    const interval = setInterval(check, pollIntervalMs);
    return () => clearInterval(interval);
  }, [check, pollIntervalMs]);

  return { connected, recheck: check };
}