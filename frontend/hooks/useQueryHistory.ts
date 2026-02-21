"use client";

import { useState, useEffect, useCallback } from "react";
import type { QueryHistoryItem } from "@/types/api";
import { STORAGE_KEY } from "@/lib/constants";

const MAX_HISTORY = 50;

export function useQueryHistory() {
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setHistory(JSON.parse(stored));
    } catch {
      /* corrupted storage, start fresh */
    }
  }, []);

  const persist = useCallback((items: QueryHistoryItem[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch {
      /* storage full or unavailable */
    }
  }, []);

  const addEntry = useCallback(
    (entry: Omit<QueryHistoryItem, "id" | "timestamp">) => {
      setHistory((prev) => {
        const item: QueryHistoryItem = {
          ...entry,
          id: crypto.randomUUID(),
          timestamp: Date.now(),
        };
        const next = [item, ...prev].slice(0, MAX_HISTORY);
        persist(next);
        return next;
      });
    },
    [persist]
  );

  const removeEntry = useCallback(
    (id: string) => {
      setHistory((prev) => {
        const next = prev.filter((item) => item.id !== id);
        persist(next);
        return next;
      });
    },
    [persist]
  );

  const clearHistory = useCallback(() => {
    setHistory([]);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return { history, addEntry, removeEntry, clearHistory };
}