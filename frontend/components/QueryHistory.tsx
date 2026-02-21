"use client";

import { History, Trash2, X } from "lucide-react";
import { cn } from "@/lib/cn";
import type { QueryHistoryItem } from "@/types/api";

interface QueryHistoryProps {
  history: QueryHistoryItem[];
  onSelect: (item: QueryHistoryItem) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

function formatTime(ts: number): string {
  const now = Date.now();
  const diff = now - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(ts).toLocaleDateString();
}

export function QueryHistory({
  history,
  onSelect,
  onRemove,
  onClear,
}: QueryHistoryProps) {
  if (history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-4">
        <History className="w-8 h-8 text-muted-foreground/30 mb-2" />
        <p className="text-xs text-muted-foreground">No queries yet</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium">History</span>
          <span className="text-[11px] text-muted-foreground">({history.length})</span>
        </div>
        <button
          onClick={onClear}
          className="text-[11px] text-muted-foreground hover:text-destructive transition-colors"
        >
          Clear all
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {history.map((item) => (
          <div
            key={item.id}
            className="group relative border-b last:border-0 hover:bg-muted/50 transition-colors cursor-pointer"
            onClick={() => onSelect(item)}
          >
            <div className="px-4 py-3">
              <p className="text-xs font-medium leading-snug line-clamp-2 pr-6">
                {item.question}
              </p>
              <p className="text-[11px] text-muted-foreground mt-1 font-mono line-clamp-1">
                {item.generated_sql}
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">
                {formatTime(item.timestamp)}
              </p>
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove(item.id);
              }}
              className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-destructive/10 hover:text-destructive transition-all"
              aria-label="Remove"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}