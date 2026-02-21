"use client";

import { KeyboardEvent } from "react";
import { Send, Loader2, Sparkles } from "lucide-react";
import { cn } from "@/lib/cn";

interface QueryInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
  disabled?: boolean;
}

export function QueryInput({
  value,
  onChange,
  onSubmit,
  loading,
  disabled,
}: QueryInputProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading && value.trim()) onSubmit();
    }
  };

  const canSubmit = value.trim().length > 0 && !loading && !disabled;

  return (
    <div className="border rounded-xl bg-card overflow-hidden transition-shadow focus-within:ring-2 focus-within:ring-primary/20">
      <div className="flex items-center gap-2 px-4 pt-3">
        <Sparkles className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium">Ask a Question</span>
      </div>

      <div className="relative">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. How many employees earn more than 50000?"
          rows={3}
          className="w-full px-4 pt-2 pb-3 bg-transparent resize-none text-sm leading-relaxed placeholder:text-muted-foreground/50 focus:outline-none"
          disabled={loading}
        />
      </div>

      <div className="flex items-center justify-between px-4 pb-3">
        <span className="text-[11px] text-muted-foreground">
          Press <kbd className="px-1 py-0.5 rounded border bg-muted text-[10px] font-mono">Enter</kbd> to generate
          {" "}&middot;{" "}
          <kbd className="px-1 py-0.5 rounded border bg-muted text-[10px] font-mono">Shift+Enter</kbd> for new line
        </span>

        <button
          onClick={onSubmit}
          disabled={!canSubmit}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
            canSubmit
              ? "bg-primary text-primary-foreground hover:opacity-90 active:scale-[0.98]"
              : "bg-muted text-muted-foreground cursor-not-allowed"
          )}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Send className="w-3.5 h-3.5" />
              Generate SQL
            </>
          )}
        </button>
      </div>
    </div>
  );
}