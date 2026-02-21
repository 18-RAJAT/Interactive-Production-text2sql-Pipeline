"use client";

import { useState } from "react";
import { Copy, Check, Code2, Gauge, Table2 } from "lucide-react";
import { cn } from "@/lib/cn";
import type { GenerateSQLResponse } from "@/types/api";

interface SQLOutputProps {
  result: GenerateSQLResponse | null;
  error: string | null;
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color =
    pct >= 80
      ? "text-emerald-600 bg-emerald-500/10 dark:text-emerald-400"
      : pct >= 50
      ? "text-amber-600 bg-amber-500/10 dark:text-amber-400"
      : "text-red-500 bg-red-500/10";

  return (
    <span className={cn("flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-md", color)}>
      <Gauge className="w-3 h-3" />
      {pct}% confidence
    </span>
  );
}

function ResultsTable({ data }: { data: Record<string, unknown>[] }) {
  if (!data.length) return null;
  const columns = Object.keys(data[0]);

  return (
    <div className="border rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 border-b">
        <Table2 className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs font-medium">Execution Results</span>
        <span className="text-[11px] text-muted-foreground">({data.length} rows)</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b bg-muted/30">
              {columns.map((col) => (
                <th key={col} className="text-left px-3 py-2 font-medium text-muted-foreground">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 20).map((row, i) => (
              <tr key={i} className="border-b last:border-0 hover:bg-muted/20 transition-colors">
                {columns.map((col) => (
                  <td key={col} className="px-3 py-1.5 font-mono">
                    {String(row[col] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function SQLOutput({ result, error }: SQLOutputProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!result?.generated_sql) return;
    await navigator.clipboard.writeText(result.generated_sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (error) {
    return (
      <div className="border border-destructive/30 rounded-xl bg-destructive/5 p-4 animate-fade-in">
        <p className="text-sm text-destructive font-medium">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="border border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center">
        <Code2 className="w-10 h-10 text-muted-foreground/30 mb-3" />
        <p className="text-sm text-muted-foreground">
          Generated SQL will appear here
        </p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          Enter a schema and question above to get started
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="border rounded-xl overflow-hidden bg-card">
        <div className="flex items-center justify-between px-4 py-2.5 border-b bg-muted/30">
          <div className="flex items-center gap-2">
            <Code2 className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">Generated SQL</span>
          </div>
          <div className="flex items-center gap-2">
            {result.confidence !== undefined && (
              <ConfidenceBadge confidence={result.confidence} />
            )}
            {result.latency_ms !== undefined && (
              <span className="text-[11px] text-muted-foreground">{result.latency_ms}ms</span>
            )}
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded-md hover:bg-muted transition-colors"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-500" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>

        <pre className="p-4 overflow-x-auto scrollbar-thin">
          <code className="text-sm font-mono leading-relaxed text-primary">
            {result.generated_sql}
          </code>
        </pre>
      </div>

      {result.execution_result && result.execution_result.length > 0 && (
        <ResultsTable data={result.execution_result} />
      )}
    </div>
  );
}