"use client";

import { useState, useCallback } from "react";
import { PanelLeftClose, PanelLeft } from "lucide-react";
import { Header } from "@/components/Header";
import { SchemaEditor } from "@/components/SchemaEditor";
import { QueryInput } from "@/components/QueryInput";
import { SQLOutput } from "@/components/SQLOutput";
import { QueryHistory } from "@/components/QueryHistory";
import { useGenerateSQL } from "@/hooks/useGenerateSQL";
import { useQueryHistory } from "@/hooks/useQueryHistory";
import { useBackendStatus } from "@/hooks/useBackendStatus";
import { useToast } from "@/components/Toast";
import { cn } from "@/lib/cn";
import type { QueryHistoryItem } from "@/types/api";

export default function Home() {
  const [schema, setSchema] = useState("");
  const [question, setQuestion] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const { result, loading, error, generate, reset } = useGenerateSQL();
  const { history, addEntry, removeEntry, clearHistory } = useQueryHistory();
  const { connected } = useBackendStatus();
  const { toast } = useToast();

  const handleGenerate = useCallback(async () => {
    if (!question.trim()) {
      toast("Please enter a question", "error");
      return;
    }
    if (!schema.trim()) {
      toast("Please provide a database schema", "error");
      return;
    }

    const data = await generate(question, schema);
    if (data) {
      addEntry({
        question,
        schema,
        generated_sql: data.generated_sql,
        confidence: data.confidence,
      });
      toast("SQL generated successfully", "success");
    }
  }, [question, schema, generate, addEntry, toast]);

  const handleHistorySelect = useCallback(
    (item: QueryHistoryItem) => {
      setSchema(item.schema);
      setQuestion(item.question);
      reset();
    },
    [reset]
  );

  return (
    <div className="flex flex-col h-screen">
      <Header connected={connected} />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — Query History */}
        <aside
          className={cn(
            "border-r bg-card flex-shrink-0 transition-all duration-300 overflow-hidden",
            sidebarOpen ? "w-72" : "w-0"
          )}
        >
          <QueryHistory
            history={history}
            onSelect={handleHistorySelect}
            onRemove={removeEntry}
            onClear={clearHistory}
          />
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Sidebar Toggle */}
          <div className="px-4 pt-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1.5 rounded-md hover:bg-muted transition-colors text-muted-foreground"
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? (
                <PanelLeftClose className="w-4 h-4" />
              ) : (
                <PanelLeft className="w-4 h-4" />
              )}
            </button>
          </div>

          <div className="flex-1 flex overflow-hidden">
            {/* Left: Schema Editor */}
            <div className="w-[45%] border-r flex flex-col overflow-hidden">
              <SchemaEditor value={schema} onChange={setSchema} />
            </div>

            {/* Right: Query + Output */}
            <div className="flex-1 flex flex-col overflow-hidden">
              <div className="flex-1 overflow-y-auto p-5 space-y-5 scrollbar-thin">
                <QueryInput
                  value={question}
                  onChange={setQuestion}
                  onSubmit={handleGenerate}
                  loading={loading}
                  disabled={connected === false}
                />
                <SQLOutput result={result} error={error} />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}