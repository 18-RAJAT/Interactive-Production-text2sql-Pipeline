"use client";

import { useState } from "react";
import { RotateCcw, ChevronDown, TableProperties } from "lucide-react";
import { cn } from "@/lib/cn";
import { SAMPLE_SCHEMAS } from "@/lib/constants";

interface SchemaEditorProps {
  value: string;
  onChange: (value: string) => void;
}

export function SchemaEditor({ value, onChange }: SchemaEditorProps) {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <TableProperties className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium">Database Schema</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md bg-muted hover:bg-accent transition-colors"
            >
              Samples
              <ChevronDown className={cn("w-3 h-3 transition-transform", dropdownOpen && "rotate-180")} />
            </button>

            {dropdownOpen && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setDropdownOpen(false)}
                />
                <div className="absolute right-0 top-full mt-1 z-20 w-48 rounded-lg border bg-card shadow-lg animate-fade-in">
                  {SAMPLE_SCHEMAS.map((schema) => (
                    <button
                      key={schema.label}
                      onClick={() => {
                        onChange(schema.value);
                        setDropdownOpen(false);
                      }}
                      className="w-full text-left text-xs px-3 py-2 hover:bg-muted transition-colors first:rounded-t-lg last:rounded-b-lg"
                    >
                      {schema.label}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          <button
            onClick={() => onChange("")}
            className="p-1.5 rounded-md hover:bg-muted transition-colors text-muted-foreground"
            aria-label="Clear schema"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={`CREATE TABLE employees (\n  id INTEGER PRIMARY KEY,\n  name TEXT,\n  salary REAL\n);`}
        className="flex-1 w-full p-4 bg-transparent resize-none text-sm font-mono leading-relaxed placeholder:text-muted-foreground/50 focus:outline-none scrollbar-thin"
        spellCheck={false}
      />
    </div>
  );
}