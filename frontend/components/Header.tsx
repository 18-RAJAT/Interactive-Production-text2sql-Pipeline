"use client";

import { Moon, Sun, Database, Wifi, WifiOff } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";

interface HeaderProps {
  connected: boolean | null;
}

export function Header({ connected }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  return (
    <header className="flex items-center justify-between border-b px-6 py-3 bg-card">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10">
          <Database className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Text-to-SQL</h1>
          <p className="text-xs text-muted-foreground">
            QLoRA Fine-Tuned Generator
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div
          className={cn(
            "flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full transition-colors",
            connected === true && "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
            connected === false && "bg-destructive/10 text-destructive",
            connected === null && "bg-muted text-muted-foreground"
          )}
        >
          {connected === true ? (
            <Wifi className="w-3.5 h-3.5" />
          ) : (
            <WifiOff className="w-3.5 h-3.5" />
          )}
          {connected === true
            ? "Backend Connected"
            : connected === false
            ? "Backend Offline"
            : "Checking..."}
        </div>

        {mounted && (
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </button>
        )}
      </div>
    </header>
  );
}