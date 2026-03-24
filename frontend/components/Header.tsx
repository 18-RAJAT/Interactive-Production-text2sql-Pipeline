"use client";

import { Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/cn";

interface HeaderProps {
  connected: boolean | null;
}

export function Header({ connected }: HeaderProps) {
  return (
    <header className="flex items-center justify-end border-b px-6 py-2 bg-card">
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
    </header>
  );
}