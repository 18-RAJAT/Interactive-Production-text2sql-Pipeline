"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Database, LayoutDashboard, MessageSquare, Home } from "lucide-react";
import { cn } from "@/lib/cn";

const navLinks = [
  { href: "/", label: "Home", icon: Home },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 flex items-center justify-between px-5 py-2.5 border-b border-[#E8E5DD] bg-[#FAF9F5]/90 backdrop-blur-xl">
      <Link href="/" className="flex items-center gap-2.5 group">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10 group-hover:bg-primary/15 transition-colors">
          <Database className="w-4 h-4 text-primary" />
        </div>
        <span className="text-sm font-semibold tracking-tight text-foreground hidden sm:block">
          Text-to-SQL
        </span>
      </Link>

      <div className="flex items-center gap-1">
        {navLinks.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted",
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">{label}</span>
              {isActive && (
                <span className="absolute -bottom-[13px] left-1/2 -translate-x-1/2 w-5 h-[2px] rounded-full bg-primary" />
              )}
            </Link>
          );
        })}
      </div>

      <div className="w-8" />
    </nav>
  );
}
