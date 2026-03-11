import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: "hsl(var(--card))",
        "card-foreground": "hsl(var(--card-foreground))",
        muted: "hsl(var(--muted))",
        "muted-foreground": "hsl(var(--muted-foreground))",
        accent: "hsl(var(--accent))",
        "accent-foreground": "hsl(var(--accent-foreground))",
        primary: "hsl(var(--primary))",
        "primary-foreground": "hsl(var(--primary-foreground))",
        destructive: "hsl(var(--destructive))",

        "claude-bg-0": "var(--claude-bg-0)",
        "claude-bg-000": "var(--claude-bg-000)",
        "claude-bg-100": "var(--claude-bg-100)",
        "claude-bg-200": "var(--claude-bg-200)",
        "claude-bg-300": "var(--claude-bg-300)",
        "claude-text-100": "var(--claude-text-100)",
        "claude-text-200": "var(--claude-text-200)",
        "claude-text-300": "var(--claude-text-300)",
        "claude-text-400": "var(--claude-text-400)",
        "claude-text-500": "var(--claude-text-500)",
        "claude-accent": "var(--claude-accent)",
        "claude-accent-hover": "var(--claude-accent-hover)",
      },
      fontFamily: {
        serif: ['"Source Serif 4"', "Georgia", "serif"],
        sans: ["var(--font-sans)", "Inter", "Onest", "system-ui", "sans-serif"],
      },
      borderRadius: {
        lg: "var(--radius)",
      },
      boxShadow: {
        "claude-input": "0 1px 2px -1px rgba(0,0,0,0.08), 0 2px 8px -2px rgba(0,0,0,0.04)",
        "claude-input-hover": "0 1px 2px -1px rgba(0,0,0,0.08), 0 4px 12px -2px rgba(0,0,0,0.08)",
        "claude-input-focus": "0 0 0 2px rgba(217,119,87,0.1), 0 4px 12px -2px rgba(0,0,0,0.08)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in": {
          "0%": { opacity: "0", transform: "translateX(-8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        "claude-fade-in": {
          "0%": {
            opacity: "0",
            transform: "translateY(8px) scale(0.98)",
            filter: "blur(4px)",
          },
          "100%": {
            opacity: "1",
            transform: "translateY(0) scale(1)",
            filter: "blur(0)",
          },
        },
      },
      animation: {
        "fade-in": "fade-in 0.3s ease-out",
        "slide-in": "slide-in 0.2s ease-out",
        pulse: "pulse 1.5s ease-in-out infinite",
        "claude-fade-in": "claude-fade-in 0.4s ease-out forwards",
      },
    },
  },
  plugins: [],
};

export default config;