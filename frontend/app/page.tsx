"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Database,
  Zap,
  Brain,
  Code2,
  Sparkles,
  Terminal,
} from "lucide-react";
import { cn } from "@/lib/cn";

const features = [
  {
    icon: Brain,
    title: "LoRA Fine-Tuned",
    description:
      "Powered by a LLM fine-tuned with LoRA on thousands of text-to-SQL pairs for high accuracy.",
  },
  {
    icon: Zap,
    title: "Instant Generation",
    description:
      "Convert natural language questions into optimized SQL queries in milliseconds.",
  },
  {
    icon: Database,
    title: "Schema-Aware",
    description:
      "Understands your database schema to generate contextually accurate, runnable queries.",
  },
  {
    icon: Code2,
    title: "Multi-Dialect",
    description:
      "Generates SQL compatible with PostgreSQL, MySQL, SQLite and more.",
  },
];

const steps = [
  {
    number: "01",
    title: "Paste Your Schema",
    description: "Drop in your CREATE TABLE statements or schema definition.",
    icon: Terminal,
  },
  {
    number: "02",
    title: "Ask in Plain English",
    description:
      'Type a natural language question like "Show me total sales by region."',
    icon: Sparkles,
  },
  {
    number: "03",
    title: "Get Your SQL",
    description:
      "Receive an optimized, ready-to-run SQL query in milliseconds.",
    icon: Code2,
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay: i * 0.1, ease: [0.25, 0.4, 0.25, 1] as [number, number, number, number] },
  }),
};

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <section className="relative py-24 md:py-36 overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-primary/[0.04] blur-[120px]" />
          <div className="absolute top-1/4 right-1/4 w-[400px] h-[400px] rounded-full bg-primary/[0.03] blur-[80px]" />
        </div>

        <div className="relative container mx-auto px-4 md:px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" />
              Text-to-SQL
            </div>

            <h1 className="text-5xl sm:text-6xl md:text-7xl font-serif font-light tracking-tight text-foreground mb-6 leading-[1.1]">
              Natural Language to
              <br />
              <span className="text-primary italic">SQL Queries</span>
            </h1>

            <p className="text-lg text-muted-foreground max-w-xl mx-auto mb-10 leading-relaxed">
              Transform plain English questions into precise SQL queries
              using a LoRA fine-tuned language model. Schema-aware, instant,
              and open-source.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-primary text-primary-foreground font-medium text-sm hover:opacity-90 transition-opacity"
              >
                Open Dashboard
                <ArrowRight className="w-4 h-4" />
              </Link>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full border text-muted-foreground font-medium text-sm hover:bg-muted transition-colors"
              >
                View on GitHub
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="py-24 md:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-serif font-light tracking-tight text-foreground mb-4">
              Why <span className="text-primary italic">Text-to-SQL</span>?
            </h2>
            <p className="text-muted-foreground text-lg max-w-md mx-auto">
              Fine-tuned precision meets developer experience.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 max-w-5xl mx-auto">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                custom={i}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-60px" }}
                className="group"
              >
                <div className="h-full rounded-2xl border bg-card p-6 transition-all hover:shadow-sm hover:border-primary/20">
                  <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10 mb-4">
                    <feature.icon className="w-5 h-5 text-primary" />
                  </div>
                  <h3 className="text-base font-semibold text-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 md:py-32 bg-card">
        <div className="container mx-auto px-4 md:px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-serif font-light tracking-tight text-foreground mb-4">
              Three Steps.{" "}
              <span className="text-primary italic">Zero Complexity.</span>
            </h2>
            <p className="text-muted-foreground text-lg max-w-md mx-auto">
              From schema to query in seconds.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6 md:gap-8 max-w-4xl mx-auto">
            {steps.map((step, i) => (
              <motion.div
                key={step.number}
                custom={i}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-40px" }}
                className="relative text-center"
              >
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-background border mb-5">
                  <step.icon className="w-6 h-6 text-primary" />
                </div>
                <div className="text-xs font-mono text-primary/60 tracking-widest uppercase mb-2">
                  Step {step.number}
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">
                  {step.title}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {step.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-24 md:py-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="container mx-auto px-4 md:px-6 text-center"
        >
          <h2 className="text-3xl sm:text-4xl font-serif font-light tracking-tight text-foreground mb-4">
            Ready to{" "}
            <span className="text-primary italic">Query Smarter</span>?
          </h2>
          <p className="text-muted-foreground text-lg max-w-lg mx-auto mb-10">
            Stop writing SQL by hand. Let the fine-tuned model handle the
            translation while you focus on insights.
          </p>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-primary text-primary-foreground font-semibold text-base hover:opacity-90 transition-opacity"
          >
            Launch App
            <ArrowRight className="w-4 h-4" />
          </Link>
        </motion.div>
      </section>

      <footer className="border-t py-8">
        <div className="container mx-auto px-4 md:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-muted-foreground">
              Text-to-SQL
            </span>
          </div>
          <p className="text-xs text-muted-foreground">
            Built with LoRA fine-tuning &middot; Powered by open-source LLMs
          </p>
        </div>
      </footer>
    </div>
  );
}
