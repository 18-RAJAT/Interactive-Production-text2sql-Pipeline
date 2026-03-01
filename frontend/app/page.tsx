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
import { HeroGeometric } from "@/components/ui/shape-landing-hero";
import { cn } from "@/lib/cn";

const features = [
  {
    icon: Brain,
    title: "LoRA Fine-Tuned",
    description:
      "Powered by a LLM fine-tuned with LoRA on thousands of text-to-SQL pairs for high accuracy.",
    gradient: "from-violet-500 to-indigo-500",
  },
  {
    icon: Zap,
    title: "Instant Generation",
    description:
      "Convert natural language questions into optimized SQL queries in milliseconds.",
    gradient: "from-amber-500 to-orange-500",
  },
  {
    icon: Database,
    title: "Schema-Aware",
    description:
      "Understands your database schema to generate contextually accurate, runnable queries.",
    gradient: "from-emerald-500 to-teal-500",
  },
  {
    icon: Code2,
    title: "Multi-Dialect",
    description:
      "Generates SQL compatible with PostgreSQL, MySQL, SQLite and more.",
    gradient: "from-rose-500 to-pink-500",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.7,
      delay: i * 0.1,
      ease: [0.25, 0.4, 0.25, 1] as [number, number, number, number],
    },
  }),
};

function FeatureCard({
  icon: Icon,
  title,
  description,
  gradient,
  index,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  gradient: string;
  index: number;
}) {
  return (
    <motion.div
      custom={index}
      variants={fadeUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-60px" }}
      className="group relative"
    >
      <div className="relative h-full rounded-2xl border border-white/[0.06] bg-white/[0.02] backdrop-blur-sm p-6 transition-colors hover:border-white/[0.12] hover:bg-white/[0.04]">
        <div
          className={cn(
            "inline-flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br mb-4",
            gradient
          )}
        >
          <Icon className="w-5 h-5 text-white" />
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
        <p className="text-sm text-white/50 leading-relaxed">{description}</p>
      </div>
    </motion.div>
  );
}

function HowItWorks() {
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

  return (
    <section className="relative py-24 md:py-32">
      <div className="container mx-auto px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white mb-4">
            Three Steps.{" "}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-300 to-rose-300">
              Zero Complexity.
            </span>
          </h2>
          <p className="text-white/40 text-lg max-w-md mx-auto">
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
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-white/[0.04] border border-white/[0.08] mb-5">
                <step.icon className="w-6 h-6 text-indigo-400" />
              </div>
              <div className="text-xs font-mono text-indigo-400/60 tracking-widest uppercase mb-2">
                Step {step.number}
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                {step.title}
              </h3>
              <p className="text-sm text-white/40 leading-relaxed">
                {step.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className="relative py-24 md:py-32">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-indigo-500/[0.03] to-transparent" />
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7 }}
        className="relative container mx-auto px-4 md:px-6 text-center"
      >
        <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white mb-4">
          Ready to{" "}
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-300 to-rose-300">
            Query Smarter
          </span>
          ?
        </h2>
        <p className="text-white/40 text-lg max-w-lg mx-auto mb-10">
          Stop writing SQL by hand. Let the fine-tuned model handle the
          translation while you focus on insights.
        </p>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-white text-[#030303] font-semibold text-base hover:bg-white/90 transition-colors"
        >
          Launch App
          <ArrowRight className="w-4 h-4" />
        </Link>
      </motion.div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-white/[0.06] py-8">
      <div className="container mx-auto px-4 md:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-indigo-400" />
          <span className="text-sm font-medium text-white/60">
            Text-to-SQL
          </span>
        </div>
        <p className="text-xs text-white/30">
          Built with LoRA fine-tuning &middot; Powered by open-source LLMs
        </p>
      </div>
    </footer>
  );
}

export default function LandingPage() {
  return (
    <div className="bg-[#030303] min-h-screen">
      <HeroGeometric
        badge="Text-to-SQL"
        title1="Natural Language to"
        title2="SQL Queries"
        description="Transform plain English questions into precise SQL queries using a LoRA fine-tuned language model. Schema-aware, instant, and open-source."
      >
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-white text-[#030303] font-semibold text-sm hover:bg-white/90 transition-colors"
          >
            Open Dashboard
            <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full border border-white/[0.12] text-white/70 font-medium text-sm hover:bg-white/[0.04] transition-colors"
          >
            View on GitHub
          </a>
        </div>
      </HeroGeometric>

      <section className="relative py-24 md:py-32">
        <div className="container mx-auto px-4 md:px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white mb-4">
              Why{" "}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-300 to-rose-300">
                Text-to-SQL
              </span>
              ?
            </h2>
            <p className="text-white/40 text-lg max-w-md mx-auto">
              Fine-tuned precision meets developer experience.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 max-w-5xl mx-auto">
            {features.map((feature, i) => (
              <FeatureCard key={feature.title} {...feature} index={i} />
            ))}
          </div>
        </div>
      </section>

      <HowItWorks />
      <CTASection />
      <Footer />
    </div>
  );
}
