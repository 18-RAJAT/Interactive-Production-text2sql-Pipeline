"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare, Plus, PanelLeftClose, PanelLeft,
  Database, Copy, Check, Clock, Trash2,
  ChevronRight, Sparkles, AlertCircle, Wifi, WifiOff,
  BookOpen, Zap,
} from "lucide-react";
import ClaudeChatInput from "@/components/ui/claude-style-chat-input";
import type { AttachedFile } from "@/components/ui/claude-style-chat-input";
import { generateSQL, checkHealth } from "@/lib/api";
import { cn } from "@/lib/cn";
import { findKnowledgeEntry, classifyIntent, getAllTopicNames } from "@/lib/sql-knowledge";

type MessageType = "sql" | "explanation" | "error";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  type?: MessageType;
  sql?: string;
  exampleSql?: string;
  confidence?: number;
  latency?: number;
  timestamp: number;
}

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  schema: string;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "text-to-sql-conversations";
const ACTIVE_KEY = "text-to-sql-active-conv";

const MODELS = [
  { id: "lora-finetuned", name: "LoRA Fine-Tuned", description: "Schema-aware SQL generation" },
  { id: "base-model", name: "Base Model", description: "Standard text-to-SQL" },
  { id: "rule-engine", name: "Rule Engine", description: "Heuristic-based generation" },
];

const DEFAULT_SCHEMA = `CREATE TABLE employees (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  department VARCHAR(50),
  salary DECIMAL(10,2),
  hire_date DATE
);

CREATE TABLE departments (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  manager_id INT
);`;

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function formatTime(ts: number): string {
  const now = Date.now();
  const diff = now - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function groupConversations(convs: Conversation[]) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterday = today - 86400000;
  const weekAgo = today - 7 * 86400000;

  const groups: { label: string; items: Conversation[] }[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Older", items: [] },
  ];

  const sorted = [...convs].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const c of sorted) {
    if (c.updatedAt >= today) groups[0].items.push(c);
    else if (c.updatedAt >= yesterday) groups[1].items.push(c);
    else if (c.updatedAt >= weekAgo) groups[2].items.push(c);
    else groups[3].items.push(c);
  }

  return groups.filter(g => g.items.length > 0);
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
    >
      {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function SqlBlock({ code, label }: { code: string; label?: string }) {
  return (
    <div className="mt-3 rounded-lg overflow-hidden border border-claude-bg-300/60">
      <div className="flex items-center justify-between px-3 py-1.5 bg-claude-bg-200/70">
        <span className="text-[11px] font-medium text-claude-text-400 uppercase tracking-wider">
          {label || "SQL"}
        </span>
        <CopyButton text={code} />
      </div>
      <pre className="px-3 py-3 bg-claude-bg-0 overflow-x-auto custom-scrollbar">
        <code className="text-xs font-mono text-claude-text-200 leading-relaxed">{code}</code>
      </pre>
    </div>
  );
}

function ExplanationContent({ content }: { content: string }) {
  const parts = content.split("\n\n");
  return (
    <div className="space-y-2.5">
      {parts.map((block, i) => {
        if (block.startsWith("\u2022") || block.includes("\n\u2022")) {
          const lines = block.split("\n").filter(Boolean);
          return (
            <ul key={i} className="space-y-1">
              {lines.map((line, j) => (
                <li key={j} className="text-sm text-claude-text-200 leading-relaxed flex gap-2">
                  {line.startsWith("\u2022") ? (
                    <>
                      <span className="text-claude-accent mt-0.5 flex-shrink-0">{"\u2022"}</span>
                      <span>{line.slice(1).trim()}</span>
                    </>
                  ) : (
                    <span>{line}</span>
                  )}
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i} className="text-sm text-claude-text-200 leading-relaxed">
            {block}
          </p>
        );
      })}
    </div>
  );
}

function SchemaPanel({ schema, onChange }: { schema: string; onChange: (s: string) => void }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border-b border-claude-bg-300/50">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full px-4 py-2.5 text-xs font-medium text-claude-text-300 hover:text-claude-text-200 hover:bg-claude-bg-200/50 transition-colors"
      >
        <Database className="w-3.5 h-3.5" />
        <span>Schema Context</span>
        <ChevronRight className={cn("w-3 h-3 ml-auto transition-transform", expanded && "rotate-90")} />
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <textarea
              value={schema}
              onChange={(e) => onChange(e.target.value)}
              className="w-full h-40 px-4 py-2 text-xs font-mono bg-claude-bg-100 text-claude-text-200 resize-none outline-none border-t border-claude-bg-300/50 custom-scrollbar"
              placeholder="Paste your CREATE TABLE statements here..."
              spellCheck={false}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="flex gap-3 justify-end"
      >
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-claude-accent text-white px-4 py-3">
          <p className="text-sm leading-relaxed">{msg.content}</p>
          <p className="text-[10px] text-white/50 text-right mt-2">{formatTime(msg.timestamp)}</p>
        </div>
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-claude-accent flex items-center justify-center text-white text-xs font-semibold mt-0.5">
          R
        </div>
      </motion.div>
    );
  }

  const isError = msg.type === "error";
  const isExplanation = msg.type === "explanation";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex gap-3 justify-start"
    >
      <div className={cn(
        "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center mt-0.5",
        isExplanation ? "bg-blue-100" : "bg-claude-accent/10"
      )}>
        {isExplanation
          ? <BookOpen className="w-3.5 h-3.5 text-blue-600" />
          : <Sparkles className="w-3.5 h-3.5 text-claude-accent" />
        }
      </div>

      <div className={cn(
        "max-w-[85%] rounded-2xl rounded-bl-md px-4 py-3",
        isError
          ? "bg-red-50 border border-red-200"
          : "bg-claude-bg-100 border border-claude-bg-300/60"
      )}>
        {isError && (
          <div className="flex items-center gap-1.5 mb-1.5 text-red-600">
            <AlertCircle className="w-3.5 h-3.5" />
            <span className="text-xs font-medium">Error</span>
          </div>
        )}

        {isExplanation && msg.content && (
          <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-claude-bg-300/40">
            <span className="text-xs font-semibold text-claude-text-100">{msg.content}</span>
          </div>
        )}

        {isError ? (
          <p className="text-sm leading-relaxed text-red-600">{msg.content}</p>
        ) : isExplanation && msg.exampleSql ? (
          <>
            <ExplanationContent content={msg.sql || ""} />
            <SqlBlock code={msg.exampleSql} label="Example" />
          </>
        ) : (
          <>
            {!isExplanation && (
              <p className="text-sm leading-relaxed text-claude-text-200">{msg.content}</p>
            )}
            {msg.sql && !isExplanation && <SqlBlock code={msg.sql} />}
          </>
        )}

        {!isError && !isExplanation && (msg.confidence != null || msg.latency != null) && (
          <div className="flex items-center gap-3 mt-2.5 pt-2 border-t border-claude-bg-300/40">
            {msg.confidence != null && (
              <span className={cn(
                "text-[11px] font-medium px-2 py-0.5 rounded-full",
                msg.confidence >= 0.8 ? "bg-emerald-50 text-emerald-600" :
                msg.confidence >= 0.5 ? "bg-amber-50 text-amber-600" :
                "bg-red-50 text-red-500"
              )}>
                {(msg.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
            {msg.latency != null && (
              <span className="text-[11px] text-claude-text-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {msg.latency.toFixed(0)}ms
              </span>
            )}
          </div>
        )}

        <p className={cn(
          "text-[10px] mt-2",
          isError ? "text-red-400" : "text-claude-text-500"
        )}>
          {formatTime(msg.timestamp)}
        </p>
      </div>
    </motion.div>
  );
}

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingType, setLoadingType] = useState<"sql" | "explanation">("sql");
  const [connected, setConnected] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeConv = conversations.find(c => c.id === activeId) || null;

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setConversations(JSON.parse(stored) as Conversation[]);
      const storedActive = localStorage.getItem(ACTIVE_KEY);
      if (storedActive) setActiveId(storedActive);
    } catch {}
  }, []);

  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    }
  }, [conversations]);

  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
  }, [activeId]);

  useEffect(() => {
    const check = async () => setConnected(await checkHealth());
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConv?.messages.length, loading]);

  const updateConversation = useCallback((id: string, updater: (c: Conversation) => Conversation) => {
    setConversations(prev => prev.map(c => c.id === id ? updater(c) : c));
  }, []);

  const createNewChat = useCallback(() => {
    const newConv: Conversation = {
      id: generateId(),
      title: "New conversation",
      messages: [],
      schema: DEFAULT_SCHEMA,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations(prev => [newConv, ...prev]);
    setActiveId(newConv.id);
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeId === id) setActiveId(null);
  }, [activeId]);

  const buildFallbackTopicList = useCallback(() => {
    const topics = getAllTopicNames();
    const topicList = topics.slice(0, 15).map(t => `\u2022 "${t}"`).join("\n");
    return `I don't have a specific entry for that topic yet, but I cover ${topics.length} SQL topics. Here are some you can ask about:\n\n${topicList}\n\nYou can also ask me to generate SQL queries like:\n\n\u2022 "Show all employees with salary above 50000"\n\u2022 "Find departments with more than 10 employees"\n\u2022 "Get the average salary by department"`;
  }, []);

  const handleSend = useCallback(async (data: {
    message: string;
    files: AttachedFile[];
    model: string;
    isThinkingEnabled: boolean;
  }) => {
    if (!data.message.trim()) return;

    let convId = activeId;
    let schema = DEFAULT_SCHEMA;

    if (!convId) {
      const newConv: Conversation = {
        id: generateId(),
        title: data.message.slice(0, 60),
        messages: [],
        schema: DEFAULT_SCHEMA,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setConversations(prev => [newConv, ...prev]);
      setActiveId(newConv.id);
      convId = newConv.id;
      schema = newConv.schema;
    } else {
      const conv = conversations.find(c => c.id === convId);
      schema = conv?.schema || DEFAULT_SCHEMA;
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: data.message,
      timestamp: Date.now(),
    };

    const currentConvId = convId;

    updateConversation(currentConvId, c => {
      const title = c.messages.length === 0 ? data.message.slice(0, 60) : c.title;
      return { ...c, title, messages: [...c.messages, userMsg], updatedAt: Date.now() };
    });

    const intent = classifyIntent(data.message);

    if (intent === "explanation") {
      setLoading(true);
      setLoadingType("explanation");

      await new Promise(r => setTimeout(r, 400 + Math.random() * 400));

      const entry = findKnowledgeEntry(data.message);

      const assistantMsg: ChatMessage = entry
        ? {
            id: generateId(),
            role: "assistant",
            type: "explanation",
            content: entry.title,
            sql: entry.content,
            exampleSql: entry.example,
            timestamp: Date.now(),
          }
        : {
            id: generateId(),
            role: "assistant",
            type: "explanation",
            content: "SQL Topics",
            sql: buildFallbackTopicList(),
            timestamp: Date.now(),
          };

      updateConversation(currentConvId, c => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        updatedAt: Date.now(),
      }));

      setLoading(false);
      return;
    }

    setLoading(true);
    setLoadingType("sql");

    try {
      const response = await generateSQL({ question: data.message, schema });

      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        type: "sql",
        content: response.generated_sql
          ? "Here's the generated SQL query for your question:"
          : "I couldn't generate a SQL query for that question. Try rephrasing or check the schema context.",
        sql: response.generated_sql,
        confidence: response.confidence,
        latency: response.latency_ms,
        timestamp: Date.now(),
      };

      updateConversation(currentConvId, c => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        updatedAt: Date.now(),
      }));
    } catch (err: unknown) {
      const errorMessage = (err && typeof err === "object" && "message" in err)
        ? (err as { message: string }).message
        : "Failed to generate SQL. Please check if the API server is running.";

      const errorMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        type: "error",
        content: errorMessage,
        timestamp: Date.now(),
      };

      updateConversation(currentConvId, c => ({
        ...c,
        messages: [...c.messages, errorMsg],
        updatedAt: Date.now(),
      }));
    } finally {
      setLoading(false);
    }
  }, [activeId, conversations, updateConversation, buildFallbackTopicList]);

  const currentHour = new Date().getHours();
  let greeting = "Good morning";
  if (currentHour >= 12 && currentHour < 18) greeting = "Good afternoon";
  else if (currentHour >= 18) greeting = "Good evening";

  const groups = groupConversations(conversations);
  const showHero = !activeConv || activeConv.messages.length === 0;

  return (
    <div className="flex h-[calc(100vh-49px)] bg-claude-bg-0 font-sans text-claude-text-100">
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex-shrink-0 border-r border-claude-bg-300/50 bg-claude-bg-100 flex flex-col overflow-hidden"
          >
            <div className="flex items-center justify-between px-3 py-3 border-b border-claude-bg-300/50">
              <button
                onClick={createNewChat}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
              >
                <Plus className="w-4 h-4" />
                New chat
              </button>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 rounded-lg text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
              >
                <PanelLeftClose className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar px-2 py-2">
              {groups.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center p-4 opacity-50">
                  <MessageSquare className="w-8 h-8 text-claude-text-400 mb-2" />
                  <p className="text-xs text-claude-text-400">No conversations yet</p>
                </div>
              )}
              {groups.map(group => (
                <div key={group.label} className="mb-3">
                  <p className="px-2 py-1.5 text-[11px] font-medium text-claude-text-400 uppercase tracking-wider">
                    {group.label}
                  </p>
                  {group.items.map(conv => (
                    <div
                      key={conv.id}
                      onClick={() => setActiveId(conv.id)}
                      className={cn(
                        "group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-colors mb-0.5",
                        activeId === conv.id
                          ? "bg-claude-bg-200 text-claude-text-100"
                          : "text-claude-text-300 hover:bg-claude-bg-200/50 hover:text-claude-text-200"
                      )}
                    >
                      <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                      <span className="text-sm truncate flex-1">{conv.title}</span>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                        className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-claude-bg-300/50 hover:text-red-500 transition-all"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div className="px-3 py-2.5 border-t border-claude-bg-300/50">
              <div className={cn(
                "flex items-center gap-2 text-[11px] font-medium px-2 py-1 rounded-md",
                connected === true && "text-emerald-600",
                connected === false && "text-red-500",
                connected === null && "text-claude-text-400"
              )}>
                {connected === true ? <Wifi className="w-3 h-3" /> : connected === false ? <WifiOff className="w-3 h-3" /> : <Clock className="w-3 h-3 animate-pulse" />}
                {connected === true ? "API Connected" : connected === false ? "API Disconnected" : "Checking..."}
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col min-w-0">
        {!sidebarOpen && (
          <div className="flex items-center gap-2 px-3 py-2 border-b border-claude-bg-300/30">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded-lg text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
            >
              <PanelLeft className="w-4 h-4" />
            </button>
            <button
              onClick={createNewChat}
              className="p-1.5 rounded-lg text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        )}

        {activeConv && (
          <SchemaPanel
            schema={activeConv.schema}
            onChange={(s) => updateConversation(activeConv.id, c => ({ ...c, schema: s }))}
          />
        )}

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {showHero ? (
            <div className="flex flex-col items-center justify-center h-full p-4 animate-claude-fade-in">
              <div className="w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                  <defs>
                    <ellipse id="hero-petal" cx="100" cy="100" rx="90" ry="22" />
                  </defs>
                  <g fill="#D46B4F" fillRule="evenodd">
                    <use href="#hero-petal" transform="rotate(0 100 100)" />
                    <use href="#hero-petal" transform="rotate(45 100 100)" />
                    <use href="#hero-petal" transform="rotate(90 100 100)" />
                    <use href="#hero-petal" transform="rotate(135 100 100)" />
                  </g>
                </svg>
              </div>
              <h1 className="text-3xl sm:text-4xl font-serif font-light text-claude-text-200 mb-3 tracking-tight text-center">
                {greeting},{" "}
                <span className="relative inline-block pb-2">
                  Rajat
                  <svg
                    className="absolute w-[140%] h-[20px] -bottom-1 -left-[5%] text-claude-accent"
                    viewBox="0 0 140 24"
                    fill="none"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    <path d="M6 16 Q 70 24, 134 14" stroke="currentColor" strokeWidth="3" strokeLinecap="round" fill="none" />
                  </svg>
                </span>
              </h1>
              <p className="text-sm text-claude-text-400 mb-8 text-center max-w-md">
                Generate SQL queries from natural language or ask about SQL concepts.
              </p>

              <div className="w-full max-w-xl space-y-4">
                <div>
                  <div className="flex items-center gap-2 mb-2 justify-center">
                    <Zap className="w-3.5 h-3.5 text-claude-accent" />
                    <span className="text-xs font-medium text-claude-text-300 uppercase tracking-wider">Generate SQL</span>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      "Show all employees with salary above 50000",
                      "Get the average salary by department",
                      "Find departments with more than 10 employees",
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => handleSend({ message: q, files: [], model: MODELS[0].id, isThinkingEnabled: false })}
                        className="px-3 py-1.5 text-xs text-claude-text-300 border border-claude-bg-300 rounded-full hover:bg-claude-bg-200 hover:text-claude-text-200 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-2 justify-center">
                    <BookOpen className="w-3.5 h-3.5 text-blue-500" />
                    <span className="text-xs font-medium text-claude-text-300 uppercase tracking-wider">Learn SQL</span>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      "What is SQL?",
                      "DDL/DML/DCL/TCL",
                      "Explain JOINs",
                      "Window functions",
                      "What is CTE?",
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => handleSend({ message: q, files: [], model: MODELS[0].id, isThinkingEnabled: false })}
                        className="px-3 py-1.5 text-xs text-claude-text-300 border border-blue-200 rounded-full hover:bg-blue-50 hover:text-blue-600 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              {activeConv?.messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}

              {loading && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <div className={cn(
                    "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center",
                    loadingType === "explanation" ? "bg-blue-100" : "bg-claude-accent/10"
                  )}>
                    {loadingType === "explanation"
                      ? <BookOpen className="w-3.5 h-3.5 text-blue-600" />
                      : <Sparkles className="w-3.5 h-3.5 text-claude-accent" />
                    }
                  </div>
                  <div className="bg-claude-bg-100 border border-claude-bg-300/60 rounded-2xl rounded-bl-md px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <span className={cn("w-1.5 h-1.5 rounded-full animate-pulse", loadingType === "explanation" ? "bg-blue-400" : "bg-claude-accent/60")} style={{ animationDelay: "0ms" }} />
                        <span className={cn("w-1.5 h-1.5 rounded-full animate-pulse", loadingType === "explanation" ? "bg-blue-400" : "bg-claude-accent/60")} style={{ animationDelay: "200ms" }} />
                        <span className={cn("w-1.5 h-1.5 rounded-full animate-pulse", loadingType === "explanation" ? "bg-blue-400" : "bg-claude-accent/60")} style={{ animationDelay: "400ms" }} />
                      </div>
                      <span className="text-xs text-claude-text-400">
                        {loadingType === "explanation" ? "Looking up concept..." : "Generating SQL..."}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="flex-shrink-0 px-4 pb-4 pt-2">
          <ClaudeChatInput
            onSendMessage={handleSend}
            customModels={MODELS}
            defaultModel="lora-finetuned"
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
}
